import os
from pdf2image import convert_from_path
from PIL import Image
import json
import base64
import vertexai
import vertexai.generative_models as gm
from typing_extensions import TypedDict
from bounding_box_retry import get_improved_bounding_box
import time

"""
pip install pdf2image
brew install poppler
"""

SYSTEM_MESSAGE = """
You are a PDF parser. Your task is to analyze the provided PDF page (provided as an image) and return a structured JSON response containing all of the elements on the page.

Here are the element types you can use:
- NarrativeText
    - This is the main text content of the page, including paragraphs, lists, titles, and any other text content that is not part of a header, footer, figure, table, or image. Not all pages have narrative text, but most do.
- Figure
    - This covers charts, graphs, diagrams, etc. Associated titles, legends, axis titles, etc. should be considered to be part of the figure. Be sure your descriptions and bounding boxes fully capture these associated items, as they are essential for providing context to the figure. Not all pages have figures.
- Image
    - This is any visual content on the page that isn't a figure. Not all pages have images.
- Table
    - This is a table on the page. If the table can be represented accurately using Markdown, then it should be included as a Table element. If not, it should be included as an Image element to ensure accuracy.
- Header
    - This is the header of the page. You should never user more than one header element per page. Not all pages have a header.
- Footnote
    - This is a footnote on the page. Footnotes should always be included as a separate element from the main text content as they aren't part of the main linear reading flow of the page. Not all pages have footnotes.
- Footer
    - This is the footer of the page. You should never user more than one footer element per page. Not all pages have a footer, but when they do it is always the very last element on the page.

For Image and Figure elements ONLY, you must provide a detailed description of the image or figure. Do not transcribe the actual text contained in the image or figure. For all other element types, you must provide the text content.

Output format
- Your output should be an ordered (from top to bottom) list of elements on the page, where each element is a dictionary with the following keys:
    - type: str - the type of the element (e.g. "NarrativeText", "Figure", "Image", "Table", "Header", "Footnote", or "Footer")
    - content: str - the content of the element (ONLY include when type is "NarrativeText", "Table", "Header", "Footnote", or "Footer". For other element types, just use an empty string here). You can use Markdown formatting for text content. Always use Markdown for tables.
    - description: str (ONLY include when type is "Image" or "Figure". For other element types, just use an empty string here) - a detailed description of the image or figure.
    - bounding_box: list[int] (ONLY include when type is "Image" or "Figure". For other element types, just use an empty list here) - a bounding box around the image or figure, in the format [ymin, xmin, ymax, xmax].

Additional instructions
- Ignore background images or other images that don't convey any information.
- The element types described above are the only ones you are allowed to use.
- Be sure to include all page content in your response.
- Image and Figure elements MUST have accurate bounding boxes.
"""

class Element(TypedDict):
    type: str
    content: str
    image_bounding_box: list

response_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
            },
            "content": {
                "type": "string",
            },
            "description": {
                "type": "string",
            },
            "bounding_box": {
                "type": "array",
                "items": {
                    "type": "number",
                },
            },
        },
        "required": ["type", "content", "description", "bounding_box"],
    },
}

def pdf_to_images(pdf_path, output_folder, dpi=150):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Convert PDF to images
    images = convert_from_path(pdf_path, dpi=dpi)

    # Save each image
    for i, image in enumerate(images):
        image.save(os.path.join(output_folder, f'page_{i+1}.png'), 'PNG')

    print(f"Converted {len(images)} pages to images in {output_folder}")

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def make_llm_call_gemini(image_path: str, model: str = "gemini-1.5-pro-002", max_tokens: int = 4000) -> str:
    project_id = "brilliant-era-430616-a1"
    vertexai.init(project=project_id, location="us-central1")
    model = gm.GenerativeModel(model)
    generation_config = gm.GenerationConfig(temperature=0.0, max_output_tokens=max_tokens, response_mime_type="application/json", response_schema=response_schema)
    response = model.generate_content(
        [
            gm.Part.from_image(gm.Image.load_from_file(image_path)),
            SYSTEM_MESSAGE,
        ],
        generation_config=generation_config,
    )
    return response.text

def extract_images(page_image_file_path: str, page_number: int, bounding_boxes: list[list], output_folder: str, padding: int = 100):
    """
    Given a page image and a list of bounding boxes, extract the images from the bounding boxes by cropping the page image.
    - Leave a bit of extra padding around the provided bounding boxes to ensure that the entire content is captured.

    Inputs:
    - page_image_file_path: str, path to the page image
    - page_number: int, the page number (used for naming the extracted images)
    - bounding_boxes: list, list of bounding boxes in the format [ymin, xmin, ymax, xmax], where (xmin, ymin) is the top-left corner and (xmax, ymax) is the bottom-right corner and the top-left corner is (0, 0)
    - output_folder: str, path to the output folder where the extracted images will be saved
    - padding: int, padding around the VLM-generated bounding boxes to include extra content, provided in 1000x1000 coordinate space

    Outputs:
    - Saves the extracted images to the output folder
    """
    for bounding_box in bounding_boxes:
        ymin, xmin, ymax, xmax = bounding_box
        
        # Add some padding to the bounding box
        xmin = max(0, xmin - padding)
        ymin = max(0, ymin - padding)
        xmax = min(1000, xmax + padding)
        ymax = min(1000, ymax + padding)

        # Crop the image using the bounding box
        output_path = os.path.join(output_folder, f"image_page_{page_number}_bbox_{xmin}_{ymin}_{xmax}_{ymax}.png")
        crop_image(page_image_file_path, [ymin, xmin, ymax, xmax], output_path)

def crop_image(image_path, bounding_box, output_path=None):
    """
    Crops an image based on the provided bounding box and saves the cropped image.

    Inputs:
    - image_path (str): Path to the original image.
    - bounding_box (list of int): [ymin, xmin, ymax, xmax] coordinates scaled as if the image was 1000x1000.
    - output_path (str, optional): Path to save the cropped image. 
                                   If not provided, appends '_cropped' to the original filename.

    Outputs:
    - Saves the cropped image to the specified output path.
    """

    # Open the image
    with Image.open(image_path) as img:
        width, height = img.size
        print(f"Original image size: {width}x{height}")

        # Calculate actual pixel coordinates
        ymin_scaled, xmin_scaled, ymax_scaled, xmax_scaled = bounding_box
        actual_ymin = int(ymin_scaled / 1000 * height)
        actual_xmin = int(xmin_scaled / 1000 * width)
        actual_ymax = int(ymax_scaled / 1000 * height)
        actual_xmax = int(xmax_scaled / 1000 * width)

        # Ensure coordinates are within image bounds
        actual_ymin = max(0, actual_ymin)
        actual_xmin = max(0, actual_xmin)
        actual_ymax = min(height, actual_ymax)
        actual_xmax = min(width, actual_xmax)

        # Crop the image
        cropped_img = img.crop((actual_xmin, actual_ymin, actual_xmax, actual_ymax)) # the order is (left, top, right, bottom) for the crop function

        # Determine output path
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_cropped{ext}"

        # Save the cropped image
        cropped_img.save(output_path)
        print(f"Cropped image saved to: {output_path}")

def parse_page(image_path: str, page_number: int) -> list[dict]:
    """
    Given an image of a page, use LLM to extract the content of the page.

    Inputs:
    - image_path: str, path to the image of the page
    - page_number: int, the page number
    
    Outputs:
    - page_content: list of dictionaries, each containing information about an element on the page
    """
    llm_output = make_llm_call_gemini(image_path)
    try:
        page_content = json.loads(llm_output)
    except json.JSONDecodeError:
        print(f"Error decoding JSON response from LLM for {image_path}")
        print(llm_output)
        page_content = []

    # save images for each bounding box
    i = 0 # counter for the number of images extracted
    for element in page_content:
        if element["type"] in ["Image", "Figure"]:
            bounding_box = element["bounding_box"]

            # run the bounding box through the bounding box retry function to improve accuracy
            bounding_box = get_improved_bounding_box(image_path, bounding_box)
            element["improved_bounding_box"] = bounding_box

            print(f"Extracting image from bounding box: {bounding_box}")
            extract_images(image_path, page_number, [bounding_box], "extracted_images")

            # add image path to the element
            element["image_path"] = f"extracted_images/image_page_{page_number}_bbox_{i}.png"
            i += 1

    return page_content

def parse_file(pdf_path: str, image_folder_path: str) -> list[dict]:
    pdf_to_images(pdf_path, image_folder_path)
    image_file_names = os.listdir(image_folder_path)
    image_file_names = [f for f in image_file_names if f.endswith(".png")] # ignore any non-image files (like .DS_Store)
    sorted_image_file_names = sorted(image_file_names, key=lambda x: int(x.split("_")[1].split(".")[0])) # sort by page number
    all_page_content = []
    for i, image_path in sorted_image_file_names[0:5]:
        print (f"Processing {image_path}")
        image_path = os.path.join(image_folder_path, image_path)
        page_content = parse_page(image_path, page_number=i+1)
        all_page_content.extend(page_content)
        time.sleep(10) # sleep for 10 seconds to avoid rate limit issues with the Gemini API

    return all_page_content


if __name__ == "__main__":
    #pdf_path = '/Users/zach/Code/dsRAG/tests/data/levels_of_agi.pdf'
    #pdf_path = '/Users/zach/Code/GDOT-Tech Assessment - RFQ.pdf'
    pdf_path = "/Users/zach/Code/mck_energy.pdf"

    image_folder_path = 'pdf_to_images/mck_energy'
    #image_folder_path = 'pdf_to_images/levels_of_agi'

    all_page_content = parse_file(pdf_path, image_folder_path)

    #page_number = 24
    #image_path = f"/Users/zach/Code/pdf_to_images/mck_energy/page_{page_number}.png"
    #image_path = f"/Users/zach/Code/pdf_to_images/levels_of_agi/page_{page_number}.png"
    #all_page_content = parse_page(image_path, page_number)

    # save the extracted content to a JSON file
    output_file = "extracted_content_mck_energy.json"
    #output_file = "extracted_content_levels_of_agi.json"
    with open(output_file, "w") as f:
        json.dump(all_page_content, f, indent=2)