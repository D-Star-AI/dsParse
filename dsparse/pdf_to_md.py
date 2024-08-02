import os
from pdf2image import convert_from_path
from openai import OpenAI
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig, Image

"""
pip install pdf2image
brew install poppler
"""

SYSTEM_MESSAGE = """
Convert the following PDF page to markdown. 
Return only the markdown with no explanation text. 
Do not exclude any content from the page.
"""

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

def make_llm_call_openai(image_path: str, model: str = "gpt-4o-mini", max_tokens: int = 2000) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    base64_image = encode_image(image_path)
    chat_messages = [
        {
            "role": "system", 
            "content": SYSTEM_MESSAGE
        }, 
        {
            "role": "user", 
            "content": [
                {
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    
    response = client.chat.completions.create(
        model=model,
        messages=chat_messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    llm_output = response.choices[0].message.content.strip()
    return llm_output

def make_llm_call_gemini(image_path: str, model: str = "gemini-1.5-flash-001", max_tokens: int = 2000) -> str:
    project_id = "brilliant-era-430616-a1"
    vertexai.init(project=project_id, location="us-central1")
    model = GenerativeModel(model)
    generation_config = GenerationConfig(temperature=0.0, max_output_tokens=max_tokens)
    response = model.generate_content(
        [
            Part.from_image(Image.load_from_file(image_path)),
            SYSTEM_MESSAGE,
        ],
        generation_config=generation_config,
    )
    return response.text

def images_to_markdown(image_folder_path: str, save_path: str = "output.md"):
    markdown_str = ""
    image_file_names = os.listdir(image_folder_path)
    # sort by page number
    image_file_names = sorted(image_file_names, key=lambda x: int(x.split("_")[1].split(".")[0]))
    for image_path in image_file_names[:5]:
        print (f"Processing {image_path}")
        image_path = os.path.join(image_folder_path, image_path)
        #llm_output = make_llm_call_openai(image_path)
        llm_output = make_llm_call_gemini(image_path)
        markdown_str += llm_output + "\n\n"

    # remove markdown artifacts
    markdown_str = remove_markdown_artifacts(markdown_str)

    # save to markdown file
    with open(save_path, "w") as f:
        f.write(markdown_str)

def remove_markdown_artifacts(text: str) -> str:
    # remove ```markdown and ```
    text = text.replace("```markdown", "").replace("```", "")

    # remove double (or more) newlines
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    # remove leading and trailing whitespaces
    text = text.strip()
    return text


#pdf_path = '/Users/zach/Code/dsRAG/tests/data/levels_of_agi.pdf'
pdf_path = '/Users/zach/Code/GDOT-Tech Assessment - RFQ.pdf'
output_folder = 'pdf_to_images_output'
save_path = 'GDOT_gemini.md'
#pdf_to_images(pdf_path, output_folder)
images_to_markdown(output_folder, save_path)

with open(save_path, "r") as f:
    text = f.read()
    text = remove_markdown_artifacts(text)
    # save to markdown file
    with open(save_path, "w") as f:
        f.write(text)