import PyPDF2
import docx2txt

""" Basic file parsing functions """
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)

        # Initialize an empty string to store the extracted text
        extracted_text = ""

        # Loop through all the pages in the PDF file
        for page_num in range(len(pdf_reader.pages)):
            # Extract the text from the current page
            page_text = pdf_reader.pages[page_num].extract_text()

            # Add the extracted text to the final text
            extracted_text += page_text

    return extracted_text


def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)


def parse_file(file_path: str, file_or_url: str) -> str:
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    elif file_path.endswith(".txt") or file_path.endswith(".md"):
        with open(file_path, "r") as f:
            return f.read()
    else:
        raise ValueError("File must be a .pdf, .txt, .md or .docx file")