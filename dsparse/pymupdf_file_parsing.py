import pymupdf4llm

def extract_text_pymupdf4llm(pdf_path: str) -> str:
    md_text = pymupdf4llm.to_markdown(pdf_path, write_images=False)
    return md_text

if __name__ == "__main__":
    pdf_path = "/Users/zach/Code/mck_energy.pdf"
    md_text = extract_text_pymupdf4llm(pdf_path)