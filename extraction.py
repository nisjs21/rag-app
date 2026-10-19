import pymupdf
import re

def extract_text_from_pdf(path: str) -> str:

    full_text = []

    with pymupdf.open(path) as pdf:
        for page_num in range(len(pdf)):

            page = pdf.load_page(page_num)
            page_text = page.get_text("text")

            page_text = re.sub(r'\s+', ' ', page_text)

            

            full_text.append(page_text)

    text = " ".join(full_text).strip()

    return text

