import fitz  # PyMuPDF
import requests
from typing import List, Tuple
def extract_text_from_pdf(url: str) -> str:
    response = requests.get(url)
    with open("temp.pdf", "wb") as f:
        f.write(response.content)

    doc = fitz.open("temp.pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    return full_text
