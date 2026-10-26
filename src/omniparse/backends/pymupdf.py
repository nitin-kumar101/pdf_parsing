import fitz
from ..models import Page

def parse_pdf(path: str) -> list[Page]:
    with fitz.open(path) as doc:
        return [Page(number=i + 1, text=page.get_text(sort=True), engine="pymupdf") for i, page in enumerate(doc)]
