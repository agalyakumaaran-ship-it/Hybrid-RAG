
from pathlib import Path
from pypdf import PdfReader
from docx import Document


def read_document(file_path: str) -> str:
    """Extract text from a PDF, TXT, or DOCX document."""

    path = Path(file_path)
    extension = path.suffix.lower()

    if not path.is_file():
        raise FileNotFoundError(
            f"Document not found: {file_path}"
        )

    if extension == ".pdf":
        reader = PdfReader(str(path))
        text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    elif extension == ".txt":
        text = path.read_text(encoding="utf-8-sig")

    elif extension == ".docx":
        document = Document(str(path))
        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

    else:
        raise ValueError(
            "Unsupported file type. Use PDF, TXT, or DOCX."
        )

    if not text.strip():
        raise ValueError(
            "No readable text found in this document."
        )

    return text.strip()
