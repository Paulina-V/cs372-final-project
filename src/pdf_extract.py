"""
Module for extracting text from medical bill PDFs and images.
Supports text-based PDFs, images, and plain text demo bills.

AI-assisted portions of this file are documented in ATTRIBUTION.md.
"""

from pypdf import PdfReader
from PIL import Image
import pytesseract
import os


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file. Falls back to OCR if text extraction yields little content."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"

    # If we got very little text, the PDF might be scanned; try OCR.
    if len(text.strip()) < 50:
        text = ocr_pdf(pdf_path)

    return text.strip()


def extract_text_from_image(image_path: str) -> str:
    """Extract text from an image file using Tesseract OCR."""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text.strip()


def extract_text_from_txt(txt_path: str) -> str:
    """Read text directly from a plain text bill."""
    with open(txt_path, "r", encoding="utf-8") as file:
        return file.read().strip()


def ocr_pdf(pdf_path: str) -> str:
    """OCR a scanned PDF by converting pages to images first."""
    try:
        from pdf2image import convert_from_path

        images = convert_from_path(pdf_path)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img) + "\n"
        return text.strip()
    except ImportError:
        return "[OCR fallback unavailable: install pdf2image, poppler, and tesseract]"
    except Exception as exc:
        raise RuntimeError(
            "OCR fallback failed. For scanned PDFs, install Poppler and Tesseract "
            "or upload a text-based PDF, image, or .txt bill."
        ) from exc


def extract_text(file_path: str) -> str:
    """Route to the correct extraction method based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    elif ext in [".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"]:
        return extract_text_from_image(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
