"""
Resume Parser Module.
Extracts raw text from PDF and TXT files, supporting file paths and byte streams.
Uses PyMuPDF (fitz) and PyPDF as robust fallback engines.
"""

import os
import io
from typing import Union, Tuple, Optional
import pymupdf
import pypdf


def extract_text_from_pdf(source: Union[str, bytes, io.BytesIO]) -> Tuple[str, Optional[str]]:
    """
    Extracts text from a PDF given a filepath, raw bytes, or BytesIO buffer.
    Returns (extracted_text, error_message).
    """
    text = ""
    error = None

    # First attempt: PyMuPDF (fast, high fidelity)
    try:
        if isinstance(source, (bytes, bytearray)):
            doc = pymupdf.open(stream=source, filetype="pdf")
        elif isinstance(source, io.BytesIO):
            source.seek(0)
            doc = pymupdf.open(stream=source.read(), filetype="pdf")
        elif isinstance(source, str):
            if not os.path.exists(source):
                return "", f"File does not exist: {source}"
            doc = pymupdf.open(source)
        else:
            return "", "Unsupported PDF source type"

        pages_text = []
        for page in doc:
            pages_text.append(page.get_text("text"))
        doc.close()
        text = "\n".join(pages_text).strip()

        if text:
            return text, None
    except Exception as e:
        error = f"PyMuPDF failed: {str(e)}"

    # Second attempt: PyPDF fallback
    try:
        if isinstance(source, (bytes, bytearray)):
            reader = pypdf.PdfReader(io.BytesIO(source))
        elif isinstance(source, io.BytesIO):
            source.seek(0)
            reader = pypdf.PdfReader(source)
        elif isinstance(source, str):
            reader = pypdf.PdfReader(source)
        else:
            return "", "Unsupported PDF source type"

        pages_text = []
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                pages_text.append(extracted)
        fallback_text = "\n".join(pages_text).strip()

        if fallback_text:
            return fallback_text, None
        elif not text:
            return "", error or "No text could be extracted from PDF (may be image-only/scanned)."
    except Exception as e:
        return text, f"PyPDF fallback also failed: {str(e)}"

    return text, error


def extract_text_from_txt(source: Union[str, bytes, io.BytesIO]) -> Tuple[str, Optional[str]]:
    """
    Extracts text from a TXT source given a filepath, bytes, or BytesIO buffer.
    Handles UTF-8, Latin-1, and Windows-1252 encodings.
    """
    try:
        if isinstance(source, (bytes, bytearray)):
            raw_bytes = source
        elif isinstance(source, io.BytesIO):
            source.seek(0)
            raw_bytes = source.read()
        elif isinstance(source, str):
            if not os.path.exists(source):
                return "", f"File does not exist: {source}"
            with open(source, "rb") as f:
                raw_bytes = f.read()
        else:
            return "", "Unsupported TXT source type"

        # Try encodings in order
        for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                return raw_bytes.decode(enc).strip(), None
            except UnicodeDecodeError:
                continue

        return raw_bytes.decode("utf-8", errors="replace").strip(), None
    except Exception as e:
        return "", f"Failed to read text file: {str(e)}"


def parse_resume(source: Union[str, bytes, io.BytesIO], filename: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    High-level resume parser. Detects format (PDF or TXT) and returns extracted raw text.
    """
    # Detect file type from filename if provided or from path string
    file_type = ""
    if filename:
        file_type = filename.lower()
    elif isinstance(source, str):
        file_type = source.lower()

    if file_type.endswith(".pdf"):
        return extract_text_from_pdf(source)
    elif file_type.endswith(".txt") or file_type.endswith(".text") or file_type.endswith(".md"):
        return extract_text_from_txt(source)

    # Magic byte check if source is bytes or BytesIO
    if isinstance(source, (bytes, bytearray, io.BytesIO)):
        if isinstance(source, io.BytesIO):
            source.seek(0)
            head = source.read(5)
            source.seek(0)
        else:
            head = source[:5]

        if head.startswith(b"%PDF"):
            return extract_text_from_pdf(source)
        else:
            return extract_text_from_txt(source)

    # Default fallback
    return extract_text_from_txt(source)
