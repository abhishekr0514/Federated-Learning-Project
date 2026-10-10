"""Text Extraction and OCR Abstraction Service.

SCIENTIFIC & PRIVACY INVARIANTS:
1. Honest Extraction Capabilities:
   - DIGITAL_PDF_TEXT: Extracts embedded digital text via pdfplumber.
   - SCANNED_DOCUMENT_OCR / IMAGE_OCR: If local OCR engine (Tesseract) is unavailable on host,
     the status MUST be reported as NOT_IMPLEMENTED. Never fabricate or fake OCR text.
2. Zero Central Database Storage:
   - Raw extracted text must remain inside the in-memory processing worker or local protected storage.
   - Central PostgreSQL must NEVER receive or store raw unmasked text.
"""

import io
import shutil
from typing import Any, Dict, Optional
from PIL import Image

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import pytesseract
except ImportError:
    pytesseract = None


class TextExtractor:
    """Multi-modal document text extractor."""

    @staticmethod
    def _is_tesseract_binary_available() -> bool:
        """Check if tesseract CLI executable exists on system PATH."""
        if pytesseract is None:
            return False
        return shutil.which("tesseract") is not None

    @classmethod
    def extract_text(
        cls,
        content: bytes,
        extension: str,
    ) -> Dict[str, Any]:
        """Extract text from document content.
        
        Returns:
            Dict containing:
                - status: 'SUCCESS', 'NOT_IMPLEMENTED', 'EMPTY_TEXT', or 'FAILED'
                - extraction_mode: 'DIGITAL_PDF_TEXT', 'SCANNED_DOCUMENT_OCR', or 'IMAGE_OCR'
                - text: In-memory extracted string (MUST NOT BE PERSISTED TO CENTRAL POSTGRESQL)
                - page_count: int
                - character_count: int
                - message: str
        """
        clean_ext = extension.lower().lstrip(".")

        if clean_ext == "pdf":
            return cls._extract_pdf(content)
        elif clean_ext in ["jpg", "jpeg", "png"]:
            return cls._extract_image(content)
        else:
            return {
                "status": "FAILED",
                "extraction_mode": "UNKNOWN",
                "text": "",
                "page_count": 0,
                "character_count": 0,
                "message": f"Unsupported extension for text extraction: {clean_ext}",
            }

    @classmethod
    def _extract_pdf(cls, content: bytes) -> Dict[str, Any]:
        """Extract text from PDF, first attempting digital text, then OCR if digital text is absent."""
        if pdfplumber is None:
            return {
                "status": "FAILED",
                "extraction_mode": "DIGITAL_PDF_TEXT",
                "text": "",
                "page_count": 0,
                "character_count": 0,
                "message": "pdfplumber library not installed.",
            }

        extracted_pages = []
        page_count = 0
        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        extracted_pages.append(page_text.strip())

            full_text = "\n\n".join(extracted_pages).strip()

            if len(full_text) > 0:
                return {
                    "status": "SUCCESS",
                    "extraction_mode": "DIGITAL_PDF_TEXT",
                    "text": full_text,
                    "page_count": page_count,
                    "character_count": len(full_text),
                    "message": "Digital PDF embedded text extracted successfully.",
                }

            # If digital text is empty, check for scanned OCR availability
            if not cls._is_tesseract_binary_available():
                return {
                    "status": "NOT_IMPLEMENTED",
                    "extraction_mode": "SCANNED_DOCUMENT_OCR",
                    "text": "",
                    "page_count": page_count,
                    "character_count": 0,
                    "message": (
                        "PDF contains scanned raster pages with no embedded text, and "
                        "local Tesseract OCR engine is NOT_IMPLEMENTED / unavailable on this host."
                    ),
                }

            # If Tesseract binary were available
            ocr_pages = []
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page in pdf.pages:
                    pil_img = page.to_image(resolution=300).original
                    ocr_text = pytesseract.image_to_string(pil_img)
                    if ocr_text:
                        ocr_pages.append(ocr_text.strip())

            full_ocr_text = "\n\n".join(ocr_pages).strip()
            return {
                "status": "SUCCESS" if len(full_ocr_text) > 0 else "EMPTY_TEXT",
                "extraction_mode": "SCANNED_DOCUMENT_OCR",
                "text": full_ocr_text,
                "page_count": page_count,
                "character_count": len(full_ocr_text),
                "message": "Scanned PDF OCR completed.",
            }

        except Exception as e:
            return {
                "status": "FAILED",
                "extraction_mode": "DIGITAL_PDF_TEXT",
                "text": "",
                "page_count": page_count,
                "character_count": 0,
                "message": f"PDF text extraction failed: {str(e)}",
            }

    @classmethod
    def _extract_image(cls, content: bytes) -> Dict[str, Any]:
        """Extract text from image using local OCR engine."""
        if not cls._is_tesseract_binary_available():
            return {
                "status": "NOT_IMPLEMENTED",
                "extraction_mode": "IMAGE_OCR",
                "text": "",
                "page_count": 1,
                "character_count": 0,
                "message": (
                    "Image OCR capability is NOT_IMPLEMENTED because Tesseract binary is "
                    "not installed on this host environment."
                ),
            }

        try:
            img = Image.open(io.BytesIO(content))
            ocr_text = pytesseract.image_to_string(img).strip()
            return {
                "status": "SUCCESS" if len(ocr_text) > 0 else "EMPTY_TEXT",
                "extraction_mode": "IMAGE_OCR",
                "text": ocr_text,
                "page_count": 1,
                "character_count": len(ocr_text),
                "message": "Image OCR completed successfully.",
            }
        except Exception as e:
            return {
                "status": "FAILED",
                "extraction_mode": "IMAGE_OCR",
                "text": "",
                "page_count": 1,
                "character_count": 0,
                "message": f"Image OCR extraction error: {str(e)}",
            }
