"""
Phase 1: PDF Text Extraction Module
Handles extraction of text from PDF files with layout preservation and cleaning.
Supports both native text extraction and OCR for scanned PDFs.
"""

import fitz  # PyMuPDF
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

from src.config import (
    RAW_PDFS_DIR,
    EXTRACTED_TEXT_DIR,
    MIN_PAGE_TEXT_LENGTH,
    OCR_ENABLED,
    OCR_LANGUAGES,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# TEXT CLEANING UTILITIES
# =============================================================================

def remove_headers_footers(text: str, common_phrases: List[str] = None) -> str:
    """
    Remove common headers and footers that appear repeatedly across pages.
    
    Args:
        text: The extracted text
        common_phrases: List of known header/footer phrases to remove
    
    Returns:
        Cleaned text
    """
    if common_phrases:
        for phrase in common_phrases:
            text = text.replace(phrase, "")
    
    # Remove common page number patterns (Arabic and English)
    text = re.sub(r'\n?\s*\d+\s*\n?\s*$', '', text)  # trailing page numbers
    text = re.sub(r'^\s*\d+\s*\n?', '', text)         # leading page numbers
    
    return text


def fix_broken_words(text: str) -> str:
    """
    Fix words broken at line breaks (hyphenation).
    
    Args:
        text: Text with potential broken words
    
    Returns:
        Text with fixed word breaks
    """
    # Fix hyphenated words at line breaks
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    text = re.sub(r'(\w)-\s+', r'\1', text)
    
    return text


def remove_extra_whitespace(text: str) -> str:
    """
    Normalize whitespace while preserving paragraph structure.
    
    Args:
        text: Raw extracted text
    
    Returns:
        Cleaned text
    """
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    # Replace more than 2 newlines with 2 newlines (preserve paragraphs)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def normalize_arabic_text(text: str, remove_tashkeel: bool = True) -> str:
    """
    Normalize Arabic text by standardizing characters.
    
    Args:
        text: Arabic text to normalize
        remove_tashkeel: Whether to remove diacritics (tashkeel)
    
    Returns:
        Normalized Arabic text
    """
    if remove_tashkeel:
        # Remove tashkeel (diacritics)
        tashkeel_pattern = re.compile(r'[\u064B-\u0652\u0640]')
        text = tashkeel_pattern.sub('', text)
    
    # Normalize Arabic alef variants
    text = text.replace('\u0623', '\u0627')  # أ -> ا
    text = text.replace('\u0625', '\u0627')  # إ -> ا
    text = text.replace('\u0622', '\u0627')  # آ -> ا
    
    # Normalize Arabic ya variants
    text = text.replace('\u0649', '\u064A')  # ى -> ي
    
    return text


def clean_extracted_text(text: str, is_arabic: bool = True) -> str:
    """
    Apply all cleaning steps to extracted text.
    
    Args:
        text: Raw extracted text
        is_arabic: Whether text contains Arabic content
    
    Returns:
        Fully cleaned text
    """
    if not text or not text.strip():
        return ""
    
    # Step 1: Remove headers and footers
    text = remove_headers_footers(text)
    
    # Step 2: Fix broken words
    text = fix_broken_words(text)
    
    # Step 3: Remove extra whitespace
    text = remove_extra_whitespace(text)
    
    # Step 4: Normalize Arabic text if applicable
    if is_arabic:
        text = normalize_arabic_text(text, remove_tashkeel=True)
    
    return text


# =============================================================================
# PDF EXTRACTION
# =============================================================================

def is_scanned_pdf(doc: fitz.Document, sample_pages: int = 3) -> bool:
    """
    Detect if a PDF is scanned (image-based) vs text-based.
    
    Args:
        doc: PyMuPDF document object
        sample_pages: Number of pages to sample
    
    Returns:
        True if PDF appears to be scanned, False otherwise
    """
    pages_to_check = min(sample_pages, len(doc))
    text_ratios = []
    
    for page_num in range(pages_to_check):
        page = doc[page_num]
        
        # Get text length
        text = page.get_text("text")
        text_length = len(text.strip())
        
        # Get page dimensions
        rect = page.rect
        page_area = rect.width * rect.height
        
        # Estimate text density (characters per unit area)
        if page_area > 0:
            text_density = text_length / page_area
            text_ratios.append(text_density)
    
    # If average text density is very low, likely a scanned PDF
    avg_density = sum(text_ratios) / len(text_ratios) if text_ratios else 0
    
    # Threshold: less than 0.001 characters per point² suggests scanned
    return avg_density < 0.001


def extract_text_from_pdf(
    pdf_path: str,
    output_dir: str = None,
    clean: bool = True,
    is_arabic: bool = True
) -> List[Dict]:
    """
    Extract text from a PDF file with metadata preservation.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted text files
        clean: Whether to apply text cleaning
        is_arabic: Whether text contains Arabic content
    
    Returns:
        List of dictionaries containing extracted text and metadata
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    logger.info(f"Processing PDF: {pdf_path.name}")
    
    doc = fitz.open(str(pdf_path))
    filename = pdf_path.stem
    extracted_data = []
    
    # Check if PDF is scanned
    scanned = is_scanned_pdf(doc)
    if scanned:
        logger.warning(f"PDF appears to be scanned: {pdf_path.name}")
        if OCR_ENABLED:
            logger.info("OCR is enabled, proceeding with OCR extraction")
            doc.close()
            return extract_with_ocr(str(pdf_path), output_dir)
        else:
            logger.warning("OCR is disabled. Attempting PyMuPDF image-text fallback...")
    
    # Extract text from each page
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        
        # If page has very little text, try extracting from images/blocks
        if len(text.strip()) < MIN_PAGE_TEXT_LENGTH:
            # Try "blocks" extraction which can get text from embedded images
            blocks = page.get_text("blocks")
            block_texts = [b[4] for b in blocks if b[6] == 0]  # type 0 = text blocks
            block_text = "\n".join(block_texts)
            if len(block_text.strip()) > len(text.strip()):
                text = block_text
            
            # Still empty? Try "dict" extraction for structured text
            if len(text.strip()) < MIN_PAGE_TEXT_LENGTH:
                try:
                    page_dict = page.get_text("dict")
                    dict_texts = []
                    for block in page_dict.get("blocks", []):
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                if span.get("text", "").strip():
                                    dict_texts.append(span["text"])
                    dict_text = " ".join(dict_texts)
                    if len(dict_text.strip()) > len(text.strip()):
                        text = dict_text
                except Exception:
                    pass
        
        # Clean text if requested
        if clean:
            text = clean_extracted_text(text, is_arabic=is_arabic)
        else:
            text = text.strip()
            text = ' '.join(text.split())
        
        # Filter out empty/minimal pages
        if len(text) >= MIN_PAGE_TEXT_LENGTH:
            extracted_data.append({
                "text": text,
                "page_number": page_num + 1,
                "source_file": filename,
                "total_pages": len(doc),
                "extraction_method": "native" if not scanned else "ocr_fallback"
            })
    
    doc.close()
    
    logger.info(f"Extracted {len(extracted_data)} pages from {pdf_path.name}")
    
    # Merge adjacent pages to avoid losing info at page boundaries
    extracted_data = merge_adjacent_pages(extracted_data)
    
    # Save to file if output directory is specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{filename}_extracted.json"
        
        import json
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved extracted text to: {output_file}")
    
    return extracted_data


def extract_with_ocr(pdf_path: str, output_dir: str = None) -> List[Dict]:
    """
    Extract text from scanned PDFs using OCR.
    
    Note: Requires pytesseract and pdf2image to be installed.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted text files
    
    Returns:
        List of dictionaries containing extracted text and metadata
    """
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError:
        logger.error("OCR dependencies not installed. Install with: pip install pytesseract pdf2image")
        raise ImportError("pytesseract and pdf2image are required for OCR")
    
    pdf_path = Path(pdf_path)
    logger.info(f"Processing with OCR: {pdf_path.name}")
    
    images = convert_from_path(str(pdf_path))
    text_data = []
    
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image, lang=OCR_LANGUAGES)
        text = clean_extracted_text(text, is_arabic=True)
        
        if len(text) >= MIN_PAGE_TEXT_LENGTH:
            text_data.append({
                "text": text,
                "page_number": i + 1,
                "source_file": pdf_path.stem,
                "total_pages": len(images),
                "extraction_method": "OCR"
            })
    
    logger.info(f"OCR extracted {len(text_data)} pages from {pdf_path.name}")
    
    # Save to file if output directory is specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        output_file = output_dir / f"{pdf_path.stem}_extracted.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(text_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved OCR text to: {output_file}")
    
    return text_data


def merge_adjacent_pages(extracted_data: List[Dict]) -> List[Dict]:
    """
    Merge adjacent pages from the same file into overlapping groups.
    This ensures info spanning 2 pages isn't lost during chunking.
    
    For pages [1,2,3,4], creates merged entries:
    - Page 1 (original)
    - Pages 1+2 (merged)
    - Page 2 (original) 
    - Pages 2+3 (merged)
    - etc.
    """
    if len(extracted_data) <= 1:
        return extracted_data
    
    result = []
    
    for i, item in enumerate(extracted_data):
        # Add original page
        result.append(item)
        
        # Create a merged entry with the next page (if same file)
        if i + 1 < len(extracted_data):
            next_item = extracted_data[i + 1]
            if item.get('source_file') == next_item.get('source_file'):
                merged = {
                    'text': item['text'] + '\n\n' + next_item['text'],
                    'page_number': item['page_number'],
                    'source_file': item['source_file'],
                    'total_pages': item.get('total_pages', 0),
                    'extraction_method': item.get('extraction_method', 'native'),
                    'merged_pages': f"{item['page_number']}-{next_item['page_number']}",
                }
                result.append(merged)
    
    logger.info(f"Page merging: {len(extracted_data)} pages -> {len(result)} entries (with overlaps)")
    return result


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_all_pdfs(
    input_dir: str = None,
    output_dir: str = None,
    clean: bool = True,
    is_arabic: bool = True
) -> List[Dict]:
    """
    Process all PDF files in a directory.
    
    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory to save extracted text files
        clean: Whether to apply text cleaning
        is_arabic: Whether text contains Arabic content
    
    Returns:
        Combined list of all extracted data from all PDFs
    """
    input_dir = Path(input_dir) if input_dir else RAW_PDFS_DIR
    output_dir = Path(output_dir) if output_dir else EXTRACTED_TEXT_DIR
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    all_data = []
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return all_data
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        try:
            data = extract_text_from_pdf(
                str(pdf_file),
                output_dir=str(output_dir),
                clean=clean,
                is_arabic=is_arabic
            )
            all_data.extend(data)
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {str(e)}")
            continue
    
    logger.info(f"Total extracted pages across all PDFs: {len(all_data)}")
    return all_data


# =============================================================================
# QUALITY GATE 1
# =============================================================================

def validate_extraction(extracted_data: List[Dict]) -> Dict:
    """
    Validate extracted data against Quality Gate 1 criteria.
    
    Args:
        extracted_data: List of extracted page data
    
    Returns:
        Validation report dictionary
    """
    report = {
        "total_pages": len(extracted_data),
        "passed": True,
        "checks": {},
        "issues": []
    }
    
    # Check 1: No empty extracted data
    check1 = len(extracted_data) > 0
    report["checks"]["has_data"] = check1
    if not check1:
        report["passed"] = False
        report["issues"].append("No data was extracted")
    
    # Check 2: No empty pages
    empty_pages = [i for i, page in enumerate(extracted_data) if not page.get("text") or len(page["text"].strip()) == 0]
    check2 = len(empty_pages) == 0
    report["checks"]["no_empty_pages"] = check2
    report["empty_pages_count"] = len(empty_pages)
    if not check2:
        report["issues"].append(f"Found {len(empty_pages)} empty pages")
    
    # Check 3: All pages have metadata
    missing_metadata = []
    for i, page in enumerate(extracted_data):
        if not page.get("source_file") or not page.get("page_number"):
            missing_metadata.append(i)
    check3 = len(missing_metadata) == 0
    report["checks"]["metadata_complete"] = check3
    if not check3:
        report["issues"].append(f"Found {len(missing_metadata)} pages with missing metadata")
    
    # Check 4: Page numbers are valid
    invalid_pages = [i for i, page in enumerate(extracted_data) if page.get("page_number", 0) < 1]
    check4 = len(invalid_pages) == 0
    report["checks"]["valid_page_numbers"] = check4
    if not check4:
        report["issues"].append(f"Found {len(invalid_pages)} pages with invalid page numbers")
    
    # Summary stats
    text_lengths = [len(page["text"]) for page in extracted_data]
    report["avg_text_length"] = sum(text_lengths) / len(text_lengths) if text_lengths else 0
    report["min_text_length"] = min(text_lengths) if text_lengths else 0
    report["max_text_length"] = max(text_lengths) if text_lengths else 0
    
    if not all(report["checks"].values()):
        report["passed"] = False
    
    return report


# =============================================================================
# MAIN (for direct execution)
# =============================================================================

if __name__ == "__main__":
    # Example usage
    logger.info("Starting PDF extraction...")
    
    # Process all PDFs in the raw directory
    all_data = process_all_pdfs()
    
    # Run validation
    validation_report = validate_extraction(all_data)
    
    logger.info("\n" + "="*50)
    logger.info("EXTRACTION VALIDATION REPORT")
    logger.info("="*50)
    logger.info(f"Total pages extracted: {validation_report['total_pages']}")
    logger.info(f"Average text length: {validation_report['avg_text_length']:.0f} chars")
    logger.info(f"All checks passed: {validation_report['passed']}")
    
    if validation_report["issues"]:
        logger.info(f"Issues found: {len(validation_report['issues'])}")
        for issue in validation_report["issues"]:
            logger.info(f"  - {issue}")
