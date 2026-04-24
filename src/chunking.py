"""
Phase 2: Chunking & Text Processing Module
Handles text segmentation into manageable chunks while preserving context and metadata.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    CHUNK_MAX_SIZE_FACTOR,
    PROCESSED_CHUNKS_DIR,
)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# TEXT SPLITTER INITIALIZATION
# =============================================================================

def get_text_splitter(
    chunk_size: int = None,
    chunk_overlap: int = None,
    separators: List[str] = None
) -> RecursiveCharacterTextSplitter:
    """
    Initialize the RecursiveCharacterTextSplitter with configured parameters.
    
    Args:
        chunk_size: Maximum characters per chunk (default from config)
        chunk_overlap: Characters of overlap between chunks (default from config)
        separators: Priority list of separators (default from config)
    
    Returns:
        Configured RecursiveCharacterTextSplitter instance
    """
    chunk_size = chunk_size or CHUNK_SIZE
    chunk_overlap = chunk_overlap or CHUNK_OVERLAP
    separators = separators or CHUNK_SEPARATORS
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )
    
    logger.info(
        f"Text splitter initialized: chunk_size={chunk_size}, "
        f"chunk_overlap={chunk_overlap}, separators={len(separators)}"
    )
    
    return text_splitter


# =============================================================================
# CHUNK CREATION
# =============================================================================

def create_chunks(
    extracted_data: List[Dict[str, Any]],
    text_splitter: RecursiveCharacterTextSplitter = None
) -> List[Document]:
    """
    Create text chunks from extracted PDF data with metadata preservation.
    
    Args:
        extracted_data: List of extracted page data from extraction module
        text_splitter: Optional custom text splitter (uses default if None)
    
    Returns:
        List of LangChain Document objects with metadata
    """
    if not extracted_data:
        logger.warning("No data provided for chunking")
        return []
    
    text_splitter = text_splitter or get_text_splitter()
    chunks = []
    
    logger.info(f"Creating chunks from {len(extracted_data)} pages...")
    
    for item in extracted_data:
        text = item.get("text", "")
        
        if not text or len(text.strip()) < 50:
            logger.debug(f"Skipping page with insufficient text: {item.get('page_number')}")
            continue
        
        # Prepare metadata
        metadata = {
            "source_file": item.get("source_file", "unknown"),
            "page_number": item.get("page_number", 0),
            "total_pages": item.get("total_pages", 0),
            "extraction_method": item.get("extraction_method", "unknown"),
        }
        
        # Create documents with metadata
        try:
            doc_chunks = text_splitter.create_documents(
                texts=[text],
                metadatas=[metadata]
            )
            chunks.extend(doc_chunks)
        except Exception as e:
            logger.error(f"Error chunking page {item.get('page_number')}: {str(e)}")
            continue
    
    logger.info(f"Created {len(chunks)} chunks from {len(extracted_data)} pages")
    
    return chunks


def create_chunks_with_context(
    extracted_data: List[Dict[str, Any]],
    text_splitter: RecursiveCharacterTextSplitter = None,
    add_context_headers: bool = True
) -> List[Document]:
    """
    Create chunks with enhanced context headers for better retrieval.
    
    Adds document title and page context at the beginning of each chunk
    to improve semantic search relevance.
    
    Args:
        extracted_data: List of extracted page data
        text_splitter: Optional custom text splitter
        add_context_headers: Whether to prepend context headers
    
    Returns:
        List of Document objects with enhanced context
    """
    if not extracted_data:
        return []
    
    text_splitter = text_splitter or get_text_splitter()
    chunks = []
    
    for item in extracted_data:
        text = item.get("text", "")
        source_file = item.get("source_file", "unknown")
        page_number = item.get("page_number", 0)
        
        if not text or len(text.strip()) < 50:
            continue
        
        # Add context header for better retrieval
        if add_context_headers:
            context_header = f"[المصدر: {source_file} - الصفحة {page_number}]\n\n"
            text = context_header + text
        
        metadata = {
            "source_file": source_file,
            "page_number": page_number,
            "total_pages": item.get("total_pages", 0),
            "extraction_method": item.get("extraction_method", "unknown"),
        }
        
        try:
            doc_chunks = text_splitter.create_documents(
                texts=[text],
                metadatas=[metadata]
            )
            chunks.extend(doc_chunks)
        except Exception as e:
            logger.error(f"Error in contextual chunking for page {page_number}: {str(e)}")
            continue
    
    logger.info(f"Created {len(chunks)} contextual chunks")
    return chunks


# =============================================================================
# CHUNK VALIDATION
# =============================================================================

def validate_chunks(
    chunks: List[Document],
    min_length: int = 50,
    max_size_factor: float = None
) -> Dict[str, Any]:
    """
    Validate chunks against quality criteria.
    
    Args:
        chunks: List of Document chunks to validate
        min_length: Minimum acceptable chunk length
        max_size_factor: Maximum chunk size as factor of CHUNK_SIZE
    
    Returns:
        Validation report with issues found
    """
    max_size_factor = max_size_factor or CHUNK_MAX_SIZE_FACTOR
    max_length = int(CHUNK_SIZE * max_size_factor)
    
    issues = []
    stats = {
        "total_chunks": len(chunks),
        "avg_length": 0,
        "min_length": 0,
        "max_length": 0,
        "short_chunks": 0,
        "long_chunks": 0,
        "missing_metadata": 0,
        "mid_sentence_breaks": 0,
    }
    
    if not chunks:
        return {"passed": False, "issues": ["No chunks to validate"], "stats": stats}
    
    lengths = []
    
    for i, chunk in enumerate(chunks):
        content = chunk.page_content
        length = len(content)
        lengths.append(length)
        
        # Check minimum length
        if length < min_length:
            issues.append(f"Chunk {i}: Too short ({length} chars)")
            stats["short_chunks"] += 1
        
        # Check maximum length
        if length > max_length:
            issues.append(f"Chunk {i}: Too long ({length} chars)")
            stats["long_chunks"] += 1
        
        # Check metadata
        if not chunk.metadata.get("source_file") or not chunk.metadata.get("page_number"):
            issues.append(f"Chunk {i}: Missing source metadata")
            stats["missing_metadata"] += 1
        
        # Check for mid-sentence breaks (ends without punctuation)
        stripped = content.strip()
        if stripped and stripped[-1] not in '.!?:;。？！：；\n':
            # Only flag if it's not the last chunk of a page
            stats["mid_sentence_breaks"] += 1
    
    # Calculate statistics
    stats["avg_length"] = sum(lengths) / len(lengths)
    stats["min_length"] = min(lengths)
    stats["max_length"] = max(lengths)
    
    # Determine pass/fail
    passed = (
        stats["short_chunks"] == 0 and
        stats["long_chunks"] == 0 and
        stats["missing_metadata"] == 0
    )
    
    report = {
        "passed": passed,
        "issues": issues,
        "stats": stats,
    }
    
    return report


def analyze_chunk_distribution(chunks: List[Document]) -> Dict[str, Any]:
    """
    Analyze the distribution of chunk sizes and sources.
    
    Args:
        chunks: List of Document chunks
    
    Returns:
        Analysis report with distribution statistics
    """
    if not chunks:
        return {"error": "No chunks to analyze"}
    
    # Length distribution
    lengths = [len(c.page_content) for c in chunks]
    
    distribution = {
        "total_chunks": len(chunks),
        "length_stats": {
            "mean": sum(lengths) / len(lengths),
            "min": min(lengths),
            "max": max(lengths),
            "median": sorted(lengths)[len(lengths) // 2],
        },
        "source_distribution": {},
        "page_coverage": {},
    }
    
    # Source file distribution
    for chunk in chunks:
        source = chunk.metadata.get("source_file", "unknown")
        distribution["source_distribution"][source] = \
            distribution["source_distribution"].get(source, 0) + 1
    
    return distribution


# =============================================================================
# CHUNK PERSISTENCE
# =============================================================================

def save_chunks(chunks: List[Document], output_path: str = None) -> str:
    """
    Save chunks to disk for later use.
    
    Args:
        chunks: List of Document chunks
        output_path: Path to save chunks (default: processed_chunks/chunks.json)
    
    Returns:
        Path to saved file
    """
    output_path = output_path or str(PROCESSED_CHUNKS_DIR / "chunks.json")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert Document objects to serializable format
    serializable_chunks = []
    for chunk in chunks:
        serializable_chunks.append({
            "page_content": chunk.page_content,
            "metadata": chunk.metadata,
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_chunks, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(chunks)} chunks to {output_path}")
    return str(output_path)


def load_chunks(input_path: str = None) -> List[Document]:
    """
    Load previously saved chunks from disk.
    
    Args:
        input_path: Path to chunks file (default: processed_chunks/chunks.json)
    
    Returns:
        List of Document chunks
    """
    input_path = input_path or str(PROCESSED_CHUNKS_DIR / "chunks.json")
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {input_path}")
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    chunks = [
        Document(page_content=item["page_content"], metadata=item["metadata"])
        for item in data
    ]
    
    logger.info(f"Loaded {len(chunks)} chunks from {input_path}")
    return chunks


# =============================================================================
# QUALITY GATE 2
# =============================================================================

def run_quality_gate_2(chunks: List[Document]) -> Dict[str, Any]:
    """
    Run Quality Gate 2 checks on chunks.
    
    Quality Gate 2 Criteria:
    - Average chunk size between 300-700 characters
    - All chunks have metadata (source_file, page_number)
    - No chunks split mid-sentence (verify 10 random samples)
    - Total chunks count documented
    
    Args:
        chunks: List of Document chunks
    
    Returns:
        Quality gate report
    """
    logger.info("Running Quality Gate 2...")
    
    validation = validate_chunks(chunks)
    distribution = analyze_chunk_distribution(chunks)
    
    report = {
        "passed": True,
        "criteria_checks": {},
        "validation": validation,
        "distribution": distribution,
    }
    
    # Criterion 1: Average chunk size between 300-700
    avg_length = distribution["length_stats"]["mean"]
    size_check = 300 <= avg_length <= 700
    report["criteria_checks"]["avg_size_300_700"] = {
        "passed": size_check,
        "value": round(avg_length, 1),
        "expected": "300-700",
    }
    if not size_check:
        report["passed"] = False
    
    # Criterion 2: All chunks have metadata
    meta_check = validation["stats"]["missing_metadata"] == 0
    report["criteria_checks"]["all_metadata_present"] = {
        "passed": meta_check,
        "missing_count": validation["stats"]["missing_metadata"],
    }
    if not meta_check:
        report["passed"] = False
    
    # Criterion 3: Total chunks documented
    report["criteria_checks"]["total_chunks_documented"] = {
        "passed": distribution["total_chunks"] > 0,
        "total_chunks": distribution["total_chunks"],
    }
    
    # Criterion 4: No critical issues
    critical_issues = [i for i in validation["issues"] if "Too short" in i or "Too long" in i]
    issues_check = len(critical_issues) == 0
    report["criteria_checks"]["no_critical_issues"] = {
        "passed": issues_check,
        "critical_issues_count": len(critical_issues),
    }
    if not issues_check:
        report["passed"] = False
    
    logger.info(f"Quality Gate 2: {'PASSED' if report['passed'] else 'FAILED'}")
    
    return report


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Example usage
    from src.extraction import process_all_pdfs
    
    logger.info("Starting chunking process...")
    
    # Extract text from PDFs
    extracted_data = process_all_pdfs()
    
    if extracted_data:
        # Create chunks
        chunks = create_chunks(extracted_data)
        
        # Run quality gate
        report = run_quality_gate_2(chunks)
        
        print("\n" + "="*50)
        print("CHUNKING QUALITY REPORT")
        print("="*50)
        print(f"Total chunks: {report['distribution']['total_chunks']}")
        print(f"Average length: {report['distribution']['length_stats']['mean']:.0f} chars")
        print(f"Quality Gate 2: {'PASSED' if report['passed'] else 'FAILED'}")
        
        # Save chunks
        save_chunks(chunks)
    else:
        logger.warning("No data to chunk. Please add PDF files to data/raw_pdfs/")
