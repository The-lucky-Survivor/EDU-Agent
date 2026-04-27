"""
Configuration module for the Educational AI Agent.
Centralizes all settings, paths, and constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file (local development)
load_dotenv(BASE_DIR / ".env", override=True)

# Also try loading from Streamlit Cloud secrets (cloud deployment)
try:
    import streamlit as st
    if hasattr(st, 'secrets'):
        for key in ['GROQ_API_KEY', 'GROQ_REASONING_EFFORT', 'GROQ_MAX_COMPLETION_TOKENS',
                     'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'COHERE_API_KEY']:
            if key in st.secrets:
                os.environ[key] = str(st.secrets[key])
except Exception:
    pass  # Not running in Streamlit context
DATA_DIR = BASE_DIR / "data"
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs"
EXTRACTED_TEXT_DIR = DATA_DIR / "extracted_text"
PROCESSED_CHUNKS_DIR = DATA_DIR / "processed_chunks"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

# Ensure directories exist
for dir_path in [RAW_PDFS_DIR, EXTRACTED_TEXT_DIR, PROCESSED_CHUNKS_DIR, VECTOR_DB_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PDF EXTRACTION SETTINGS
# =============================================================================
MIN_PAGE_TEXT_LENGTH = 50  # Minimum characters to consider a page valid
OCR_ENABLED = False  # Set to True if using OCR for scanned PDFs
OCR_LANGUAGES = "eng+ara"  # Tesseract language configuration

# =============================================================================
# CHUNKING SETTINGS
# =============================================================================
CHUNK_SIZE = 500          # Characters per chunk
CHUNK_OVERLAP = 100       # Overlap between chunks
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]
CHUNK_MAX_SIZE_FACTOR = 1.5  # Maximum allowed chunk size as factor of CHUNK_SIZE

# =============================================================================
# EMBEDDING SETTINGS
# =============================================================================
# Option A: OpenAI (Requires API key)
# EMBEDDING_MODEL = "text-embedding-3-large"
# EMBEDDING_PROVIDER = "openai"  # Options: "openai", "huggingface"

# Option B: HuggingFace (Free, open-source) ← Active
EMBEDDING_MODEL = "BAAI/bge-large-en"
EMBEDDING_PROVIDER = "huggingface"

# =============================================================================
# VECTOR DATABASE SETTINGS
# =============================================================================
VECTOR_DB_PROVIDER = "chromadb"  # Options: "chromadb", "pinecone"
COLLECTION_NAME = "edu_lectures"
PERSIST_DIRECTORY = str(VECTOR_DB_DIR)

# =============================================================================
# RETRIEVAL SETTINGS
# =============================================================================
RETRIEVER_TOP_K = 10       # Number of initial candidates
RETRIEVER_FINAL_K = 5      # Number of results after re-ranking
USE_RERANKING = False      # Set to True if Cohere API key is available
SIMILARITY_SEARCH_K = 5    # Default k for basic similarity search

# =============================================================================
# LLM SETTINGS
# =============================================================================
# Option A: OpenAI GPT-4o
# LLM_PROVIDER = "openai"
# LLM_MODEL = "gpt-4o"
# LLM_TEMPERATURE = 0.1

# Option B: Groq with Reasoning Model ← Active
LLM_PROVIDER = "groq"
LLM_MODEL = "openai/gpt-oss-120b"
LLM_TEMPERATURE = 1.0      # Recommended for reasoning models

# Groq Reasoning-specific settings
GROQ_REASONING_EFFORT = "medium"      # Options: "low", "medium", "high"
GROQ_MAX_COMPLETION_TOKENS = 8192     # Max tokens for response
GROQ_STREAM = True                    # Enable streaming

# Option C: Anthropic Claude
# LLM_PROVIDER = "anthropic"
# LLM_MODEL = "claude-3-5-sonnet-20241022"
# LLM_TEMPERATURE = 0.1

# =============================================================================
# API KEYS
# =============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")

# =============================================================================
# VALIDATION
# =============================================================================
def validate_config():
    """Validate that required API keys are set based on provider selection."""
    errors = []
    
    if EMBEDDING_PROVIDER == "openai" and not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is required for OpenAI embeddings")
    
    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is required for OpenAI LLM")
    elif LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        errors.append("GROQ_API_KEY is required for Groq LLM")
    elif LLM_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
        errors.append("ANTHROPIC_API_KEY is required for Anthropic LLM")
    
    if USE_RERANKING and not COHERE_API_KEY:
        errors.append("COHERE_API_KEY is required for re-ranking")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True

# =============================================================================
# RESPONSE GENERATION SETTINGS
# =============================================================================
MAX_RESPONSE_TIME = 5  # Maximum acceptable response time in seconds
MIN_CONFIDENCE_THRESHOLD = "medium"  # Minimum confidence level for answers

# =============================================================================
# TESTING SETTINGS
# =============================================================================
TEST_QUERIES = [
    "ما هو تعريف التعلم الآلي؟",
    "اشرح مفهوم الشبكات العصبية",
    "ما الفرق بين التعلم الموجه وغير الموجه؟",
    "كيف يعمل خوارزمية K-Means؟",
    "ما هي مراحل بناء نموذج تعلم آلي؟",
]

PERFORMANCE_TARGETS = {
    "avg_response_time": 3.0,      # seconds
    "retrieval_accuracy": 0.90,     # 90%
    "hallucination_rate": 0.05,     # < 5%
    "citation_rate": 1.0,           # 100%
}
