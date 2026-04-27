"""
FastAPI Backend Server for Educational AI Agent.
Replaces Streamlit with a proper REST API.
"""

import os
import sys

# Fix for ChromaDB in Docker / Hugging Face Spaces
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import json
import random
import re
import logging
from pathlib import Path
from typing import Optional, List

# Fix paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import RAW_PDFS_DIR, VECTOR_DB_DIR, COLLECTION_NAME, validate_config
from embedding import get_embeddings_model, create_vector_store, load_vector_store
from llm_chain import create_rag_chain, get_llm, post_process_answer
from extraction import process_all_pdfs, validate_extraction
from chunking import create_chunks, get_text_splitter, run_quality_gate_2, save_chunks

from langchain_community.vectorstores import Chroma

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# APP STATE
# =============================================================================

class AppState:
    """Global application state (replaces Streamlit session_state)."""
    def __init__(self):
        self.vectorstore = None
        self.chain = None
        self.llm = None
        self.embeddings = None
        self.current_subject = None
        self.stats = {}
        self.quiz_banks = {}  # {lecture_name: raw_text}

state = AppState()

# =============================================================================
# SUBJECTS MANAGEMENT
# =============================================================================

SUBJECTS_DIR = BASE_DIR / "subjects"
SUBJECTS_FILE = SUBJECTS_DIR / "subjects.json"

def get_subjects() -> dict:
    if SUBJECTS_FILE.exists():
        with open(SUBJECTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_subjects(subjects: dict):
    SUBJECTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUBJECTS_FILE, "w", encoding="utf-8") as f:
        json.dump(subjects, f, ensure_ascii=False, indent=2)

def get_subject_db_dir(key: str) -> Path:
    return SUBJECTS_DIR / key / "vector_db"

def get_subject_pdfs_dir(key: str) -> Path:
    d = SUBJECTS_DIR / key / "pdfs"
    d.mkdir(parents=True, exist_ok=True)
    return d

def migrate_default():
    subjects = get_subjects()
    if subjects:
        return
    old_db = VECTOR_DB_DIR / "chroma.sqlite3"
    if old_db.exists():
        subjects["multi_agent"] = {
            "name": "Multi-Agent Systems",
            "icon": "🤖",
            "db_dir": str(VECTOR_DB_DIR),
            "pdfs_dir": str(RAW_PDFS_DIR),
            "use_legacy_path": True,
        }
    else:
        # Create a blank default subject so the UI is never empty
        subjects["general_subject"] = {
            "name": "General Subject",
            "icon": "📚",
            "db_dir": str(get_subject_db_dir("general_subject")),
            "pdfs_dir": str(get_subject_pdfs_dir("general_subject")),
            "use_legacy_path": False,
        }
    save_subjects(subjects)

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(title="EDU Agent API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve static frontend
WEB_DIR = BASE_DIR / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ChatRequest(BaseModel):
    question: str

class SubjectCreate(BaseModel):
    name: str
    icon: str = "📖"

class QuizRequest(BaseModel):
    lecture: str
    regenerate: bool = False

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def serve_index():
    return FileResponse(str(WEB_DIR / "index.html"))

@app.get("/api/subjects")
async def list_subjects():
    migrate_default()
    return get_subjects()

@app.post("/api/subjects")
async def create_subject(req: SubjectCreate):
    subjects = get_subjects()
    safe_key = re.sub(r'[^a-z0-9_]', '', req.name.lower().replace(" ", "_").replace("-", "_"))
    if not safe_key:
        raise HTTPException(400, "Invalid subject name")
    subjects[safe_key] = {
        "name": req.name,
        "icon": req.icon,
        "db_dir": str(get_subject_db_dir(safe_key)),
        "pdfs_dir": str(get_subject_pdfs_dir(safe_key)),
        "use_legacy_path": False,
    }
    save_subjects(subjects)
    return {"key": safe_key, "subject": subjects[safe_key]}

@app.post("/api/subjects/{key}/load")
async def load_subject(key: str):
    subjects = get_subjects()
    if key not in subjects:
        raise HTTPException(404, "Subject not found")
    subj = subjects[key]
    db_dir = subj["db_dir"] if subj.get("use_legacy_path") else str(get_subject_db_dir(key))
    db_file = Path(db_dir) / "chroma.sqlite3"
    if not db_file.exists():
        return {"loaded": False, "message": "No database found. Process lectures first."}
    try:
        if state.embeddings is None:
            state.embeddings = get_embeddings_model()
        vs = Chroma(persist_directory=db_dir, embedding_function=state.embeddings, collection_name=COLLECTION_NAME)
        count = vs._collection.count()
        if count == 0:
            return {"loaded": False, "message": "Database is empty."}
        if state.llm is None:
            state.llm = get_llm()
        state.vectorstore = vs
        state.chain = create_rag_chain(vs, state.llm)
        state.current_subject = key
        state.stats = {"vectors": count}
        state.quiz_banks = {}
        return {"loaded": True, "vectors": count}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/subjects/{key}/process")
async def process_subject(key: str):
    subjects = get_subjects()
    if key not in subjects:
        raise HTTPException(404, "Subject not found")
    subj = subjects[key]
    pdfs_dir = subj["pdfs_dir"] if subj.get("use_legacy_path") else str(get_subject_pdfs_dir(key))
    db_dir = subj["db_dir"] if subj.get("use_legacy_path") else str(get_subject_db_dir(key))
    Path(db_dir).mkdir(parents=True, exist_ok=True)
    try:
        all_data = process_all_pdfs(input_dir=pdfs_dir, clean=True, is_arabic=False)
        if not all_data:
            raise HTTPException(400, "No text extracted from PDFs.")
        splitter = get_text_splitter(chunk_size=700, chunk_overlap=150)
        chunks = create_chunks(all_data, text_splitter=splitter)
        save_chunks(chunks)
        if state.embeddings is None:
            state.embeddings = get_embeddings_model()
        vs = create_vector_store(chunks, state.embeddings, persist_dir=db_dir, collection_name=COLLECTION_NAME)
        if state.llm is None:
            state.llm = get_llm()
        state.vectorstore = vs
        state.chain = create_rag_chain(vs, state.llm)
        state.current_subject = key
        vec_count = vs._collection.count()
        state.stats = {"pages": len(all_data), "chunks": len(chunks), "vectors": vec_count}
        state.quiz_banks = {}
        return {"success": True, "pages": len(all_data), "chunks": len(chunks), "vectors": vec_count}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/subjects/{key}/upload")
async def upload_pdfs(key: str, files: List[UploadFile] = File(...)):
    subjects = get_subjects()
    if key not in subjects:
        raise HTTPException(404, "Subject not found")
    subj = subjects[key]
    save_dir = Path(subj["pdfs_dir"]) if subj.get("use_legacy_path") else get_subject_pdfs_dir(key)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        path = save_dir / f.filename
        content = await f.read()
        path.write_bytes(content)
        saved.append(f.filename)
    return {"uploaded": saved}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not state.chain:
        raise HTTPException(400, "No subject loaded. Load a subject first.")
    try:
        raw = state.chain(req.question)
        processed = post_process_answer(raw)
        return processed
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/lectures")
async def get_lectures():
    if not state.vectorstore:
        return []
    try:
        results = state.vectorstore._collection.get(include=["metadatas"])
        sources = set()
        for m in results["metadatas"]:
            src = m.get("source_file", m.get("source", ""))
            if src:
                sources.add(src)
        return sorted(sources)
    except Exception:
        return []

@app.post("/api/quiz/generate")
async def generate_quiz(req: QuizRequest):
    if not state.vectorstore or not state.llm:
        raise HTTPException(400, "No subject loaded.")
    lecture = req.lecture

    # Check cache
    if not req.regenerate and lecture in state.quiz_banks:
        return _shuffle_quiz(state.quiz_banks[lecture])

    # Retrieve content from specific lecture
    try:
        docs = state.vectorstore.similarity_search(
            query="key concepts definitions algorithms formulas examples comparisons methods",
            k=20, filter={"source_file": lecture}
        )
    except Exception:
        docs = state.vectorstore.similarity_search(
            query=f"all topics from {lecture}", k=15
        )
        docs = [d for d in docs if lecture in str(d.metadata.get("source_file", ""))]

    if not docs:
        raise HTTPException(404, "No content found for this lecture.")

    total_chars = sum(len(d.page_content) for d in docs)
    num_q = 5 if total_chars < 2000 else 10 if total_chars < 5000 else 15 if total_chars < 10000 else 20
    context = "\n\n".join(d.page_content for d in docs)

    from langchain_core.messages import HumanMessage
    prompt = f"""You are writing a real university final exam. Generate {num_q} Multiple Choice Questions (MCQ) that test deep understanding of the TECHNICAL CONTENT below.

ABSOLUTE RULES:
1. Every question must ask about a SCIENTIFIC CONCEPT, ALGORITHM, FORMULA, DEFINITION, or TECHNICAL COMPARISON.
2. NEVER mention chapter numbers, lecture numbers, slide numbers, page numbers, section titles, or any organizational references.
3. NEVER ask "which chapter", "which lecture", "which section", or anything about the material's structure.
4. NEVER ask about professor names, course codes, university names, or dates.
5. Each question has exactly 4 choices (A, B, C, D), ONE correct answer.
6. Wrong choices must be plausible.
7. All text in ENGLISH.

QUESTION TYPES YOU SHOULD USE:
- "What is the definition of [technical term]?"
- "What happens when [algorithm step / condition]?"
- "Which of the following is TRUE about [concept]?"
- "What is the difference between [X] and [Y]?"
- "Which formula correctly represents [concept]?"

FORBIDDEN: Any question where a choice is a chapter/lecture name.

FORMAT (use exactly):
Q1. [question]
A) [choice]
B) [choice]
C) [choice]
D) [choice]

ANSWER KEY:
Q1: [letter]

Content:
{context}

Generate {num_q} academic MCQ questions:"""

    try:
        resp = state.llm.invoke([HumanMessage(content=prompt)])
        bank_text = resp.content if hasattr(resp, "content") else str(resp)
        state.quiz_banks[lecture] = bank_text
        return _shuffle_quiz(bank_text)
    except Exception as e:
        raise HTTPException(500, str(e))

def _shuffle_quiz(bank_text: str):
    """Parse MCQ bank, shuffle, return 5 as structured JSON."""
    blocks = re.split(r'(?=Q\d+\.)', bank_text)
    blocks = [b.strip() for b in blocks if b.strip() and re.match(r'Q\d+\.', b.strip())]

    answer_key_text = ""
    clean = []
    for b in blocks:
        if "ANSWER KEY" in b.upper():
            parts = re.split(r'ANSWER\s*KEY', b, flags=re.IGNORECASE)
            if parts[0].strip():
                clean.append(parts[0].strip())
            if len(parts) > 1:
                answer_key_text = parts[1].strip()
        else:
            clean.append(b)

    if not answer_key_text:
        ak = re.search(r'ANSWER\s*KEY[:\s]*(.+)', bank_text, re.IGNORECASE | re.DOTALL)
        if ak:
            answer_key_text = ak.group(1).strip()

    if len(clean) < 5:
        return {"questions": [{"raw": bank_text}], "total_bank": len(clean)}

    selected = random.sample(clean, min(5, len(clean)))
    questions = []
    for i, q in enumerate(selected, 1):
        q_text = re.sub(r'^Q\d+\.', '', q).strip()
        lines = q_text.split('\n')
        question_text = ""
        choices = {}
        for line in lines:
            line = line.strip()
            m = re.match(r'^([A-D])\)\s*(.+)', line)
            if m:
                choices[m.group(1)] = m.group(2).strip()
            elif not question_text:
                question_text = line
            elif not re.match(r'^[A-D]\)', line) and not choices:
                question_text += " " + line

        orig_num = re.match(r'Q(\d+)\.', q)
        correct = ""
        if orig_num and answer_key_text:
            am = re.search(rf'Q{orig_num.group(1)}[:\s]+([A-D])', answer_key_text)
            if am:
                correct = am.group(1)

        questions.append({
            "id": i,
            "question": question_text.strip(),
            "choices": choices,
            "correct": correct,
        })

    return {"questions": questions, "total_bank": len(clean)}

# =============================================================================
# STARTUP
# =============================================================================

@app.on_event("startup")
async def startup():
    migrate_default()
    logger.info("🎓 EDU Agent API started")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
