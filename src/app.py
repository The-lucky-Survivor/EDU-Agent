"""
Phase 6: Streamlit UI
Educational AI Agent interface with Arabic support.
Features: Auto-load DB, Multi-subject, Question generation.
"""

import os
import sys
import json
import random
from pathlib import Path

# Add src and project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from langchain_community.vectorstores import Chroma

from config import (
    RAW_PDFS_DIR,
    VECTOR_DB_DIR,
    COLLECTION_NAME,
    validate_config,
)
from extraction import process_all_pdfs, validate_extraction
from chunking import create_chunks, run_quality_gate_2, save_chunks
from embedding import create_vector_store, load_vector_store, get_embeddings_model
from llm_chain import create_rag_chain, get_llm, post_process_answer


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="🎓 المساعد التعليمي الذكي",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    
    .main {
        direction: rtl;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    h1, h2, h3 {
        font-family: 'Cairo', sans-serif !important;
        color: #e94560 !important;
    }
    
    .stChatMessage {
        border-radius: 15px;
        margin: 10px 0;
    }
    
    .stChatMessage [data-testid="stChatMessageAvatarAssistant"] {
        background: #e94560 !important;
    }
    
    .stChatMessage [data-testid="stChatMessageAvatarUser"] {
        background: #0f3460 !important;
    }
    
    .sidebar .sidebar-content {
        background: #16213e;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #e94560, #ff6b6b);
        color: white;
        border-radius: 25px;
        border: none;
        padding: 10px 25px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(233, 69, 96, 0.4);
    }
    
    .success-box {
        background: rgba(0, 255, 100, 0.1);
        border: 1px solid #00ff64;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .info-box {
        background: rgba(15, 52, 96, 0.3);
        border: 1px solid #0f3460;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .source-tag {
        background: #e94560;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
    }
    
    .subject-card {
        background: rgba(233, 69, 96, 0.1);
        border: 1px solid #e94560;
        border-radius: 12px;
        padding: 12px;
        margin: 5px 0;
        text-align: center;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    ::-webkit-scrollbar-thumb {
        background: #e94560;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SUBJECTS MANAGEMENT
# =============================================================================

SUBJECTS_DIR = Path(__file__).resolve().parent.parent / "subjects"
SUBJECTS_FILE = SUBJECTS_DIR / "subjects.json"


def get_subjects() -> dict:
    """Load subjects list from JSON file."""
    if SUBJECTS_FILE.exists():
        with open(SUBJECTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_subjects(subjects: dict):
    """Save subjects list to JSON file."""
    SUBJECTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUBJECTS_FILE, "w", encoding="utf-8") as f:
        json.dump(subjects, f, ensure_ascii=False, indent=2)


def get_subject_db_dir(subject_key: str) -> Path:
    """Get the vector DB directory for a specific subject."""
    return SUBJECTS_DIR / subject_key / "vector_db"


def get_subject_pdfs_dir(subject_key: str) -> Path:
    """Get the PDFs directory for a specific subject."""
    d = SUBJECTS_DIR / subject_key / "pdfs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def migrate_default_subject():
    """Migrate existing data/vector_db to the subjects system on first run."""
    subjects = get_subjects()
    
    # If subjects already exist, skip migration
    if subjects:
        return
    
    # Check if there's an existing vector_db from the old system
    old_db = VECTOR_DB_DIR
    old_db_file = old_db / "chroma.sqlite3"
    
    if old_db_file.exists():
        # There's existing data, create a default subject pointing to the old DB
        subjects["multi_agent"] = {
            "name": "Multi-Agent Systems",
            "icon": "🤖",
            "db_dir": str(old_db),
            "pdfs_dir": str(RAW_PDFS_DIR),
            "use_legacy_path": True,
        }
        save_subjects(subjects)


# =============================================================================
# SESSION STATE
# =============================================================================

def init_session_state():
    defaults = {
        'messages': [],
        'vectorstore': None,
        'chain': None,
        'processing_done': False,
        'stats': {},
        'current_subject': None,
        'llm': None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# =============================================================================
# AUTO-LOAD EXISTING DATABASE
# =============================================================================

def try_auto_load(subject_key: str = None):
    """
    Try to load an existing vector database automatically.
    This runs on startup so users don't need to re-upload files.
    """
    if st.session_state.processing_done and st.session_state.current_subject == subject_key:
        return True  # Already loaded
    
    subjects = get_subjects()
    
    if subject_key and subject_key in subjects:
        subj = subjects[subject_key]
        if subj.get("use_legacy_path"):
            db_dir = subj["db_dir"]
        else:
            db_dir = str(get_subject_db_dir(subject_key))
    else:
        # Fallback: try default vector_db
        db_dir = str(VECTOR_DB_DIR)
    
    db_path = Path(db_dir)
    db_file = db_path / "chroma.sqlite3"
    
    if not db_file.exists():
        return False
    
    try:
        embeddings = get_embeddings_model()
        collection_name = COLLECTION_NAME
        
        vectorstore = Chroma(
            persist_directory=db_dir,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
        
        count = vectorstore._collection.count()
        if count == 0:
            return False
        
        llm = get_llm()
        chain = create_rag_chain(vectorstore, llm)
        
        st.session_state.vectorstore = vectorstore
        st.session_state.chain = chain
        st.session_state.llm = llm
        st.session_state.processing_done = True
        st.session_state.current_subject = subject_key
        st.session_state.stats = {
            'vectors': count,
        }
        return True
        
    except Exception as e:
        st.session_state.processing_done = False
        return False


# =============================================================================
# PROCESSING PIPELINE
# =============================================================================

def process_lectures(subject_key: str = None, pdfs_dir=None, db_dir=None):
    """Run the full pipeline: extract -> chunk -> embed."""
    progress_bar = st.progress(0)
    status = st.empty()
    
    pdfs_dir = pdfs_dir or str(RAW_PDFS_DIR)
    db_dir = db_dir or str(VECTOR_DB_DIR)
    
    try:
        # Phase 1: Extract
        status.info("📄 المرحلة 1: استخراج النصوص من ملفات PDF...")
        all_data = process_all_pdfs(input_dir=pdfs_dir, clean=True, is_arabic=False)
        
        if not all_data:
            st.error("❌ لم يتم استخراج أي نص. تأكد من ملفات PDF.")
            return False
        
        validation = validate_extraction(all_data)
        progress_bar.progress(25)
        
        # Phase 2: Chunk
        status.info("✂️ المرحلة 2: تقسيم النصوص إلى أجزاء...")
        from chunking import get_text_splitter
        splitter = get_text_splitter(chunk_size=700, chunk_overlap=150)
        chunks = create_chunks(all_data, text_splitter=splitter)
        
        chunk_report = run_quality_gate_2(chunks)
        save_chunks(chunks)
        progress_bar.progress(50)
        
        # Phase 3: Embed
        status.info("🧠 المرحلة 3: إنشاء الـ Embeddings وقاعدة البيانات...")
        try:
            embeddings = get_embeddings_model()
        except Exception as e:
            st.warning(f"استخدام embeddings بديلة: {e}")
            from embedding import FastFakeEmbeddings
            embeddings = FastFakeEmbeddings(dims=384)
        
        vectorstore = create_vector_store(
            chunks, embeddings,
            persist_dir=db_dir,
            collection_name=COLLECTION_NAME,
        )
        progress_bar.progress(75)
        
        # Phase 4-5: Create RAG chain
        status.info("🔗 المرحلة 4-5: إعداد الذكاء الاصطناعي...")
        llm = get_llm()
        chain = create_rag_chain(vectorstore, llm)
        progress_bar.progress(100)
        
        # Save to session
        st.session_state.vectorstore = vectorstore
        st.session_state.chain = chain
        st.session_state.llm = llm
        st.session_state.processing_done = True
        st.session_state.current_subject = subject_key
        st.session_state.stats = {
            'pages': len(all_data),
            'chunks': len(chunks),
            'vectors': vectorstore._collection.count(),
        }
        
        status.success("✅ تمت المعالجة بنجاح!")
        return True
        
    except Exception as e:
        status.error(f"❌ خطأ: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False


# =============================================================================
# QUESTION GENERATION
# =============================================================================

def get_available_lectures():
    """Get list of available lectures from the vector store metadata."""
    if not st.session_state.vectorstore:
        return []
    
    try:
        results = st.session_state.vectorstore._collection.get(include=['metadatas'])
        sources = set()
        for m in results['metadatas']:
            src = m.get('source_file', m.get('source', ''))
            if src:
                sources.add(src)
        return sorted(sources)
    except Exception:
        return []


def generate_mcq_bank(lecture_name: str):
    """
    Generate a bank of academic MCQ questions for a specific lecture.
    Number of questions scales with lecture content size.
    Results are cached in session_state per lecture.
    """
    if not st.session_state.llm or not st.session_state.vectorstore:
        return None
    
    llm = st.session_state.llm
    vectorstore = st.session_state.vectorstore
    
    # Retrieve docs ONLY from the selected lecture using metadata filter
    try:
        docs = vectorstore.similarity_search(
            query="key concepts definitions algorithms formulas examples comparisons methods types properties",
            k=20,
            filter={"source_file": lecture_name},
        )
    except Exception:
        # Fallback if filter doesn't work
        retriever = vectorstore.as_retriever(search_kwargs={'k': 20})
        all_docs = retriever.invoke(f"all topics from {lecture_name}")
        docs = [d for d in all_docs if lecture_name in str(d.metadata.get('source_file', ''))]
        if not docs:
            docs = all_docs[:10]
    
    if not docs:
        return None
    
    # Calculate number of questions based on content size
    total_chars = sum(len(doc.page_content) for doc in docs)
    if total_chars < 2000:
        num_questions = 5
    elif total_chars < 5000:
        num_questions = 10
    elif total_chars < 10000:
        num_questions = 15
    else:
        num_questions = 20
    
    # Build context from the filtered docs - strip ALL metadata
    context_parts = []
    for doc in docs:
        # Remove any page/source references from the content itself
        text = doc.page_content
        context_parts.append(text)
    context = "\n\n".join(context_parts)
    
    from langchain_core.messages import HumanMessage
    
    prompt_text = f"""You are writing a real university final exam. Generate {num_questions} Multiple Choice Questions (MCQ) that test deep understanding of the TECHNICAL CONTENT below.

ABSOLUTE RULES:
1. Every question must ask about a SCIENTIFIC CONCEPT, ALGORITHM, FORMULA, DEFINITION, or TECHNICAL COMPARISON.
2. NEVER mention chapter numbers, lecture numbers, slide numbers, page numbers, section titles, or any organizational references in your questions or choices.
3. NEVER ask "which chapter", "which lecture", "which section", "where is this discussed", or anything about the structure/organization of the material.
4. NEVER ask about professor names, course codes, university names, or dates.
5. Each question has exactly 4 choices (A, B, C, D), ONE correct answer.
6. Wrong choices must be plausible — they should sound correct to someone who didn't study well.
7. All text in ENGLISH.

QUESTION TYPES YOU SHOULD USE:
- "What is the definition of [technical term]?"
- "What happens when [algorithm step / condition]?"
- "Which of the following is TRUE about [concept]?"
- "What is the difference between [X] and [Y]?"
- "If an agent receives reward R in state S, what will [algorithm] do?"
- "Which formula correctly represents [concept]?"
- "What is the main disadvantage of [method]?"

FORBIDDEN QUESTION TYPES (instant fail):
- "Which chapter discusses X?" ← FORBIDDEN
- "The concept of X is covered in which lecture?" ← FORBIDDEN  
- "What is the primary focus of Chapter 5?" ← FORBIDDEN
- "Where in the course is X introduced?" ← FORBIDDEN
- Any question where a choice is a chapter/lecture name ← FORBIDDEN

FORMAT:
Q1. [technical question]
A) [choice]
B) [choice]
C) [choice]
D) [choice]

(continue for all {num_questions} questions)

ANSWER KEY:
Q1: [correct letter]
Q2: [correct letter]
(etc.)

Lecture Content:
{context}

Generate {num_questions} high-quality academic MCQ questions:"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt_text)])
        content = response.content if hasattr(response, 'content') else str(response)
        return content
    except Exception as e:
        return f"❌ Error generating questions: {str(e)}"


def get_shuffled_questions(lecture_name: str):
    """
    Get 5 shuffled MCQ questions for a lecture.
    Generates bank of 20 on first call, then shuffles from cache.
    """
    cache_key = f'mcq_bank_{lecture_name}'
    
    # Check if we have a cached bank for this lecture
    if cache_key not in st.session_state or not st.session_state[cache_key]:
        # Generate new bank
        bank = generate_mcq_bank(lecture_name)
        if not bank or bank.startswith("❌"):
            return bank
        st.session_state[cache_key] = bank
    
    full_bank = st.session_state[cache_key]
    
    # Parse questions from the bank
    import re
    # Split by question pattern Q1., Q2., etc.
    question_blocks = re.split(r'(?=Q\d+\.)', full_bank)
    question_blocks = [q.strip() for q in question_blocks if q.strip() and re.match(r'Q\d+\.', q.strip())]
    
    # Separate answer key
    answer_key = ""
    clean_questions = []
    for block in question_blocks:
        if 'ANSWER KEY' in block.upper():
            # This block contains the answer key mixed in
            parts = re.split(r'ANSWER\s*KEY', block, flags=re.IGNORECASE)
            if parts[0].strip():
                clean_questions.append(parts[0].strip())
            if len(parts) > 1:
                answer_key = parts[1].strip()
        else:
            clean_questions.append(block)
    
    # Also try to extract answer key from the end of full_bank
    if not answer_key:
        ak_match = re.search(r'ANSWER\s*KEY[:\s]*(.+)', full_bank, re.IGNORECASE | re.DOTALL)
        if ak_match:
            answer_key = ak_match.group(1).strip()
    
    if len(clean_questions) < 5:
        # If parsing failed, just return 5 random sections
        return full_bank
    
    # Shuffle and pick 5
    selected = random.sample(clean_questions, min(5, len(clean_questions)))
    
    # Renumber them 1-5
    result_parts = []
    answer_lines = []
    for i, q in enumerate(selected, 1):
        # Replace original number with new number
        renumbered = re.sub(r'^Q\d+\.', f'Q{i}.', q)
        result_parts.append(renumbered)
        
        # Try to find the original answer for this question
        orig_num = re.match(r'Q(\d+)\.', q)
        if orig_num and answer_key:
            num = orig_num.group(1)
            ans_match = re.search(rf'Q{num}[:\s]+([A-D])', answer_key)
            if ans_match:
                answer_lines.append(f"Q{i}: {ans_match.group(1)}")
    
    result = "\n\n".join(result_parts)
    if answer_lines:
        result += "\n\n**ANSWER KEY:**\n" + "\n".join(answer_lines)
    
    return result


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="color: #e94560;">🎓 المساعد التعليمي</h1>
            <p style="color: #aaa;">AI Educational Agent</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ===================== SUBJECT MANAGEMENT =====================
        subjects = get_subjects()
        
        st.header("📚 المواد الدراسية")
        
        # Subject selector
        subject_options = {}
        if subjects:
            for key, subj in subjects.items():
                label = f"{subj.get('icon', '📖')} {subj['name']}"
                subject_options[label] = key
        
        subject_options["➕ إضافة مادة جديدة"] = "__new__"
        
        selected_label = st.selectbox(
            "اختر المادة",
            options=list(subject_options.keys()),
            index=0 if subjects else len(subject_options) - 1,
        )
        selected_key = subject_options[selected_label]
        
        # Handle new subject creation
        if selected_key == "__new__":
            with st.expander("➕ إضافة مادة جديدة", expanded=True):
                new_name = st.text_input("اسم المادة", placeholder="مثال: Machine Learning")
                new_icon = st.selectbox("الأيقونة", ["🤖", "🧠", "📊", "💻", "📐", "🔬", "📡", "⚙️"], index=0)
                
                if st.button("✅ إنشاء المادة") and new_name:
                    # Create a safe key
                    safe_key = new_name.lower().replace(" ", "_").replace("-", "_")
                    safe_key = "".join(c for c in safe_key if c.isalnum() or c == "_")
                    
                    subjects[safe_key] = {
                        "name": new_name,
                        "icon": new_icon,
                        "db_dir": str(get_subject_db_dir(safe_key)),
                        "pdfs_dir": str(get_subject_pdfs_dir(safe_key)),
                        "use_legacy_path": False,
                    }
                    save_subjects(subjects)
                    st.success(f"✅ تم إنشاء مادة '{new_name}'!")
                    st.rerun()
            
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; color: #666; font-size: 0.8em;">
                Made with ❤️ for AI Students
            </div>
            """, unsafe_allow_html=True)
            return None
        
        # ===================== ACTIVE SUBJECT =====================
        current_subject = subjects.get(selected_key, {})
        
        # Auto-load the selected subject's database
        if st.session_state.current_subject != selected_key:
            st.session_state.processing_done = False
            st.session_state.chain = None
            st.session_state.vectorstore = None
        
        # Try auto-loading
        auto_loaded = try_auto_load(selected_key)
        
        if auto_loaded:
            st.success(f"✅ تم تحميل قاعدة البيانات تلقائياً ({st.session_state.stats.get('vectors', 0)} vectors)")
        
        st.markdown("---")
        
        # ===================== FILE UPLOAD =====================
        st.header("📁 إضافة محاضرات جديدة")
        
        uploaded_files = st.file_uploader(
            "اختر ملفات PDF",
            type=['pdf'],
            accept_multiple_files=True,
        )
        
        if uploaded_files:
            # Determine where to save
            if current_subject.get("use_legacy_path"):
                save_dir = Path(current_subject["pdfs_dir"])
            else:
                save_dir = get_subject_pdfs_dir(selected_key)
            
            save_dir.mkdir(parents=True, exist_ok=True)
            
            for file in uploaded_files:
                save_path = save_dir / file.name
                save_path.write_bytes(file.getvalue())
            
            st.info(f"📎 تم تحميل {len(uploaded_files)} ملفات")
        
        # Process button
        if st.button("🚀 معالجة المحاضرات", use_container_width=True):
            if current_subject.get("use_legacy_path"):
                pdfs = current_subject["pdfs_dir"]
                db = current_subject["db_dir"]
            else:
                pdfs = str(get_subject_pdfs_dir(selected_key))
                db = str(get_subject_db_dir(selected_key))
            
            # Make sure db dir exists
            Path(db).mkdir(parents=True, exist_ok=True)
            
            with st.spinner("جاري المعالجة..."):
                process_lectures(
                    subject_key=selected_key,
                    pdfs_dir=pdfs,
                    db_dir=db,
                )
        
        # Show stats if processing done
        if st.session_state.processing_done and st.session_state.stats:
            st.markdown("---")
            st.subheader("📊 الإحصائيات")
            stats = st.session_state.stats
            
            if stats.get('pages'):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📄 Pages", stats.get('pages', 0))
                with col2:
                    st.metric("✂️ Chunks", stats.get('chunks', 0))
            
            st.metric("🧠 Vectors", stats.get('vectors', 0))
        
        st.markdown("---")
        
        # ===================== QUESTION GENERATION =====================
        st.header("📝 اختبر نفسك!")
        st.caption("ولّد أسئلة اختيار من متعدد (MCQ) من محاضراتك")
        
        # Get available lectures as dropdown
        available_lectures = get_available_lectures()
        
        if available_lectures:
            # Make readable names
            lecture_display = {}
            for lec in available_lectures:
                display = lec.replace('+', ' ').replace('_', ' ')
                lecture_display[display] = lec
            
            # Step 1: Select lecture
            st.markdown("**1️⃣ اختر المحاضرة:**")
            selected_lecture_display = st.selectbox(
                "المحاضرة",
                options=list(lecture_display.keys()),
                index=0,
                label_visibility="collapsed",
            )
            selected_lecture = lecture_display[selected_lecture_display]
            
            # Check if bank exists for this lecture
            cache_key = f'mcq_bank_{selected_lecture}'
            has_bank = cache_key in st.session_state and st.session_state[cache_key]
            
            # Step 2: Generate or Shuffle
            st.markdown("**2️⃣ ولّد الأسئلة:**")
            
            if not has_bank:
                # First time - need to generate
                if st.button("🎯 ابدأ توليد الأسئلة", use_container_width=True):
                    if st.session_state.processing_done:
                        with st.spinner("🤔 جاري توليد أسئلة امتحان أكاديمية... (أول مرة فقط، قد يستغرق 30-60 ثانية)"):
                            st.session_state['generated_questions'] = get_shuffled_questions(selected_lecture)
                        st.rerun()
                    else:
                        st.warning("⚠️ يجب معالجة المحاضرات أولاً!")
                st.info("💡 سيتم توليد بنك أسئلة أكاديمية (MCQ) حسب حجم المحاضرة. بعدها كل ضغطة تعطيك 5 أسئلة مختلفة فوراً.")
            else:
                # Bank exists - can shuffle instantly
                st.success("✅ بنك الأسئلة جاهز!")
                if st.button("🔀 أعطني 5 أسئلة مختلفة", use_container_width=True):
                    st.session_state['generated_questions'] = get_shuffled_questions(selected_lecture)
                    st.rerun()
                
                st.caption("كل ضغطة تعطيك 5 أسئلة عشوائية من بنك الأسئلة")
                
                # Option to regenerate from scratch
                with st.expander("⚙️ خيارات متقدمة"):
                    if st.button("♻️ أعد توليد بنك الأسئلة من الصفر"):
                        del st.session_state[cache_key]
                        with st.spinner("🤔 جاري إعادة توليد 20 سؤال MCQ جديد..."):
                            st.session_state['generated_questions'] = get_shuffled_questions(selected_lecture)
                        st.rerun()
        else:
            st.info("⚠️ قم بمعالجة المحاضرات أولاً لتفعيل توليد الأسئلة")
        
        st.markdown("---")
        
        # Clear chat
        if st.button("🗑️ مسح المحادثة", use_container_width=True):
            st.session_state.messages = []
            if 'generated_questions' in st.session_state:
                del st.session_state['generated_questions']
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.8em;">
            Made with ❤️ for AI Students
        </div>
        """, unsafe_allow_html=True)
        
        return selected_key


# =============================================================================
# MAIN CHAT
# =============================================================================

def render_chat(subject_key: str = None):
    # Header
    subjects = get_subjects()
    subject_name = "المحاضرات"
    if subject_key and subject_key in subjects:
        subj = subjects[subject_key]
        subject_name = f"{subj.get('icon', '📖')} {subj['name']}"
    
    st.markdown(f"""
    <div style="text-align: center; padding: 10px 0 20px;">
        <h1>🎓 المساعد التعليمي الذكي</h1>
        <p style="color: #aaa;">اسأل عن {subject_name}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Show generated questions if available
    if 'generated_questions' in st.session_state and st.session_state['generated_questions']:
        with st.expander("📝 الأسئلة المولّدة", expanded=True):
            st.markdown(st.session_state['generated_questions'])
    
    # Show welcome if no processing done
    if not st.session_state.processing_done:
        if subject_key == "__new__" or subject_key is None:
            st.info("👈 اختر مادة من الشريط الجانبي أو أنشئ مادة جديدة")
        else:
            st.info("👈 ارفع ملفات PDF واضغط 'معالجة المحاضرات' من الشريط الجانبي")
        
        st.markdown("""
        <div class="info-box">
        <h4>📚 كيفية الاستخدام:</h4>
        <ol style="text-align: right; direction: rtl;">
            <li>اختر المادة الدراسية من الشريط الجانبي (أو أنشئ مادة جديدة)</li>
            <li>إذا كانت هذه المرة الأولى، ارفع ملفات PDF واضغط <b>معالجة المحاضرات</b></li>
            <li>من المرة التالية، سيتم تحميل قاعدة البيانات تلقائياً ✨</li>
            <li>ابدأ في طرح الأسئلة أو اطلب توليد أسئلة تدريبية! 🤖</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            if message['role'] == 'assistant' and 'processed' in message:
                st.markdown(message['content'])
                # Show sources in expander
                if message.get('sources'):
                    with st.expander("📚 المصادر"):
                        for src in message['sources']:
                            st.markdown(f"- {src}")
            else:
                st.markdown(message['content'])
    
    # Chat input
    if prompt := st.chat_input("اسأل سؤالاً عن المحاضرات..."):
        # Add user message
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        with st.chat_message('user'):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message('assistant'):
            with st.spinner("🤔 جاري التفكير والبحث في المحاضرات... (نموذج الاستدلال قد يستغرق 10-30 ثانية)"):
                try:
                    response = st.session_state.chain(prompt)
                    processed = post_process_answer(response)
                    
                    st.markdown(processed['answer'])
                    
                    # Show sources
                    if processed['sources']:
                        with st.expander("📚 المصادر"):
                            for src in processed['sources']:
                                st.markdown(f"- {src}")
                    
                    # Show confidence badge
                    confidence_color = {
                        'high': '🟢',
                        'medium': '🟡',
                        'low': '🔴'
                    }
                    conf = processed.get('confidence', 'medium')
                    st.caption(f"{confidence_color.get(conf, '🟡')} Confidence: {conf}")
                    
                    # Save assistant message
                    st.session_state.messages.append({
                        'role': 'assistant',
                        'content': processed['answer'],
                        'sources': processed['sources'],
                        'processed': True,
                    })
                    
                except Exception as e:
                    error_msg = f"❌ خطأ: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        'role': 'assistant',
                        'content': error_msg,
                    })


# =============================================================================
# MAIN
# =============================================================================

def main():
    init_session_state()
    
    # Migrate existing data to subjects system on first run
    migrate_default_subject()
    
    selected_subject = render_sidebar()
    render_chat(subject_key=selected_subject)


if __name__ == '__main__':
    main()
