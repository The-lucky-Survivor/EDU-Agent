# 🎓 EDU Agent — Full Project Documentation

> **AI-Powered Educational Assistant Using Retrieval-Augmented Generation (RAG)**

| | |
|---|---|
| **Live Demo** | [https://tinyurl.com/edu-agent-ap](https://tinyurl.com/edu-agent-ap) |
| **Source Code** | [https://github.com/The-lucky-Survivor/EDU-Agent](https://github.com/The-lucky-Survivor/EDU-Agent) |
| **Deployment** | Hugging Face Spaces (Docker) |

---

## 1. Project Overview

**EDU Agent** is an intelligent web application that helps university students study more effectively. It reads lecture PDFs, builds a searchable knowledge base, and provides two core capabilities:

1. **Q&A Chat** — Ask any question and get accurate answers derived exclusively from your uploaded lecture materials.
2. **MCQ Quiz Generator** — Automatically generate practice exam questions (Multiple Choice) from any lecture for self-assessment.

The system is built on the **RAG (Retrieval-Augmented Generation)** paradigm, which combines document retrieval with large language models to produce grounded, citation-backed answers — eliminating hallucination.

### Key Features

| Feature | Description |
|---|---|
| 🔄 **Persistent Knowledge Base** | Vector database is saved between sessions. No need to re-upload lectures every time. |
| 📚 **Multi-Subject Support** | Create separate subjects, each with its own files and isolated database. |
| 📝 **Smart MCQ Generation** | Generates 5–20 academic-level MCQ questions per lecture, focusing on concepts, algorithms, and definitions. |
| 🧠 **Advanced Reasoning** | Powered by Groq's reasoning models (`openai/gpt-oss-120b`) for analytical, in-depth answers. |
| 📄 **Robust PDF Processing** | Handles multi-page continuity, image-embedded text (OCR fallback), and broken-word repair. |
| 🌐 **Cloud Deployed** | Fully hosted — accessible from any device via a web browser without local installation. |

---

## 2. System Architecture

### 2.1 High-Level Architecture

The system follows a clean Client-Server architecture:

```
┌────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Browser)                          │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐                    │
│   │  Chat    │    │  Quiz    │    │  Upload  │                    │
│   │  Tab     │    │  Tab     │    │  Tab     │                    │
│   └────┬─────┘    └────┬─────┘    └────┬─────┘                    │
│        └───────────────┼───────────────┘                          │
│                        │ HTTP (REST API)                           │
└────────────────────────┼──────────────────────────────────────────┘
                         │
┌────────────────────────┼──────────────────────────────────────────┐
│                   BACKEND (FastAPI)                                │
│                        │                                          │
│   ┌────────────────────┼────────────────────────────┐            │
│   │              API Router                          │            │
│   │  /api/subjects  /api/chat  /api/quiz  /api/upload│            │
│   └────────────────────┬────────────────────────────┘            │
│                        │                                          │
│   ┌────────────────────┼────────────────────────────┐            │
│   │            RAG Engine (AI Core)                   │            │
│   │                                                   │            │
│   │  extraction.py → chunking.py → embedding.py      │            │
│   │                                                   │            │
│   │  retrieval.py → llm_chain.py                     │            │
│   └────────┬─────────────────────────┬──────────────┘            │
│            │                         │                            │
│   ┌────────┴────────┐    ┌──────────┴──────────┐                │
│   │    ChromaDB     │    │     Groq API        │                │
│   │  (Vector Store) │    │  (LLM Inference)    │                │
│   └─────────────────┘    └─────────────────────┘                │
└───────────────────────────────────────────────────────────────────┘
```

### 2.2 RAG Pipeline (Step-by-Step)

The core intelligence of EDU Agent follows a 6-phase pipeline:

```
Phase 1          Phase 2          Phase 3          Phase 4          Phase 5
┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐
│  PDF Text │──▶│  Smart    │──▶│ Embedding │──▶│ Storage   │──▶│ Retrieval │
│ Extraction│   │ Chunking  │   │ Generation│   │ ChromaDB  │   │ + LLM     │
└───────────┘   └───────────┘   └───────────┘   └───────────┘   └───────────┘
  PyMuPDF        LangChain      HuggingFace      ChromaDB        Groq API
  OCR Fallback   700 chars      BGE-Large-EN     Vector DB       GPT-OSS-120B
  Text Cleaning  150 overlap    768-dim vectors  Persist Dir     Reasoning Mode
```

**Phase 1 — PDF Text Extraction** (`extraction.py`)
- Uses **PyMuPDF (fitz)** for native text extraction from PDF files.
- Automatically detects scanned PDFs and falls back to OCR if enabled.
- Cleans extracted text: removes headers/footers, fixes broken hyphenated words, normalizes whitespace.
- Merges text across pages to preserve paragraph continuity.
- Supports both English and Arabic text.

**Phase 2 — Smart Chunking** (`chunking.py`)
- Splits extracted text into semantically meaningful chunks using **RecursiveCharacterTextSplitter** from LangChain.
- Default configuration: **700 characters per chunk** with **150-character overlap** to maintain context across chunk boundaries.
- Quality gate validates chunk sizes and discards statistical outliers (too short or too long).
- Preserves metadata: source file name and page numbers for later citation.

**Phase 3 — Embedding Generation** (`embedding.py`)
- Converts text chunks into 768-dimensional dense vector representations using **BAAI/bge-large-en** model from HuggingFace.
- This is a free, open-source model — no API key required for the embedding step.
- Vectors capture semantic meaning, enabling "meaning-based" search rather than keyword matching.

**Phase 4 — Vector Storage** (`embedding.py`)
- Stores the generated vectors in **ChromaDB**, an open-source vector database.
- Each subject gets its own isolated vector database directory under `subjects/<key>/vector_db/`.
- Database persists to disk so it survives server restarts.
- Metadata (source file, page number) is stored alongside each vector for citation purposes.

**Phase 5 — Retrieval & LLM Chain** (`retrieval.py` + `llm_chain.py`)
- When a user asks a question:
  1. The question is embedded into a vector using the same embedding model.
  2. ChromaDB performs a **similarity search** to find the top-K most relevant chunks.
  3. A prompt is constructed combining: the retrieved context + an educational system prompt + the user's question.
  4. The prompt is sent to **Groq API** (using the `openai/gpt-oss-120b` reasoning model).
  5. The response is post-processed to extract: answer text, source citations, and confidence level (high/medium/low).

---

## 3. Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | HTML5, CSS3, Vanilla JavaScript | Custom Glassmorphism dark-mode UI with responsive layout |
| **Backend** | FastAPI (Python) | Async REST API server with automatic OpenAPI docs |
| **AI Framework** | LangChain | RAG pipeline orchestration, prompt management, chain composition |
| **LLM** | Groq API (`openai/gpt-oss-120b`) | Answer generation with advanced reasoning capabilities |
| **Embeddings** | HuggingFace (`BAAI/bge-large-en`) | Text-to-vector conversion (768 dimensions) |
| **Vector DB** | ChromaDB | Persistent vector storage & similarity search |
| **PDF Processing** | PyMuPDF (fitz) | Native text extraction with OCR fallback |
| **Containerization** | Docker | Reproducible deployment environment |
| **Hosting** | Hugging Face Spaces | Free cloud hosting with Docker SDK support |
| **Version Control** | Git + GitHub | Source code management and collaboration |

---

## 4. Project Structure

```
EDU-Agent/
│
├── api/                           # ── Backend Layer ──
│   └── server.py                  # FastAPI server: all REST endpoints, app state,
│                                  # subject management, quiz generation logic
│
├── web/                           # ── Frontend Layer ──
│   ├── index.html                 # Main page: sidebar, chat, quiz, upload tabs
│   ├── style.css                  # Glassmorphism dark-mode design system
│   └── app.js                     # Frontend logic: API calls, state management,
│                                  # quiz rendering, file upload handling
│
├── src/                           # ── AI Engine (Core) ──
│   ├── config.py                  # Centralized settings: API keys, model names,
│   │                              # chunk sizes, retrieval parameters
│   ├── extraction.py              # Phase 1: PDF → raw text with cleaning
│   ├── chunking.py                # Phase 2: raw text → semantic chunks
│   ├── embedding.py               # Phase 3-4: chunks → vectors → ChromaDB
│   ├── retrieval.py               # Phase 5a: query → similarity search → context
│   ├── llm_chain.py               # Phase 5b: context + question → LLM → answer
│   └── testing.py                 # Automated testing & quality metrics
│
├── subjects/                      # ── Runtime Data (auto-generated) ──
│   └── <subject_key>/
│       ├── pdfs/                  # Uploaded lecture PDF files
│       └── vector_db/             # ChromaDB vector database for this subject
│
├── data/                          # ── Legacy/Cached Data ──
│   ├── raw_pdfs/                  # Original PDF directory
│   ├── extracted_text/            # Cached extraction results (JSON)
│   └── processed_chunks/          # Cached chunk data (JSON)
│
├── Dockerfile                     # Docker: Python 3.10, non-root user, port 7860
├── docker-compose.yml             # Docker Compose config
├── requirements.txt               # Python dependencies (17 packages)
├── start.bat                      # One-click Windows startup script
├── .env                           # API keys (not committed to Git)
├── .gitignore                     # Excludes venv, PDFs, vector DBs, secrets
└── README.md                      # Project documentation (English)
```

---

## 5. API Reference

The backend exposes the following RESTful API endpoints via FastAPI:

### 5.1 Subject Management

| Method | Endpoint | Request Body | Response | Description |
|---|---|---|---|---|
| `GET` | `/api/subjects` | — | `{key: {name, icon, ...}}` | List all created subjects |
| `POST` | `/api/subjects` | `{name, icon}` | `{key, subject}` | Create a new subject |
| `POST` | `/api/subjects/{key}/load` | — | `{loaded, vectors}` | Load subject's vector DB into memory |
| `POST` | `/api/subjects/{key}/upload` | `FormData (files)` | `{uploaded: [...]}` | Upload PDF files to subject |
| `POST` | `/api/subjects/{key}/process` | — | `{pages, chunks, vectors}` | Full pipeline: extract → chunk → embed → store |

### 5.2 Chat (Q&A)

| Method | Endpoint | Request Body | Response | Description |
|---|---|---|---|---|
| `POST` | `/api/chat` | `{question}` | `{answer, sources, confidence}` | Ask a question about loaded lectures |

**Example Request:**
```json
{
  "question": "What is the difference between supervised and unsupervised learning?"
}
```

**Example Response:**
```json
{
  "answer": "Supervised learning uses labeled training data where each input has a corresponding correct output. The model learns to map inputs to outputs. Unsupervised learning, on the other hand, works with unlabeled data and tries to find hidden patterns or groupings...",
  "sources": [
    "Lec_3_Machine_Learning.pdf (p.12)",
    "Lec_3_Machine_Learning.pdf (p.15)"
  ],
  "confidence": "high"
}
```

### 5.3 Quiz Generation

| Method | Endpoint | Request Body | Response | Description |
|---|---|---|---|---|
| `GET` | `/api/lectures` | — | `[lecture_names]` | List all lectures in loaded subject |
| `POST` | `/api/quiz/generate` | `{lecture, regenerate}` | `{questions, total_bank}` | Generate MCQ questions from a lecture |

**Quiz Response Structure:**
```json
{
  "questions": [
    {
      "id": 1,
      "question": "What is the primary advantage of using a hash table?",
      "choices": {
        "A": "O(1) average time complexity for lookups",
        "B": "O(log n) worst case for insertions",
        "C": "Guaranteed sorted order of elements",
        "D": "Minimal memory usage"
      },
      "correct": "A"
    }
  ],
  "total_bank": 15
}
```

---

## 6. Frontend Design

The frontend uses a custom **Glassmorphism** dark-mode design with no external UI frameworks — 100% custom HTML/CSS/JS.

### 6.1 UI Sections

| Tab | Icon | Function |
|---|---|---|
| **Chat** | 💬 | Interactive Q&A with the AI agent. Displays answers with source citations and confidence level badges (🟢 High, 🟡 Medium, 🔴 Low). Includes typing indicator animation. |
| **Quiz** | 📝 | Select a lecture → Generate 5 MCQ questions from the question bank → Answer interactively with instant correct/wrong visual feedback → Track score with animated progress bar. |
| **Upload** | 📁 | Drag-and-drop PDF upload zone with file list management → "Process Lectures" button triggers the full RAG pipeline → Real-time progress indicator with step-by-step log messages. |

### 6.2 Design System

- **Color Palette:** Deep navy gradients (`#0a0e1a` → `#1a1e3a`) with pink/red accent (`#e74c6f`)
- **Glass Effect:** Semi-transparent cards with `backdrop-filter: blur()` and subtle borders
- **Typography:** Inter (UI text) + Cairo (Arabic support) from Google Fonts
- **Animations:** Smooth transitions on all interactive elements, typing dots animation, card fade-in effects
- **Responsive:** Fully responsive layout with collapsible sidebar for mobile devices

---

## 7. Deployment Architecture

```
┌──────────────┐         ┌──────────────┐         ┌──────────────────────────┐
│  Developer   │  git    │   GitHub     │  auto   │  Hugging Face Spaces     │
│  (VS Code)   │──push──▶│  Repository  │─deploy─▶│                          │
│              │         │              │         │  ┌────────────────────┐  │
└──────────────┘         └──────────────┘         │  │ Docker Container   │  │
                                                   │  │ Python 3.10-slim  │  │
                                                   │  │                    │  │
                                                   │  │ FastAPI Server     │  │
                                                   │  │ Port 7860          │  │
                                                   │  │                    │  │
                                                   │  │ ChromaDB (local)   │  │
                                                   │  └────────┬───────────┘  │
                                                   └───────────┼──────────────┘
                                                               │
                                          ┌────────────────────┼────────────────┐
                                          │                    │                │
                                   ┌──────┴──────┐     ┌──────┴──────┐
                                   │  Groq API   │     │ HuggingFace │
                                   │  (LLM)      │     │ Hub (Model) │
                                   └─────────────┘     └─────────────┘
```

**Deployment Details:**
- **Platform:** Hugging Face Spaces (Free tier, Docker SDK)
- **Base Image:** `python:3.10-slim`
- **Security:** Non-root user inside the container (UID 1000)
- **Port:** 7860 (Hugging Face mandatory default)
- **Environment Variables:** `GROQ_API_KEY` configured via HF Spaces Settings → Variables & Secrets
- **Auto-deploy:** Every `git push` to the `hf` remote triggers an automatic container rebuild and redeployment
- **Cold Start:** First request after idle may take ~30 seconds as the container wakes up

---

## 8. How to Run Locally

### 8.1 Prerequisites
- Python 3.10 or higher
- Git
- A free Groq API key (get one at [console.groq.com](https://console.groq.com))

### 8.2 Quick Start — Windows (One-Click)

```bash
# 1. Clone the repository
git clone https://github.com/The-lucky-Survivor/EDU-Agent.git
cd EDU-Agent

# 2. Create .env file with your API key
echo GROQ_API_KEY=gsk_your_api_key_here > .env

# 3. Double-click start.bat — it handles everything automatically!
```

The `start.bat` script performs the following automatically:
1. Creates a Python virtual environment (`venv_win`)
2. Installs all dependencies from `requirements.txt`
3. Checks for `.env` file and prompts for API key if missing
4. Launches the FastAPI server via Uvicorn
5. Opens `http://localhost:8000` in the default browser

### 8.3 Manual Start — Mac / Linux

```bash
git clone https://github.com/The-lucky-Survivor/EDU-Agent.git
cd EDU-Agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "GROQ_API_KEY=gsk_your_api_key_here" > .env
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### 8.4 Docker

```bash
docker build -t edu-agent .
docker run -p 7860:7860 -e GROQ_API_KEY=gsk_your_key edu-agent
```

---

## 9. Configuration Reference

All configurable settings are centralized in `src/config.py`:

### 9.1 Model Settings

| Setting | Default Value | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-large-en` | HuggingFace sentence embedding model |
| `EMBEDDING_PROVIDER` | `huggingface` | Embedding provider (huggingface or openai) |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Groq reasoning LLM model |
| `LLM_PROVIDER` | `groq` | LLM provider (groq, openai, or anthropic) |
| `LLM_TEMPERATURE` | `1.0` | Sampling temperature (1.0 recommended for reasoning) |

### 9.2 RAG Pipeline Settings

| Setting | Default Value | Description |
|---|---|---|
| `CHUNK_SIZE` | 700 | Characters per text chunk |
| `CHUNK_OVERLAP` | 150 | Overlap between consecutive chunks |
| `RETRIEVER_TOP_K` | 10 | Number of initial retrieval candidates |
| `RETRIEVER_FINAL_K` | 5 | Final number of results after filtering |
| `SIMILARITY_SEARCH_K` | 5 | Default K for basic similarity search |

### 9.3 Groq-Specific Settings

| Setting | Default Value | Description |
|---|---|---|
| `GROQ_REASONING_EFFORT` | `medium` | Reasoning depth: low, medium, or high |
| `GROQ_MAX_COMPLETION_TOKENS` | 8192 | Maximum tokens in model response |
| `GROQ_STREAM` | `True` | Enable response streaming |

---

## 10. User Flow (How It Works for Students)

```
Step 1: Open the website
    └──▶ Select or create a subject (e.g., "Machine Learning")

Step 2: Upload lectures
    └──▶ Go to Upload tab → Drag & drop PDF files → Click "Process Lectures"
    └──▶ System extracts text → creates chunks → generates embeddings → stores in ChromaDB
    └──▶ Progress bar shows real-time status

Step 3: Ask questions (Chat tab)
    └──▶ Type: "What is gradient descent?"
    └──▶ System searches the knowledge base → retrieves relevant chunks
    └──▶ LLM generates an answer using ONLY the lecture content
    └──▶ Answer includes source citations and confidence level

Step 4: Practice exams (Quiz tab)
    └──▶ Select a lecture from the dropdown
    └──▶ Click "Generate Quiz" → AI creates 5 MCQ questions
    └──▶ Click answers → instant feedback (green = correct, red = wrong)
    └──▶ Score tracker shows progress
    └──▶ Click "New Set" for different questions from the same bank
```

---

## 11. Known Limitations & Future Work

### Current Limitations
| Limitation | Explanation |
|---|---|
| **Shared state** | All users share the same knowledge base. This is by design — it's a shared educational platform where the instructor uploads content once and all students benefit. |
| **Ephemeral storage** | On Hugging Face free tier, uploaded data is lost when the container restarts (typically after ~48 hours of inactivity). |
| **No authentication** | No login system — the app is open access for simplicity. |
| **English-optimized** | The embedding model (`bge-large-en`) is optimized for English. Arabic text works but with reduced accuracy. |

### Future Enhancements
- 🔐 User authentication with per-user isolated databases
- 📊 Analytics dashboard for tracking student engagement and weak topics
- 🖼️ Full OCR integration for scanned PDF lectures (Tesseract)
- 🌍 Multi-language embedding models for Arabic/French support
- 📱 Progressive Web App (PWA) for mobile offline access
- 🔄 Real-time streaming responses for better user experience
- 📈 Automatic difficulty grading for generated quiz questions

---

## 12. Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| ChromaDB SQLite error on cloud | Old SQLite version on Linux servers | Automatically fixed via `pysqlite3-binary` module swap in `server.py` |
| `protobuf` descriptor error | Version incompatibility with newer protobuf | Pinned to `protobuf<=3.20.3` in `requirements.txt` |
| Slow responses (10–30 seconds) | Reasoning model thinking time | Normal behavior — the model analyzes context deeply before answering |
| Process button not responding | Browser caching old JavaScript | Hard refresh with `Ctrl+F5` to clear browser cache |
| "No database found" after restart | Ephemeral storage on free hosting | Re-upload and process lectures (data doesn't persist on container restart) |

---

## 13. Credits & Technologies

| Component | Provider | License |
|---|---|---|
| LLM Inference | [Groq Cloud](https://groq.com) | Free API tier |
| Embedding Model | [BAAI/bge-large-en](https://huggingface.co/BAAI/bge-large-en) | MIT License |
| Vector Database | [ChromaDB](https://www.trychroma.com/) | Apache 2.0 |
| PDF Processing | [PyMuPDF](https://pymupdf.readthedocs.io/) | AGPL-3.0 |
| AI Framework | [LangChain](https://langchain.com/) | MIT License |
| Web Server | [FastAPI](https://fastapi.tiangolo.com/) | MIT License |
| Cloud Hosting | [Hugging Face Spaces](https://huggingface.co/spaces) | Free tier |

---

> **Built with ❤️ for students** | EDU Agent © 2026
