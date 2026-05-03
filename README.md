---
title: EDU Agent
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🎓 EDU Agent — AI-Powered Educational Assistant

An intelligent web application built on **RAG (Retrieval-Augmented Generation)** that helps students understand their lecture materials and generate practice exam questions (MCQ). The agent reads PDF files, indexes their content, and answers questions exclusively from the studied material.

🔗 **[Live Demo](https://tinyurl.com/edu-agent-ap)** · 💻 **[Source Code](https://github.com/The-lucky-Survivor/EDU-Agent)**

---

## ✨ Key Features

- 🔄 **Persistent Knowledge Base** — Vector database is saved between sessions. No need to re-upload lectures.
- 📚 **Multi-Subject Support** — Create separate subjects, each with its own files and isolated database.
- 📝 **Smart MCQ Generator** — Generates 5–20 academic MCQ questions per lecture focusing on concepts, algorithms, and definitions.
- 🧠 **Advanced Reasoning** — Powered by Groq's reasoning models (`openai/gpt-oss-120b`) for analytical, in-depth answers.
- 📄 **Robust PDF Processing** — Handles multi-page continuity, text cleaning, and OCR fallback for scanned documents.
- 🌐 **Cloud Deployed** — Fully hosted on Hugging Face Spaces — accessible from any device.

---

## 🏗️ Architecture

```
[PDF Files] → [Text Extraction] → [Smart Chunking] → [Embedding (HuggingFace)] → [ChromaDB]
                                                                    ↑
[Student Question] → [Query Embedding] → [Similarity Search] → [Top-K Chunks]
                                                                    |
                     [Groq LLM + System Prompt] ← (Question + Context) → [Accurate Answer]
```

### Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | HTML5, CSS3 (Glassmorphism Dark Mode), Vanilla JavaScript |
| **Backend** | FastAPI (Python) |
| **AI Framework** | LangChain |
| **LLM** | Groq API (`openai/gpt-oss-120b` reasoning model) |
| **Embeddings** | HuggingFace (`BAAI/bge-large-en`) |
| **Vector DB** | ChromaDB |
| **PDF Processing** | PyMuPDF (fitz) |
| **Deployment** | Docker → Hugging Face Spaces |

---

## 🚀 Quick Start (Local Development)

### Prerequisites
- Python 3.10+
- A free Groq API key from [console.groq.com](https://console.groq.com)

### Windows (One-Click)

```bash
# 1. Clone the repository
git clone https://github.com/The-lucky-Survivor/EDU-Agent.git
cd EDU-Agent

# 2. Create .env file
echo GROQ_API_KEY=gsk_your_api_key_here > .env

# 3. Double-click start.bat — it handles everything!
```

The `start.bat` script automatically:
1. Creates a Python virtual environment
2. Installs all dependencies
3. Launches the FastAPI server at **http://localhost:8000**

### Mac / Linux

```bash
git clone https://github.com/The-lucky-Survivor/EDU-Agent.git
cd EDU-Agent
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "GROQ_API_KEY=gsk_your_api_key_here" > .env
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

---

## 📁 Project Structure

```
EDU-Agent/
├── api/
│   └── server.py              # FastAPI server & all REST endpoints
├── web/
│   ├── index.html             # Main page structure
│   ├── style.css              # Glassmorphism dark-mode design
│   └── app.js                 # Frontend logic & API interactions
├── src/
│   ├── config.py              # Centralized settings & constants
│   ├── extraction.py          # PDF text extraction & cleaning
│   ├── chunking.py            # Text splitting into semantic chunks
│   ├── embedding.py           # Vector embedding & ChromaDB storage
│   ├── retrieval.py           # Similarity search & prompt engineering
│   ├── llm_chain.py           # LLM connection & response processing
│   └── testing.py             # Automated testing & quality metrics
├── subjects/                  # Per-subject data (auto-generated)
├── Dockerfile                 # Docker container definition
├── docker-compose.yml         # Docker Compose config
├── requirements.txt           # Python dependencies
├── start.bat                  # One-click Windows startup script
└── .env                       # API keys (not tracked by Git)
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/subjects` | List all subjects |
| `POST` | `/api/subjects` | Create a new subject |
| `POST` | `/api/subjects/{key}/load` | Load a subject's knowledge base |
| `POST` | `/api/subjects/{key}/upload` | Upload PDF files |
| `POST` | `/api/subjects/{key}/process` | Process PDFs into vector database |
| `POST` | `/api/chat` | Ask a question |
| `GET` | `/api/lectures` | List lectures in current subject |
| `POST` | `/api/quiz/generate` | Generate MCQ quiz from a lecture |

---

## ⚙️ Configuration

All settings are in `src/config.py`:

| Setting | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-large-en` | HuggingFace embedding model |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Groq reasoning model |
| `CHUNK_SIZE` | 700 | Characters per text chunk |
| `CHUNK_OVERLAP` | 150 | Overlap between chunks |
| `GROQ_REASONING_EFFORT` | `medium` | Reasoning depth (low/medium/high) |

---

## 🐳 Docker Deployment

```bash
# Build and run locally
docker build -t edu-agent .
docker run -p 7860:7860 -e GROQ_API_KEY=gsk_your_key edu-agent

# Or use Docker Compose
docker-compose up
```

For Hugging Face Spaces deployment, push to the `hf` remote — it auto-builds from the Dockerfile.

---

## ⚠️ Troubleshooting

| Issue | Solution |
|---|---|
| ChromaDB SQLite error | Automatically fixed via `pysqlite3-binary` in the codebase |
| `protobuf` descriptor error | Pinned to `protobuf<=3.20.3` in requirements |
| Slow responses (10–30s) | Normal — reasoning model analyzes context deeply before answering |
| Process button not responding | Hard refresh with `Ctrl+F5` to clear browser cache |

---

## 📜 License & Credits

| Component | Provider |
|---|---|
| LLM Inference | [Groq](https://groq.com) (Free tier) |
| Embeddings | [BAAI/bge-large-en](https://huggingface.co/BAAI/bge-large-en) |
| Vector DB | [ChromaDB](https://www.trychroma.com/) |
| AI Framework | [LangChain](https://langchain.com/) |
| Hosting | [Hugging Face Spaces](https://huggingface.co/spaces) |

---

**Built with ❤️ for students** | EDU Agent © 2026
