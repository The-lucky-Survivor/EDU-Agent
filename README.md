# 🎓 Educational AI Agent - RAG-Based Lecture Assistant

An intelligent educational assistant built with **Retrieval-Augmented Generation (RAG)** that answers student questions based on lecture PDFs. Upload your lecture slides, and the AI will answer questions using ONLY the provided content.

---

## 📚 Course Coverage

This agent is loaded with **8 lectures** on **Multi-Agent Systems**:

| File | Chapter | Topic | Pages |
|------|---------|-------|-------|
| L.1.pdf | Ch.1 | Introduction to Agents | 28 |
| L.2.pdf | Ch.2 | Sensors & Actuators | 45 |
| Chapter+3 | Ch.3 | Types of Intelligent Agents | 40 |
| L.4.pdf | Ch.4 | Reinforcement Learning | 46 |
| Chapter+5 | Ch.5 | Markov Decision Processes (MDPs) | 41 |
| Chapter+6 | Ch.6 | Q-Learning | 28 |
| Chapter+7 | Ch.7 | Examples on Reinforcement Learning | 42 |
| Chapter+8 | Ch.8 | Scaling Planning for Complex Tasks | 31 |

**Total: 301 pages extracted → 245 chunks → 245 vectors**

---

## 🏗️ Architecture

```
[PDF Files] → [Text Extraction] → [Chunking] → [Embeddings] → [Vector DB (Chroma)]
                                                                    ↑
[User Question] → [Embedding] → [Similarity Search] → [Top-K Chunks]
                                                                    |
                                        [LLM Prompt (Question + Context)] → [Generated Answer] → [User]
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| PDF Extraction | PyMuPDF (fitz) |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| Embeddings | OpenAI text-embedding-3-large (or HuggingFace) |
| Vector DB | ChromaDB |
| LLM | GPT-4o / GPT-4o-mini (or Groq Llama3) |
| UI | Streamlit |
| Deployment | Docker + Docker Compose |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip
- (Optional) Docker & Docker Compose

### Option 1: Local Setup

```bash
# 1. Clone/navigate to the project
cd edu-agent

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set API keys
cp .env.example .env
# Edit .env with your API keys

# 5. Run the application
streamlit run src/app.py
```

### Option 2: Docker (Recommended)

```bash
# 1. Set API keys in .env
# 2. Build and run
docker-compose up -d

# Access at http://localhost:8501
```

---

## 🔑 API Keys Setup

Edit `.env` file with your API keys:

```bash
# Option A: OpenAI (Recommended)
OPENAI_API_KEY=sk-your-key-here

# Option B: Groq (Fast, free tier available)
GROQ_API_KEY=gsk-your-key-here

# Optional: Cohere (for re-ranking)
COHERE_API_KEY=your-cohere-key
```

### Get Free API Keys:
- **Groq**: https://console.groq.com (FREE tier with Llama 3)
- **OpenAI**: https://platform.openai.com

---

## 📁 Project Structure

```
edu-agent/
├── data/
│   ├── raw_pdfs/              # Upload your PDF lectures here
│   ├── extracted_text/        # Extracted text (JSON)
│   └── processed_chunks/      # Text chunks (JSON)
├── src/
│   ├── __init__.py
│   ├── config.py              # All configuration
│   ├── extraction.py          # Phase 1: PDF text extraction
│   ├── chunking.py            # Phase 2: Text chunking
│   ├── embedding.py           # Phase 3: Embeddings & vector store
│   ├── retrieval.py           # Phase 4: RAG pipeline & prompts
│   ├── llm_chain.py           # Phase 5: LLM & response generation
│   ├── app.py                 # Phase 6: Streamlit UI
│   └── testing.py             # Phase 7: Evaluation suite
├── vector_db/                 # ChromaDB storage
├── tests/                     # Unit tests
├── .env                       # API keys (not in git)
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🔧 Configuration

All settings are centralized in `src/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNK_SIZE` | 700 | Characters per chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks |
| `EMBEDDING_PROVIDER` | openai | openai / huggingface |
| `LLM_PROVIDER` | openai | openai / groq |
| `LLM_TEMPERATURE` | 0.1 | Low = factual, High = creative |
| `RETRIEVER_TOP_K` | 10 | Initial candidates |
| `RETRIEVER_FINAL_K` | 5 | Final results after re-ranking |

---

## 🧪 Testing

Run the evaluation suite:

```python
from src.testing import run_full_evaluation, print_report

# After creating chain
report = run_full_evaluation(vectorstore, chain)
print_report(report)
```

### Quality Gates

| Gate | Criteria | Status |
|------|----------|--------|
| QG1 | All PDFs extracted, no empty pages | ✅ PASS |
| QG2 | Avg chunk 300-700 chars, all metadata | ✅ PASS |
| QG3 | Vectors match chunks, search works | ✅ PASS |
| QG4 | Relevant retrievals, prompt quality | ✅ PASS |
| QG5 | Factual answers, < 5s response | ✅ PASS |

---

## 🐳 Docker Deployment

### Build & Run

```bash
# Build image
docker build -t edu-agent .

# Run container
docker run -p 8501:8501 --env-file .env edu-agent

# Or use docker-compose
docker-compose up -d
```

### Deploy to Render/Railway

```bash
# 1. Push to GitHub
# 2. Connect repo to Render
# 3. Set environment variables
# 4. Deploy as Web Service (port 8501)
```

---

## ⚠️ Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| "No LLM API key found" | Missing API key | Set OPENAI_API_KEY or GROQ_API_KEY in .env |
| Slow embeddings | HuggingFace downloading | First run downloads model; subsequent runs are fast |
| Out of memory | Large PDFs | Reduce CHUNK_SIZE in config.py |
| Arabic text issues | Encoding | Ensure UTF-8 encoding |
| Port already in use | 8501 occupied | Change port: `--server.port=8502` |

---

## 📝 License

Educational use only. Built for AI course at university.

---

**Built with ❤️ using LangChain + ChromaDB + Streamlit**
