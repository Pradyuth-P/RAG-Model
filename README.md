# AetherRAG - Modular Retrieval-Augmented Generation Engine

A complete, production-ready, and highly modular Retrieval-Augmented Generation (RAG) application. It features document ingestion, configurable chunking, local embeddings (HuggingFace), abstract vector stores (FAISS), and LLM generation (routed via Groq) with full observability traces pushed to **LangSmith**.

---

## 🏗️ Project Architecture

The system is split into a **Python FastAPI backend** and a **React (Vite) frontend**:

```
rag-system/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py           # API Endpoint controllers (Upload, Chat Stream, Docs, Health)
│   │   ├── loaders/
│   │   │   └── document_loader.py  # Page parser for PDF, DOCX, TXT, MD
│   │   ├── models/
│   │   │   └── schemas.py          # Pydantic schema validation models
│   │   ├── services/
│   │   │   ├── embedding_service.py# Local embedding wrapper (HuggingFace)
│   │   │   ├── vector_service.py   # FAISS wrapper with manifest file indexing
│   │   │   ├── llm_service.py      # LLM provider router (Groq)
│   │   │   ├── rag_service.py      # Main pipeline orchestration with LangSmith tracing
│   │   │   └── langsmith_config.py # Tracing settings and logging callbacks
│   │   └── main.py                 # FastAPI application main entrypoint
│   └── Dockerfile                  # Container instructions for FastAPI service
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatWindow.jsx      # Streaming chat display + markdown renderer
│   │   │   ├── UploadPanel.jsx     # Dropzone + uploaded documents listing
│   │   │   ├── SourcesPanel.jsx    # Inspector for matched context chunks
│   │   │   ├── HistorySidebar.jsx  # Chat session logs + config parameters preview
│   │   │   └── SettingsPanel.jsx   # LLM, temperature, and chunking tuning overlay
│   │   ├── services/
│   │   │   └── api.js              # REST endpoints + Fetch-based SSE parser
│   │   ├── App.jsx                 # App shell and dashboard state assembly
│   │   ├── index.css               # Styling and scrollbar definitions
│   │   └── main.jsx                # Bootstrap React app
│   ├── index.html                  # Main page entry with Google Fonts
│   ├── nginx.conf                  # Nginx proxy routing inside Docker
│   └── Dockerfile                  # Multi-stage Docker instructions for frontend
│
├── storage/                        # Persistent vector indexes
├── uploads/                        # Temporary uploaded files directory
├── samples/                        # Pre-packaged text/markdown files for test runs
├── docker-compose.yml              # Multi-container service orchestrator
├── requirements.txt                # Python package list
└── README.md                       # This documentation guide
```

---

## ⚡ Quick Start (Local Setup)

### 1. Configure the Environment
Clone or copy the files into your directory, then copy the environment file:
```bash
cp .env.example .env
```
Open `.env` and configure your Groq API key:
```ini
GROQ_API_KEY=your-groq-key

# Optional LangSmith Tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=Basic_RAG
```

### 2. Run the Backend
Ensure you have Python 3.10+ installed.
```bash
# Install dependencies
pip install -r requirements.txt

# Start the development server (reload mode on port 8000)
uvicorn backend.app.main:app --reload
```

### 3. Run the Frontend
In a new terminal window, initialize the Node dependencies and run the Vite client:
```bash
cd frontend

# Install package dependencies
npm install

# Start Vite client on port 5173 (which automatically proxies /api to port 8000)
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 🐳 Docker Deployment

The application includes a `docker-compose.yml` for unified, multi-container orchestration. Both systems can be launched in a single command, sharing environment settings and binding the `storage/` directory for database persistence.

```bash
# Build and run containers
docker-compose up --build
```
Once launched:
- The React application is served on [http://localhost:8080](http://localhost:8080).
- The FastAPI server runs on [http://localhost:8000](http://localhost:8000).

---

## 🛠️ API Documentation

FastAPI automatically generates interactive documentation. When the backend is running, open:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Redoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Core Endpoints:
- `POST /api/upload`: Expects a multipart form file (PDF, DOCX, TXT, MD) and returns ingestion details.
- `POST /api/chat`: Streams LLM responses chunk-by-chunk using Server-Sent Events (`text/event-stream`).
- `GET /api/documents`: Returns details of all indexed files for the active embedding provider.
- `DELETE /api/documents/{doc_id}`: Removes a document from the vector space index and deletes the source file from uploads.
- `GET /api/health`: Reports active API key presence and LangSmith tracing state.

---

## 🧪 Testing the RAG Engine

We have pre-packaged sample files inside the `samples/` directory for immediate verification:
1. **`samples/aether_project.txt`**: Contains specifications for a distributed intelligence system named *Project Aether*.
2. **`samples/rag_tuning.md`**: Outlines guidelines on optimizing chunk size and overlaps.

### Steps to Verify:
1. Open the UI dashboard and click the settings gear (bottom-left) to confirm your LLM settings.
2. In the right panel, upload `samples/aether_project.txt`. Check the progress bar; it will transition from Uploading to Extracting & Chunking.
3. Once completed, the file will appear in the **Knowledge Base Documents** table with chunk and size details.
4. Go to the chat window and ask: *“Who is the Principal AI Architect for Project Aether?”*
5. The answer will stream into the UI: *“The Principal AI Architect is Alice Vance.”*
6. Click **“View matching sources”** inside the assistant bubble to see the exact text chunk extracted from `aether_project.txt` with its similarity score and page number.
7. You can adjust the LLM parameters (such as model and temperature) inside settings and ask follow-ups.
8. Check your **LangSmith** dashboard; a tree of nested traces will be registered for the retrieval, embedding, and generation calls.
