# RAG Pipeline — pgvector + Ollama + FastAPI + Kubernetes

A production-ready Retrieval-Augmented Generation (RAG) system built from scratch. Ingests PDF documents into a PostgreSQL vector database and exposes a REST API for conversational Q&A over those documents — fully containerized and orchestrated with Kubernetes.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                │
│                                                     │
│  ┌──────────┐   ┌──────────┐   ┌─────────────────┐ │
│  │ postgres │   │  ollama  │   │    rag-api       │ │
│  │ pgvector │   │ llama3.2 │   │    FastAPI       │ │
│  └──────────┘   └──────────┘   └─────────────────┘ │
│                                        ↕            │
│                              NodePort :30800        │
└─────────────────────────────────────────────────────┘
```

### RAG Pipeline

```
PDF Documents
     ↓
Text Extraction (PyMuPDF)
     ↓
Semantic Chunking
     ↓
Embeddings (BAAI/bge-m3)
     ↓
pgvector (PostgreSQL)
     ↓
Query → Rewrite → Search → Rerank → Generate
              (llama3.2)  (bge-m3) (bge-reranker-v2-m3) (llama3.2)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | `BAAI/bge-m3` (local, multilingual) |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| LLM | `llama3.2` via Ollama |
| Vector DB | PostgreSQL + pgvector |
| API | FastAPI + Uvicorn |
| ORM | SQLAlchemy 2.0 + psycopg3 |
| Packaging | uv |
| Containers | Docker |
| Orchestration | Kubernetes (Minikube) |

---

## Project Structure

```
├── ingest.py                   # PDF ingestion entrypoint
├── rag.py                      # Terminal chat entrypoint
├── src/
│   ├── models/
│   │   └── embeddings.py       # load_model(), generate_embeddings()
│   ├── persistence/
│   │   ├── schema.py           # ORM — DocumentChunk
│   │   ├── engine.py           # build_engine(), initialize_database()
│   │   └── chunks.py           # load_pdfs(), index_chunks(), create_hnsw_index()
│   ├── retrieval/
│   │   ├── search.py           # cosine similarity search (pgvector)
│   │   ├── reranker.py         # cross-encoder reranking
│   │   └── query.py            # query rewriting with conversation context
│   ├── generation/
│   │   └── llm.py              # answer generation via Ollama
│   ├── chat/
│   │   ├── history.py          # ConversationHistory (per-session)
│   │   └── loop.py             # terminal chat loop
│   ├── api/
│   │   ├── main.py             # FastAPI app
│   │   ├── schemas.py          # Pydantic request/response models
│   │   └── session.py          # SessionManager (in-memory, thread-safe)
│   └── utils/
│       └── constants.py        # shared configuration constants
├── .docker/
│   ├── Dockerfile.ingest
│   ├── Dockerfile.rag
│   ├── Dockerfile.api
│   └── docker-compose.yml
└── k8s/
    ├── secret.yaml
    ├── configmap.yaml
    ├── postgres.yaml
    ├── ollama.yaml
    ├── ingest.yaml
    └── rag-api.yaml
```

---

## Getting Started

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/download)
- [Minikube](https://minikube.sigs.k8s.io/docs/start/) (for Kubernetes)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- NVIDIA GPU (optional, recommended for performance)

### 1. Configure environment

```bash
cp .env.example .env
# edit .env with your settings
```

### 2. Run with Docker Compose (local)

```bash
# Start Postgres and Ollama
docker compose -f .docker/docker-compose.yml --env-file .env up postgres ollama -d

# Pull the LLM model
docker compose -f .docker/docker-compose.yml --env-file .env run ollama-pull

# Ingest PDFs
docker compose -f .docker/docker-compose.yml --env-file .env run ingest

# Start the API
docker compose -f .docker/docker-compose.yml --env-file .env up rag-api
```

### 3. Run with Kubernetes (Minikube)

```bash
# Start the cluster
minikube start --driver=docker --memory=14000

# Load local images into Minikube
minikube image load docker-ingest:latest
minikube image load docker-rag-api:latest

# Apply manifests
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/postgres.yaml
kubectl wait --for=condition=ready pod -l app=postgres --timeout=60s
kubectl apply -f k8s/ollama.yaml
kubectl apply -f k8s/rag-api.yaml

# Copy PDFs into Minikube and run ingestion via port-forward
minikube ssh "sudo mkdir -p /data/pdfs"
minikube cp ./pdf_folder/<file>.pdf /data/pdfs/<file>.pdf

kubectl port-forward svc/postgres 5433:5432  # Terminal 1
uv run python ingest.py --pdf-folder-location ./pdf_folder \  # Terminal 2
  --db-host localhost --db-port 5433 \
  --db-name rag_db --db-user postgres --db-password postgres

# Expose the API
minikube service rag-api --url
```

---

## API Endpoints

### `GET /health`
Returns API and database status.

```json
{"status": "ok", "database": "healthy"}
```

### `GET /documents`
Lists all indexed PDFs and their chunk counts.

```json
[
  {"source": "document.pdf", "chunk_count": 312}
]
```

### `POST /chat`
Sends a question and returns an answer grounded in the indexed documents. Maintains conversation history per `session_id`.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user1", "question": "What is logistic regression?"}'
```

```json
{
  "session_id": "user1",
  "question": "What is logistic regression?",
  "answer": "Logistic regression is a discriminative classification model..."
}
```

Interactive API docs available at `http://localhost:8000/docs`.

---

## How It Works

**Ingestion (`ingest.py`)**
PDFs are extracted with PyMuPDF, split into overlapping semantic chunks, encoded with `BAAI/bge-m3`, and stored in PostgreSQL with a pgvector HNSW index for fast approximate nearest-neighbor search.

**Retrieval**
At query time, the user's question is optionally rewritten using conversation history, then encoded into a vector. The top-K most similar chunks are retrieved via cosine similarity and reranked with `BAAI/bge-reranker-v2-m3` to select the most relevant passages.

**Generation**
The reranked context and conversation history are passed to `llama3.2` via Ollama, which generates a grounded answer. Sessions are maintained in memory per `session_id`, enabling multi-turn conversations.
