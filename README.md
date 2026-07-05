cat > /mnt/user-data/outputs/README.md << 'ENDOFFILE'
# rag-pgvector

A production-ready Retrieval-Augmented Generation (RAG) system. Ingests PDF documents into a PostgreSQL vector database and exposes a REST API for conversational Q&A — fully containerized with Docker and orchestrated with Kubernetes.

---

## System Context

```mermaid
C4Context
    title System Context — RAG Pipeline

    Person(user, "User", "Asks questions via REST API")

    System(rag, "RAG API", "Retrieves context from documents and generates answers")

    System_Ext(ollama, "Ollama", "Runs llama3.2 locally for LLM inference")
    System_Ext(hf, "Hugging Face", "Provides embedding and reranker models")
    SystemDb_Ext(pg, "PostgreSQL + pgvector", "Stores document chunks and vector embeddings")

    Rel(user, rag, "POST /chat", "HTTP/JSON")
    Rel(rag, pg, "Vector similarity search", "SQL")
    Rel(rag, ollama, "Query rewriting + answer generation", "HTTP")
    Rel(rag, hf, "Downloads models on startup", "HTTPS")
```

---

## Container Diagram

```mermaid
C4Container
    title Container Diagram — Kubernetes Cluster

    Person(user, "User")

    System_Boundary(k8s, "Kubernetes (Minikube)") {
        Container(api, "rag-api", "FastAPI + Uvicorn", "Exposes /chat, /health, /documents. Runs embedding model and reranker.")
        Container(ollama, "ollama", "Ollama", "Serves llama3.2 for query rewriting and answer generation.")
        ContainerDb(pg, "postgres", "PostgreSQL 16 + pgvector", "Stores document chunks with HNSW vector index.")
        Container(ingest, "ingest (Job)", "Python", "Extracts PDFs, generates embeddings and indexes chunks. Runs once.")
    }

    Rel(user, api, "POST /chat", "NodePort :30800")
    Rel(api, pg, "Similarity search + CRUD", "SQL / psycopg3")
    Rel(api, ollama, "Rewrite query + generate answer", "HTTP :11434")
    Rel(ingest, pg, "Insert chunks + embeddings", "SQL / psycopg3")
```

---

## RAG Pipeline

```mermaid
flowchart TD
    A[PDF Documents] --> B[Text Extraction\nPyMuPDF]
    B --> C[Semantic Chunking\n800 chars · 120 overlap]
    C --> D[Embeddings\nBAAI/bge-m3]
    D --> E[(PostgreSQL\npgvector · HNSW index)]

    F([User Question]) --> G{Has history?}
    G -- Yes --> H[Query Rewriting\nllama3.2]
    G -- No --> I[Original Query]
    H --> J[Encode Query\nBAAI/bge-m3]
    I --> J
    J --> K[Cosine Similarity Search\ntop-30 chunks]
    E --> K
    K --> L[Cross-Encoder Reranking\nBAAI/bge-reranker-v2-m3\ntop-3 chunks]
    L --> M[Answer Generation\nllama3.2 via Ollama]
    M --> N([Answer])
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | `BAAI/bge-m3` (local, multilingual) |
| Reranker | `BAAI/bge-reranker-v2-m3` (Cross-Encoder) |
| LLM | `llama3.2` via Ollama |
| Vector DB | PostgreSQL 16 + pgvector (HNSW index) |
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
- [Minikube](https://minikube.sigs.k8s.io/docs/start/) *(for Kubernetes)*
- [kubectl](https://kubernetes.io/docs/tasks/tools/) *(for Kubernetes)*
- NVIDIA GPU *(optional, recommended)*

### 1. Configure environment

```bash
cp .env.example .env
# edit .env with your settings
```

### 2. Run with Docker Compose

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
# Start cluster
minikube start --driver=docker --memory=14000

# Load local images
minikube image load docker-ingest:latest
minikube image load docker-rag-api:latest

# Apply manifests
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/postgres.yaml
kubectl wait --for=condition=ready pod -l app=postgres --timeout=60s
kubectl apply -f k8s/ollama.yaml
kubectl apply -f k8s/rag-api.yaml

# Copy PDFs and run ingestion via port-forward
minikube ssh "sudo mkdir -p /data/pdfs"
minikube cp ./pdf_folder/<file>.pdf /data/pdfs/<file>.pdf

# Terminal 1
kubectl port-forward svc/postgres 5433:5432

# Terminal 2
uv run python ingest.py \
  --pdf-folder-location ./pdf_folder \
  --db-host localhost --db-port 5433 \
  --db-name rag_db --db-user postgres --db-password postgres

# Expose API
minikube service rag-api --url
```

---

## API Reference

### `GET /health`

```json
{"status": "ok", "database": "healthy"}
```

### `GET /documents`

```json
[{"source": "document.pdf", "chunk_count": 312}]
```

### `POST /chat`

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

Interactive docs: `http://localhost:8000/docs`
ENDOFFILE