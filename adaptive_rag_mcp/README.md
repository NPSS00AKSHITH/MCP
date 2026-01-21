# Adaptive RAG MCP Server

A production-grade **Retrieval-Augmented Generation** server exposed via the **Model Context Protocol (MCP)**.

## Features

- **Document Ingestion**: PDF, Markdown, and text files with recursive chunking
- **Dense Retrieval**: Vector search using sentence-transformers + FAISS
- **Sparse Retrieval**: BM25 keyword search
- **Hybrid Search**: Reciprocal Rank Fusion combining dense + sparse
- **Reranking**: Cross-encoder re-ranking with quality signals
- **MCP API**: FastAPI server with 17 tools for complete RAG workflows

## Quick Start

```bash
# Install with uv
uv venv
uv pip install -e .

# Configure
cp .env.example .env
# Edit .env: ADAPTIVE_RAG_API_KEY=your-secret-key

# Run server
.venv/Scripts/python -m src.server.main
# Server runs at http://localhost:8000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/tools` | GET | List all MCP tools |
| `/tools/{tool_name}` | POST | Execute a tool |

All tool endpoints require `X-API-Key` header.

## Core Tools

### Ingestion
```bash
# Ingest a document
curl -X POST http://localhost:8000/tools/ingest_document \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/doc.md"}'

# Or ingest raw text
curl -X POST http://localhost:8000/tools/ingest_document \
  -H "X-API-Key: $API_KEY" \
  -d '{"text": "Your content...", "doc_id": "my_doc"}'
```

### Search
```bash
# Hybrid search (default)
curl -X POST http://localhost:8000/tools/search \
  -H "X-API-Key: $API_KEY" \
  -d '{"query": "What is RAG?", "mode": "hybrid", "k": 5}'

# Modes: dense | sparse | hybrid
```

### Rerank
```bash
# Rerank with quality signals
curl -X POST http://localhost:8000/tools/rerank \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "query": "What is RAG?",
    "documents": [{"id": "d1", "content": "RAG combines..."}]
  }'
```

## Architecture

```
src/
├── ingestion/     # Document loading, chunking, storage
├── retrieval/     # Embeddings, vector search, BM25, reranking
└── server/        # FastAPI, auth, logging, tool execution
```

## Dependencies

- **Server**: FastAPI, uvicorn, pydantic
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Search**: faiss-cpu
- **Sparse Search**: rank-bm25
- **Reranking**: sentence-transformers (ms-marco cross-encoder)
- **Documents**: pypdf, markdown

## Development

```bash
# Run tests
python examples/test_ingestion.py
python examples/test_retrieval.py
python examples/test_hybrid.py
python examples/test_rerank.py
```

## Epistemic Safety

The server implements explicit epistemic safety features to ensure transparent and safe outcomes.

### Outcome Types

Every retrieval response includes an explicit outcome type:

| Outcome | When Returned | Confidence |
|---------|--------------|------------|
| `answer_ready` | High confidence evidence found, no contradictions | High |
| `partial_answer` | Moderate evidence or contradictions detected | Medium/Low |
| `insufficient_evidence` | Unable to find relevant evidence | Low |
| `clarification_needed` | Query is ambiguous or unclear | Low |

### Confidence Levels

Outcomes include a confidence level (`high`, `medium`, `low`):

- **High**: top_score ≥ 0.7, score_gap ≥ 0.15, no contradictions
- **Medium**: top_score ≥ 0.5 but < 0.7
- **Low**: top_score < 0.5 OR contradictions detected

### Contradiction Detection

The system detects when top-scoring chunks contain conflicting information:

```json
{
  "outcome": {
    "type": "partial_answer",
    "explanation": "Conflicting sources detected. Unable to merge claims safely.",
    "confidence_level": "low"
  },
  "contradiction": {
    "has_contradiction": true,
    "conflicting_chunks": [["chunk_001", "chunk_007"]],
    "explanation": "Potential contradictions detected between chunks..."
  }
}
```

### Stopping Rules

Explicit stopping rules control the retrieval loop:

| Rule | Threshold | Result |
|------|-----------|--------|
| `HIGH_CONFIDENCE` | top_score ≥ 0.7 | Stop, `answer_ready` |
| `CONTRADICTION_DETECTED` | Conflicting sources | Stop, `partial_answer` |
| `MAX_ITERATIONS` | 3 iterations | Stop, `partial_answer` or `insufficient_evidence` |
| `NO_RESULTS` | Empty results | Stop, `insufficient_evidence` |

**Important Rule**: If the system cannot clearly explain WHY an answer is correct, it will NOT return `answer_ready`. It prefers refusing over guessing.

## License

MIT

