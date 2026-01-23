# Adaptive RAG MCP Server

A production-grade **Retrieval-Augmented Generation (RAG)** server exposed via the **Model Context Protocol (MCP)**. This server enables advanced RAG capabilities for LLMs, including graph-based retrieval, adaptive memory, and parallel task planning.

## Features

- **Document Ingestion**: PDF, Markdown, and text files with recursive chunking.
- **Dense Retrieval**: Vector search using sentence-transformers + FAISS.
- **Sparse Retrieval**: BM25 keyword search.
- **Hybrid Search**: Reciprocal Rank Fusion combining dense + sparse results.
- **Reranking**: Cross-encoder re-ranking with quality signals.
- **Adaptive Memory**: Learns from user feedback to improve future retrievals.
- **Task Planning**: DAG-based parallel task execution for complex queries.
- **MCP API**: FastAPI server exposing tools for complete RAG workflows.

## Prerequisites

Before getting started, ensure you have the following installed on your system:

- **Python 3.10+**: [Download Python](https://www.python.org/downloads/)
- **Git**: [Download Git](https://git-scm.com/downloads)
- **uv** (Recommended): A fast Python package installer and resolver.
  - Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
  - macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Installation

### Option 1: Using `uv` (Recommended)

`uv` provides a fast and reliable way to manage dependencies.

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd adaptive_rag_mcp
    ```

2.  **Create a virtual environment:**
    ```bash
    uv venv
    ```

3.  **Activate the virtual environment:**
    *   **Windows**:
        ```powershell
        .venv\Scripts\activate
        ```
    *   **macOS/Linux**:
        ```bash
        source .venv/bin/activate
        ```

4.  **Install dependencies:**
    ```bash
    uv pip install -e .
    ```

### Option 2: Using standard `pip`

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd adaptive_rag_mcp
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    ```

3.  **Activate the virtual environment:**
    *   **Windows**:
        ```powershell
        .venv\Scripts\activate
        ```
    *   **macOS/Linux**:
        ```bash
        source .venv/bin/activate
        ```

4.  **Install dependencies:**
    ```bash
    pip install -e .
    ```

## Configuration

1.  **Create the environment file:**
    Copy the example environment file to `.env`:
    ```bash
    cp .env.example .env
    ```
    *On Windows PowerShell:* `copy .env.example .env`

2.  **Edit `.env`:**
    Open `.env` in your text editor and configure the following:

    ```ini
    # Security
    ADAPTIVE_RAG_API_KEY=your-secret-key-change-in-production

    # Server
    HOST=0.0.0.0
    PORT=8000
    LOG_LEVEL=INFO
    
    # Optional: LLM Provider Configuration (if using generative features)
    # GOOGLE_API_KEY=your-google-api-key
    ```
    *   **ADAPTIVE_RAG_API_KEY**: Set a secure key. usage clients must provide this key in the `X-API-Key` header.

## Running the Server

1.  **Start the server:**
    Make sure your virtual environment is activated, then run:
    ```bash
    python -m src.server.main
    ```

2.  **Verify it's running:**
    Open your browser or use curl to check the health endpoint:
    ```bash
    curl http://localhost:8000/health
    ```
    You should see `{"status":"ok"}`.

## Usage Guide

### connecting with MCP Client (e.g., Claude Desktop)

To use this server with an MCP-compliant client like [Claude Desktop](https://claude.ai/download):

1.  Depending on how you run the server, add the following to your MCP client configuration (e.g., `minio-config.json` or Claude's config):

    ```json
    {
      "mcpServers": {
        "adaptive-rag": {
          "command": "python",
          "args": ["-m", "src.server.main"],
          "env": {
            "ADAPTIVE_RAG_API_KEY": "your-configured-key"
          }
        }
      }
    }
    ```
    *Note: Ensure `python` points to your virtual environment python, e.g., `D:/MCP/adaptive_rag_mcp/.venv/Scripts/python.exe`.*

### Direct API Usage

You can interact with the server directly using `curl` or any HTTP client.

#### 1. Ingest a Document
Upload a text file or PDF to be indexed.
```bash
curl -X POST http://localhost:8000/tools/ingest_document \
  -H "X-API-Key: your-secret-key-change-in-production" \
  -d '{"text": "Retrieval Augmented Generation reduces hallucinations.", "doc_id": "rag_intro"}'
```

#### 2. Search (Hybrid)
Perform a search query.
```bash
curl -X POST http://localhost:8000/tools/search \
  -H "X-API-Key: your-secret-key-change-in-production" \
  -d '{"query": "benefits of RAG", "mode": "hybrid", "k": 3}'
```

#### 3. Rerank Results
Improve ranking of retrieved documents.
```bash
curl -X POST http://localhost:8000/tools/rerank \
  -H "X-API-Key: your-secret-key-change-in-production" \
  -d '{
    "query": "benefits of RAG",
    "documents": [
        {"id": "doc1", "content": "RAG improves accuracy."},
        {"id": "doc2", "content": "Bananas are yellow."}
    ]
  }'
```

## Troubleshooting

- **`ModuleNotFoundError`**: Ensure you have activated the virtual environment and installed dependencies (`pip install -e .`).
- **`403 Forbidden`**: Check that the `X-API-Key` header in your request matches the `ADAPTIVE_RAG_API_KEY` in your `.env` file.
- **Port Conflict**: If port 8000 is in use, change the `PORT` variable in `.env`.

## Production-Grade Features

This server follows [MCP Best Practices](https://modelcontextprotocol.io/) for production deployments:

### Tool Annotations

All tools include MCP annotations to help clients understand tool behavior:

| Annotation | Description |
|------------|-------------|
| `readOnlyHint` | Tool does not modify state |
| `destructiveHint` | Tool may perform destructive operations |
| `idempotentHint` | Repeated calls have no additional effect |
| `openWorldHint` | Tool interacts with external services |

### Security Configuration

```ini
# .env security settings
ENVIRONMENT=production              # Enable production mode
ADAPTIVE_RAG_API_KEY=<secure-key>  # MUST change from default!
HOST=127.0.0.1                      # Bind to localhost (safer)
CORS_ALLOW_ALL=false                # Restrict CORS origins
CORS_ORIGINS=["https://your-app.com"]
```

### Pagination

List endpoints support pagination:

```bash
curl -X POST http://localhost:8000/tools/list_documents \
  -H "X-API-Key: your-key" \
  -d '{"limit": 20, "offset": 0}'
```

Response includes pagination metadata:
```json
{
  "documents": [...],
  "pagination": {
    "total": 150,
    "count": 20,
    "offset": 0,
    "has_more": true,
    "next_offset": 20
  }
}
```

### Enhanced Error Responses

Errors include actionable suggestions:

```json
{
  "error_code": "TOOL_NOT_FOUND",
  "message": "Tool 'serch' not found",
  "suggestion": "Did you mean: search? Use GET /tools to list all available tools.",
  "details": {"requested_tool": "serch"}
}
```

### Code Quality

Install dev dependencies and run quality checks:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run code quality checks
black --check src/ tests/
isort --check-only src/ tests/
ruff check src/ tests/
mypy src/
```

### MCP Evaluation Suite

Test the server's LLM effectiveness:

```bash
python scripts/run_evaluation.py tests/evaluation.xml -u http://localhost:8000
```


## Deployment & Remote Usage

If you want to use this MCP server from another device (e.g., accessing your customized RAG server from a laptop), you have two options:

### Option 1: Install on the Target Device
Simply follow the **Installation** instructions above on the new device. This allows you to run a completely independent instance of the server.

### Option 2: Access Remotely via Network
You can run the server on one machine (Host) and access it from another (Client).

1.  **Configure Host**:
    Ensure your `.env` file has `HOST=0.0.0.0`. This binds the server to all network interfaces, not just localhost.
    ```ini
    HOST=0.0.0.0
    ```

2.  **Get Host IP**:
    Find the local IP address of your host machine (e.g., `ipconfig` on Windows or `ifconfig` on Linux). Let's say it is `192.168.1.15`.

3.  **Access from Client**:
    You can now send requests to the host's IP address.
    ```bash
    curl -X POST http://192.168.1.15:8000/tools/search \
      -H "X-API-Key: your-secret-key" \
      -d '{"query": "hello"}'
    ```

    *Note: Ensure your host's firewall allows incoming connections on port 8000.*

## Architecture Overview

```
src/
├── ingestion/     # Document loading, chunking, storage
├── retrieval/     # Embeddings, vector search, BM25, reranking, task planning
├── policy/        # Policy engine for tool discovery
├── server/        # FastAPI, auth, logging, tool execution
└── sdk/           # Python SDK for client interaction
```

## License

MIT
