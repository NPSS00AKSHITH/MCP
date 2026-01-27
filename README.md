# Adaptive RAG MCP System

Welcome to the **Adaptive RAG MCP System** repository. This project implements a production-grade **Model Context Protocol (MCP)** server for intelligent Retrieval-Augmented Generation (RAG), paired with a test chatbot client.

## 📂 Project Structure

This repository is organized into two main components:

| Component | Directory | Description |
|-----------|-----------|-------------|
| **Server** | [`adaptive_rag_mcp/`](./adaptive_rag_mcp/) | The core MCP server containing the logic for adaptive retrieval, hybrid search (FAISS+BM25), and policy routing. Dockerized and production-ready. |
| **Client** | [`test chatbot/`](./test%20chatbot/) | A Python-based CLI chatbot that acts as an MCP client. It enables testing PDFs, querying the knowledge base, and verifying server responses. |

## 🚀 Quick Start Guide

To get the full system running, you need to start the server first, then the client.

### 1. Prerequisites
*   **Docker** (for running the server)
*   **Python 3.10+** (for running the chatbot)
*   **Gemini API Key** (required for LLM inference)

### 2. Start the Server (`adaptive_rag_mcp`)
Detailed instructions are in [`adaptive_rag_mcp/README.md`](./adaptive_rag_mcp/README.md).

```bash
cd adaptive_rag_mcp

# 1. Create .env with GEMINI_API_KEY=...
# 2. Run with Docker
docker build -t adaptive-rag .
docker run -d -p 8000:8000 --env-file .env --name adaptive-rag-server adaptive-rag
```

### 3. Start the Client (`test chatbot`)
Detailed instructions are in [`test chatbot/README.md`](./test%20chatbot/README.md).

```bash
cd "test chatbot"

# 1. Create .env matching the server config
# 2. Install dependencies
uv sync

# 3. Run the chatbot
python src/chatbot.py
```

## 🏗️ Architecture Overview
![Architecture Overview](./image/architecture.png)
## 📚 Documentation
*   **Server Architecture**: See [Server README](./adaptive_rag_mcp/README.md).
*   **Testing Guide**: See [Chatbot README](./test%20chatbot/README.md).
