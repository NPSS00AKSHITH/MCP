# Adaptive RAG Test Chatbot

Interactive chatbot that demonstrates the Adaptive RAG MCP Server with PDF loading capabilities.

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Docker (for running the MCP server)
- Gemini API key

### 1. Setup Environment

```bash
# Clone or navigate to this directory
cd "D:\MCP\test chatbot"

# Create/activate virtual environment
uv venv
.venv\Scripts\activate.ps1

# Install dependencies
uv pip install -e .
```

### 2. Configure API Keys

Create/update `.env` file:
```env
ADAPTIVE_RAG_API_KEY=<your-mcp-api-key>
GEMINI_API_KEY=<your-gemini-api-key>
MCP_SERVER_URL=http://localhost:8000
```

Get Gemini API key from: https://aistudio.google.com/apikey

### 3. Start MCP Server

```bash
# Navigate to adaptive_rag_mcp directory
cd D:\MCP\adaptive_rag_mcp

# Start via Docker
docker run -d -p 8000:8000 --env-file .env --name adaptive-rag-server adaptive-rag

# Verify it's running
docker ps | grep adaptive-rag-server
```

### 4. Run Chatbot

```bash
cd "D:\MCP\test chatbot"
.venv\Scripts\activate.ps1
python src/chatbot.py
```

---

## 📚 Usage

### Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/load <path>` | Load PDF into knowledge base | `/load data/document.pdf` |
| `/list` | List all loaded documents | `/list` |
| `/stats` | Show ingestion statistics | `/stats` |
| `/help` | Display help message | `/help` |
| `/quit` | Exit the chatbot | `/quit` |

### Ask Questions

Just type your question and press Enter:
```
You: What are the types of machine learning?
🤖 The three types are supervised, unsupervised, and reinforcement learning...
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│  Chatbot (src/chatbot.py)               │
│  • Gemini 2.5 Flash for Q&A             │
│  • Client-side PDF extraction (pypdf)   │
│  • HTTP client for MCP communication    │
└──────────────┬──────────────────────────┘
               │ HTTP REST API (port 8000)
               ▼
┌─────────────────────────────────────────┐
│  Adaptive RAG MCP Server (Docker)       │
│  • Document ingestion & indexing        │
│  • Hybrid vector search (FAISS)         │
│  • Adaptive retrieval policy            │
│  • Evidence scoring & reranking         │
└─────────────────────────────────────────┘
```

### How It Works

1. **User asks a question** → Gemini decides if it needs external knowledge
2. **If yes:** Chatbot searches MCP server with query  
   - MCP server performs hybrid search (dense + sparse)
   - Retrieves top-k relevant chunks
3. **Retrieved context** is sent to Gemini
4. **Gemini generates answer** based on the context

---

## 🧪 Testing

### Generate Sample PDF

```bash
python scripts/generate_sample_pdf.py
# Creates: data/machine_learning_basics.pdf
```

### Test Workflow

```
You: /load data/machine_learning_basics.pdf
✅ Loaded! Doc ID: machine_learning_basics.pdf, Chunks: 2

You: What are the three types of machine learning?
🤖 The three types are...

You: /stats
📊 Docs: 1, Chunks: 2
```

See [`TEST_RESULTS.md`](docs/TEST_RESULTS.md) for comprehensive test results.

---

## 📂 Project Structure

```
test chatbot/
├── src/
│   └── chatbot.py                # Main chatbot application
├── scripts/
│   ├── generate_sample_pdf.py    # PDF generator for testing
│   ├── verify_search.py          # MCP search verification script
│   └── debug_server.py           # Server debugging utility
├── data/                         # Data files (PDFs, TXT)
│   └── machine_learning_basics.pdf
├── docs/                         # Documentation
│   └── TEST_RESULTS.md           # Test validation report
├── .env                          # Environment variables
└── pyproject.toml                # Project dependencies
```

---

## 🔧 Dependencies

Defined in `pyproject.toml`:
- `google-generativeai` - Gemini LLM integration
- `mcp` - MCP protocol client
- `pypdf` - PDF text extraction
- `httpx` - HTTP client for MCP server
- `python-dotenv` - Environment variable management
- `reportlab` - PDF generation for testing

---

## 🐛 Troubleshooting

### "Cannot connect to MCP server"
```bash
# Check if server is running
docker ps | grep adaptive-rag

# If not, start it
cd D:\MCP\adaptive_rag_mcp
docker run -d -p 8000:8000 --env-file .env --name adaptive-rag-server adaptive-rag

# Check logs
docker logs adaptive-rag-server
```

### "No module named 'pypdf'"
```bash
# Activate venv and install dependencies
.venv\Scripts\activate.ps1
pip install pypdf httpx
```

### "API key was reported as leaked"
```bash
# Get new key from Google AI Studio
# Update .env file with new GEMINI_API_KEY
```

### "404 Not Found" or tool errors
```bash
# Verify MCP server is running and accessible
curl http://localhost:8000/health

# Check available tools
curl -H "X-API-Key: <your-key>" http://localhost:8000/tools
```

---

## 🚀 Next Steps

### Planned Enhancements

1. **Comprehensive Test Suite Integration**
   - Add tests from comprehensive adaptive test suite
   - Validate policy decisions, confidence thresholds
   - Test safety mechanisms (contradictions, refusals)

2. **Upgrade to MCP SDK**
   - Replace HTTP client with proper MCP protocol
   - Use `mcp.ClientSession` for better compliance

3. **Enhanced Observability**
   - Log policy decisions and confidence scores
   - Track iteration counts and strategy switches
   - Add performance metrics

4. **Update Dependencies**
   - Migrate from `google.generativeai` to `google.genai`
   - Update to latest MCP client version

5. **Advanced Features**
   - Multi-document comparison
   - Citation generation
   - Persistent conversation history
   - Batch PDF processing

---

## 📖 Related Documentation

- [MCP Integration Guide](../adaptive_rag_mcp/MCP_INTEGRATION_GUIDE.md)
- [Project Explained](../adaptive_rag_mcp/PROJECT_EXPLAINED.md)
- [Test Results](TEST_RESULTS.md)

---

## 📄 License

Part of the Adaptive RAG MCP Server project.

---

**Status:** ✅ Functional - Successfully tested with PDF loading and RAG Q&A
