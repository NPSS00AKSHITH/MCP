"""Interactive RAG Chatbot using Adaptive RAG MCP Server.

Supports loading PDFs via MCP for RAG-based Q&A.

Run the server first (in another terminal):
    cd D:/MCP/adaptive_rag_mcp
    .venv/Scripts/activate
    python -m src.server.main

Then run this chatbot:
    python chatbot.py
"""

import os
import httpx
import google.generativeai as genai
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv(override=True)

ADAPTIVE_RAG_API_KEY = os.getenv("ADAPTIVE_RAG_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")

if not all([ADAPTIVE_RAG_API_KEY, GEMINI_API_KEY]):
    print("Error: Missing environment variables. Please check .env file.")
    sys.exit(1)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')


class MCPClient:
    """HTTP client for Adaptive RAG MCP Server."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json",
        }
        self.http_client = httpx.Client(timeout=60.0)
    
    def health_check(self) -> bool:
        try:
            resp = self.http_client.get(f"{self.base_url}/health")
            return resp.status_code == 200
        except Exception:
            return False
    
    def list_tools(self) -> list:
        resp = self.http_client.get(f"{self.base_url}/tools", headers=self.headers)
        resp.raise_for_status()
        return resp.json()
    
    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        resp = self.http_client.post(
            f"{self.base_url}/tools/{tool_name}",
            headers=self.headers,
            json=arguments
        )
        resp.raise_for_status()
        return resp.json()
    
    def load_pdf(self, file_path: str) -> dict:
        """Load a PDF file and send its text to the knowledge base."""
        import pypdf
        
        try:
            # Client-side extraction
            print(f"   (Reading PDF locally...)")
            reader = pypdf.PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n\n"
            
            doc_id = os.path.basename(file_path)
            
            # Send text to server
            print(f"   (Sending {len(text)} chars to server...)")
            ingest_result = self.call_tool("ingest_document", {
                "text": text,
                "doc_id": doc_id,
                "file_name": doc_id,
                "metadata": {"source": "client_upload"}
            })
            
            if not ingest_result.get("success") or not ingest_result.get("result", {}).get("success"):
                return ingest_result
            
            # Index for search
            print(f"   (Indexing...)")
            if doc_id:
                index_result = self.call_tool("index_document", {"doc_id": doc_id})
                return {"ingest": ingest_result, "index": index_result}
            
            return ingest_result
            
        except ImportError:
            return {"result": {"success": False, "error": "pypdf not installed in client environment"}}
        except Exception as e:
            return {"result": {"success": False, "error": str(e)}}


def print_help():
    print("\n" + "=" * 60)
    print("📚 Commands:")
    print("  /load <path>    - Load a PDF into knowledge base")
    print("  /list           - List loaded documents")
    print("  /stats          - Show stats")
    print("  /quit           - Exit")
    print("\nOr type your question to search!")
    print("=" * 60 + "\n")


def main():
    print("\n" + "=" * 60)
    print("🤖 Adaptive RAG Chatbot with PDF Support")
    print("=" * 60)
    
    mcp = MCPClient(MCP_SERVER_URL, ADAPTIVE_RAG_API_KEY)
    
    print(f"\nConnecting to MCP server...")
    
    if not mcp.health_check():
        print("\n❌ Cannot connect to MCP server!")
        print("   Start it with: python -m src.server.main")
        return
    
    tools = mcp.list_tools()
    print(f"✅ Connected! Tools: {[t['name'] for t in tools]}")
    print_help()
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            
            if user_input.lower() in ['/quit', 'quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == '/help':
                print_help()
                continue
            
            if user_input.lower().startswith('/load '):
                file_path = user_input[6:].strip()
                file_path = os.path.abspath(file_path)
                
                if not os.path.exists(file_path):
                    print(f"❌ File not found: {file_path}")
                    continue
                
                print(f"📥 Loading {file_path}...")
                try:
                    result = mcp.load_pdf(file_path)
                    ingest = result.get("ingest", result).get("result", {})
                    
                    if ingest.get("success"):
                        print(f"✅ Loaded! Doc ID: {ingest.get('doc_id')}, Chunks: {ingest.get('total_chunks')}")
                        idx = result.get("index", {}).get("result", {})
                        if idx.get("success"):
                            print(f"   Indexed: {idx.get('chunks_indexed')} chunks")
                    else:
                        print(f"❌ Failed: {ingest.get('error')}")
                except Exception as e:
                    print(f"❌ Error: {e}")
                continue
            
            if user_input.lower() == '/list':
                result = mcp.call_tool("list_documents", {})
                docs = result.get("result", {}).get("documents", [])
                print(f"\n📚 Documents ({len(docs)}):")
                for d in docs:
                    print(f"   - {d.get('doc_id')}: {d.get('file_name')}")
                continue
            
            if user_input.lower() == '/stats':
                result = mcp.call_tool("get_ingestion_stats", {})
                s = result.get("result", {})
                print(f"\n📊 Docs: {s.get('document_count')}, Chunks: {s.get('chunk_count')}")
                continue
            
            # RAG search
            prompt = f"""You are a helpful assistant. The user asked: "{user_input}"
If this needs external knowledge, reply ONLY with "SEARCH: <query>".
Otherwise, answer directly."""
            
            response = model.generate_content(prompt)
            reply = response.text.strip()
            
            if reply.startswith("SEARCH: "):
                query = reply.replace("SEARCH: ", "").strip()
                print(f"(🔍 Searching: {query})")
                
                result = mcp.call_tool("search", {"query": query, "k": 5})
                results = result.get("result", {}).get("results", [])
                
                if results:
                    context = "\n".join([f"- {r.get('content', '')[:500]}" for r in results])
                    final = f"Answer based on this context:\n{context}\n\nQuestion: {user_input}"
                    answer = model.generate_content(final)
                    print(f"\n🤖 {answer.text}\n")
                else:
                    print("\n🤖 No results. Try /load <pdf> first.\n")
            else:
                print(f"\n🤖 {reply}\n")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
