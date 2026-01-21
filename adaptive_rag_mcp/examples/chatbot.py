"""Interactive RAG Chatbot using Adaptive RAG MCP Server.

Run the server first:
    python -m src.server.main

Then run this chatbot:
    python examples/chatbot.py
"""

import httpx
import json
from typing import Optional

BASE_URL = "http://localhost:8000"
API_KEY = "dev-secret-key-change-in-production"
HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json",
}


class RAGChatbot:
    """Interactive chatbot using Adaptive RAG MCP Server."""
    
    def __init__(self, base_url: str = BASE_URL, api_key: str = API_KEY):
        self.base_url = base_url
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json",
        }
        self.client = httpx.Client(timeout=30.0)
    
    def check_health(self) -> bool:
        """Check if server is running."""
        try:
            response = self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except httpx.ConnectError:
            return False
    
    def decide_retrieval(self, query: str) -> dict:
        """Decide retrieval strategy for query."""
        response = self.client.post(
            f"{self.base_url}/tools/decide_retrieval",
            headers=self.headers,
            json={"query": query}
        )
        response.raise_for_status()
        return response.json()
    
    def search(self, query: str, mode: str = "hybrid", k: int = 5) -> dict:
        """Search for relevant documents."""
        response = self.client.post(
            f"{self.base_url}/tools/search",
            headers=self.headers,
            json={"query": query, "mode": mode, "k": k}
        )
        response.raise_for_status()
        return response.json()
    
    def ingest_document(self, text: str, doc_id: Optional[str] = None) -> dict:
        """Ingest a document into the system."""
        payload = {"text": text}
        if doc_id:
            payload["doc_id"] = doc_id
        
        response = self.client.post(
            f"{self.base_url}/tools/ingest_document",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def ask(self, question: str) -> str:
        """Process a question using the RAG pipeline."""
        # Step 1: Decide retrieval strategy
        decision = self.decide_retrieval(question)
        strategy = decision.get("result", {}).get("strategy", "hybrid")
        
        # Step 2: Search for relevant documents
        search_result = self.search(question, mode=strategy, k=5)
        results = search_result.get("result", {}).get("results", [])
        
        if not results:
            return "I couldn't find any relevant information to answer your question."
        
        # Step 3: Format the response with retrieved context
        response_parts = [f"📚 Found {len(results)} relevant documents:\n"]
        
        for i, doc in enumerate(results, 1):
            content = doc.get("content", doc.get("text", "No content"))
            score = doc.get("score", 0)
            doc_id = doc.get("doc_id", doc.get("id", "unknown"))
            response_parts.append(f"\n--- Result {i} (Score: {score:.3f}) ---")
            response_parts.append(f"Source: {doc_id}")
            response_parts.append(f"Content: {content[:500]}...")
        
        return "\n".join(response_parts)


def print_banner():
    """Print welcome banner."""
    print("\n" + "=" * 60)
    print("🤖 Adaptive RAG Chatbot")
    print("=" * 60)
    print("\nCommands:")
    print("  /ingest <text>  - Ingest a document")
    print("  /help           - Show this help")
    print("  /quit           - Exit the chatbot")
    print("\nOr just type your question to search the knowledge base!")
    print("=" * 60 + "\n")


def main():
    """Run the interactive chatbot."""
    print_banner()
    
    chatbot = RAGChatbot()
    
    # Check server health
    print("🔌 Connecting to MCP server...")
    if not chatbot.check_health():
        print("\n❌ Error: Cannot connect to the MCP server!")
        print("   Please start the server first:")
        print("   python -m src.server.main")
        return
    
    print("✅ Connected to MCP server!\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() == "/quit":
                print("\n👋 Goodbye!")
                break
            
            elif user_input.lower() == "/help":
                print_banner()
                continue
            
            elif user_input.lower().startswith("/ingest "):
                text = user_input[8:].strip()
                if text:
                    print("\n📥 Ingesting document...")
                    result = chatbot.ingest_document(text)
                    print(f"✅ Document ingested: {result}")
                else:
                    print("⚠️  Please provide text to ingest.")
                continue
            
            # Regular question
            print("\n🔍 Searching knowledge base...")
            response = chatbot.ask(user_input)
            print(f"\n🤖 Bot:\n{response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except httpx.HTTPStatusError as e:
            print(f"\n❌ API Error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
