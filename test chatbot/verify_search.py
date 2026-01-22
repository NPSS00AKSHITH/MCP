import httpx
import os
import json
from dotenv import load_dotenv

# Load env from test chatbot dir
load_dotenv(r"D:\MCP\test chatbot\.env")

API_KEY = os.getenv("ADAPTIVE_RAG_API_KEY")
print(f"Using API Key: {API_KEY[:5]}...")

try:
    print("Searching for 'memory'...")
    resp = httpx.post(
        "http://localhost:8000/tools/search",
        headers={"X-API-Key": API_KEY},
        json={"query": "memory techniques", "k": 3}
    )
    
    if resp.status_code == 200:
        data = resp.json()
        results = data.get("result", {}).get("results", [])
        print(f"\n✅ Search Successful! Found {len(results)} results.")
        for i, r in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(f"Content: {r.get('content')[:200]}...")
            print(f"Source: {r.get('metadata', {}).get('file_name')}")
    else:
        print(f"❌ Search Failed: {resp.status_code} - {resp.text}")

except Exception as e:
    print(f"❌ Error: {e}")
