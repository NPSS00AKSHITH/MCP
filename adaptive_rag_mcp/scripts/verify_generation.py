"""Verification script for Generation Tools."""

import sys
import os
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.server.tools.generation_tools import summarize, compare_documents, cite
from src.config import get_settings

def verify_generation():
    settings = get_settings()
    if not settings.google_api_key:
        print("⚠️  No Google API Key found. Skipping live LLM tests.")
        return

    print("--- Testing Summarize ---")
    summary = summarize({
        "context": [
            {"id": "c1", "content": "The adaptive RAG system uses a policy engine to decide retrieval logic."},
            {"id": "c2", "content": "It can switch between dense and sparse search modes dynamically."}
        ],
        "query": "What are the key components?",
        "style": "concise"
    })
    print(json.dumps(summary, indent=2))

    print("\n--- Testing Compare ---")
    comparison = compare_documents({
        "documents": [
            {"id": "docA", "content": "Python is a dynamic language.", "label": "Python"},
            {"id": "docB", "content": "Rust is a static language with memory safety.", "label": "Rust"}
        ],
        "focus": "Type system"
    })
    print(comparison["summary"])

    print("\n--- Testing Cite ---")
    citation = cite({
        "query": "Which language is memory safe?",
        "sources": [
             {"id": "docA", "content": "Python is a dynamic language."},
             {"id": "docB", "content": "Rust is a static language with memory safety."}
        ]
    })
    print(citation["response"])

if __name__ == "__main__":
    verify_generation()
