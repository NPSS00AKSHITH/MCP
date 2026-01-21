"""Verification script for Adaptive Retrieval Policy Engine."""

import sys
import os
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.server.tools.policy_tools import decide_retrieval

def test_query(query: str, expected_type: str = None):
    print(f"\nQuery: '{query}'")
    
    result = decide_retrieval({"query": query})
    
    print(json.dumps(result, indent=2))
    
    if expected_type:
        assert result["query_type"] == expected_type, f"Expected {expected_type}, got {result['query_type']}"
        print("✅ Classification correct")

def main():
    print("Testing Policy Engine...")
    
    # Test cases based on requirements
    test_query("What is the capital of France?", "general_knowledge")
    test_query("Summarize the Q3 report", "doc_specific") 
    test_query("Compare the results required vs actual", "comparison")
    test_query("Find all documents about security across the repository", "multi_doc")
    test_query("What does the contract say about termination?", "doc_specific")
    
    print("\nAll tests completed.")

if __name__ == "__main__":
    main()
