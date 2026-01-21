"""Verification script for Adaptive Retrieval Loop."""

import sys
import os
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.server.tools.loop_tools import adaptive_retrieve

def test_loop(query: str, case_name: str):
    print(f"\n--- Testing Case: {case_name} ---")
    print(f"Query: '{query}'")
    
    # Run loop
    result = adaptive_retrieve({
        "query": query,
        "max_iterations": 3,
        "confidence_threshold": 0.7  # High threshold to force iterations if scores are mediocre
    })
    
    # Print Trace
    print("\nTrace:")
    for step in result["trace"]:
        print(f"  Step {step['step']}: Strategy={step['strategy']}, Retrieved={step['retrieved']}, TopScore={step['top_score']}, Confident={step['confident']}")
        
    print(f"\nFinal Status: Success={result['final_status']['success']}, Iterations={result['final_status']['iterations']}, Reason={result['final_status']['reason']}")

def main():
    print("Starting Loop Verification...")
    
    # Case 1: Easy query (Should stop at Step 1)
    # "Artificial Intelligence" usually yields high scores if the corpus is relevant
    test_loop("What is Artificial Intelligence?", "Easy Query")
    
    # Case 2: Hard query (Should retry)
    # Something specific that might need hybrid or dense fallback
    # Assuming corpus has limited info, this might trigge retries
    test_loop("Find specific details about the proprietary encryption protocol V2", "Hard/Specific Query")
    
    print("\nVerification Complete.")

if __name__ == "__main__":
    main()
