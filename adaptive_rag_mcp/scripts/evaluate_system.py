"""Evaluation harness for Adaptive RAG system."""

import sys
import os
import json
import statistics
from dataclasses import asdict
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.server.tools.loop_tools import adaptive_retrieve
from src.server.tools.policy_tools import decide_retrieval
from src.server.logging import setup_logging

# Configure logging to capture output
setup_logging()

DATASET = [
    {"id": "triv_1", "query": "What is 2 + 2?", "type": "trivial"},
    {"id": "triv_2", "query": "Hello there", "type": "trivial"},
    {"id": "spec_1", "query": "Summarize the Q3 financial report", "type": "doc_specific"},
    {"id": "spec_2", "query": "What are the terms of the SLA?", "type": "doc_specific"},
    {"id": "cross_1", "query": "Compare the security policies across all projects", "type": "cross_doc"},
    {"id": "cross_2", "query": "List all authors mentioned in the documentation", "type": "cross_doc"},
    {"id": "gen_1", "query": "Explain the concept of RAG", "type": "general"},
]

def evaluate():
    print("Starting System Evaluation...")
    
    results = []
    
    for case in DATASET:
        print(f"Evaluating: {case['id']} ({case['type']})")
        
        # 1. Policy Decision
        policy = decide_retrieval({"query": case["query"]})
        
        # 2. Retrieval Loop (if not skipped)
        loop_result = None
        if policy["decision"] == "retrieve":
            loop_result = adaptive_retrieve({
                "query": case["query"],
                "max_iterations": 3,
                "confidence_threshold": 0.7
            })
            
        results.append({
            "case": case,
            "policy": policy,
            "loop": loop_result
        })

    # Compute Metrics
    total = len(results)
    skipped = len([r for r in results if r["policy"]["decision"] == "skip"])
    
    retrieved_cases = [r for r in results if r["loop"]]
    avg_context_size = 0
    if retrieved_cases:
        sizes = [len(doc.get("content", "")) for r in retrieved_cases for doc in r["loop"]["results"]]
        avg_context_size = statistics.mean(sizes) if sizes else 0
        
    scores = []
    if retrieved_cases:
        scores = [
            step["top_score"] 
            for r in retrieved_cases 
            for step in r["loop"]["trace"] 
            if step.get("top_score") is not None
        ]
    
    avg_score = statistics.mean(scores) if scores else 0
    
    # Generate Report
    report = {
        "total_queries": total,
        "skipped_rate": skipped / total if total else 0,
        "avg_context_length": round(avg_context_size, 2),
        "avg_confidence_score": round(avg_score, 4),
        "details": results
    }
    
    # Save Report
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if "float" in str(type(obj)):
                return float(obj)
            return super().default(obj)
            
    with open("evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2, cls=NpEncoder)
        
    print("\n--- Evaluation Summary ---")
    print(f"Queries: {total}")
    print(f"Skipped: {skipped} ({report['skipped_rate']:.1%})")
    print(f"Avg Context Size: {report['avg_context_length']} chars")
    print(f"Avg Confidence: {report['avg_confidence_score']}")
    print("Detailed report saved to evaluation_report.json")

if __name__ == "__main__":
    evaluate()
