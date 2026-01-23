"""Test script for reranking with quality signals.

Run from project root:
    python examples/test_rerank.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.reranker import CrossEncoderReranker, SimpleReranker


def main():
    print("=" * 70)
    print("Reranking with Quality Signals Test")
    print("=" * 70)
    
    # Test documents
    documents = [
        {"id": "doc1", "content": "Python is a programming language known for its readability."},
        {"id": "doc2", "content": "Machine learning uses neural networks to learn patterns."},
        {"id": "doc3", "content": "RAG combines retrieval with language model generation."},
        {"id": "doc4", "content": "The weather today is sunny with clear skies."},
        {"id": "doc5", "content": "FAISS enables fast vector similarity search."},
    ]
    
    # Test queries
    queries = [
        {
            "query": "What is RAG?",
            "description": "Highly relevant doc exists",
        },
        {
            "query": "Tell me about cooking recipes",
            "description": "No relevant docs (should flag low confidence)",
        },
        {
            "query": "neural networks and machine learning",
            "description": "One clear match",
        },
    ]
    
    # Use simple reranker first (no model loading)
    print("\n📋 Testing SimpleReranker (no ML model)")
    simple = SimpleReranker()
    
    for test in queries[:1]:  # Just first query for simple
        query = test["query"]
        print(f"\n   Query: \"{query}\"")
        results, quality = simple.rerank(query, documents, top_k=3)
        print(f"   Top result: {results[0].id} (score: {results[0].relevance_score})")
        print(f"   Quality: spread={quality.score_spread:.3f}, flags={quality.confidence_flags}")
    
    # Now test with CrossEncoder (loads ML model)
    print("\n" + "=" * 70)
    print("📊 Testing CrossEncoderReranker (loads ML model)")
    print("=" * 70)
    
    reranker = CrossEncoderReranker()
    
    for i, test in enumerate(queries, 1):
        query = test["query"]
        print(f"\n{'─'*50}")
        print(f"Query {i}: \"{query}\"")
        print(f"Type: {test['description']}")
        print("─" * 50)
        
        results, quality = reranker.rerank(query, documents, top_k=3)
        
        print("\nReranked Results:")
        for rank, r in enumerate(results, 1):
            print(f"  {rank}. [{r.id}] score={r.relevance_score:.4f} (was #{r.original_rank})")
            print(f"     Content: {r.content[:50]}...")
        
        print("\nQuality Signals:")
        print(f"  Top score:      {quality.top_score:.4f}")
        print(f"  Score spread:   {quality.score_spread:.4f}")
        print(f"  Mean score:     {quality.mean_score:.4f}")
        print(f"  Relevant count: {quality.relevant_count}/{quality.total_count}")
        print(f"  Confidence:     {'HIGH ✓' if quality.is_high_confidence else 'LOW ⚠'}")
        if quality.confidence_flags:
            print(f"  Flags:          {', '.join(quality.confidence_flags)}")
    
    print("\n" + "=" * 70)
    print("Quality Signals Explanation")
    print("=" * 70)
    print("""
    RELEVANCE SCORE: Cross-encoder score normalized to 0-1
    - Measures how well document answers the query
    
    SCORE SPREAD: Standard deviation of scores
    - High spread = clear winner(s)
    - Low spread = similar relevance (or all irrelevant)
    
    CONFIDENCE FLAGS:
    - low_top_score:    Best result has score < 0.3
    - flat_distribution: Score spread < 0.1 (no clear winner)
    - few_relevant:      Less than 30% above threshold
    - score_drop:        Large gap between #1 and #2 (outlier)
    """)
    
    print("=" * 70)
    print("✅ Reranking tests complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
