"""Reranking tool implementation for MCP server.

Supports both standard reranking and adaptive memory-augmented reranking
that learns from user feedback to improve results over time.
"""

from typing import Any

from src.retrieval.reranker import get_reranker, RERANKERS


def rerank(input_data: dict[str, Any]) -> dict[str, Any]:
    """Rerank documents by relevance to query.
    
    Supports two modes:
    1. Standard mode (default): Uses cross-encoder for reranking
    2. Adaptive mode: Uses memory-augmented reranking that learns from feedback
    
    Input:
        query: The query to rank against
        documents: List of {id, content} dicts to rerank
        top_k: Optional limit on results
        model: Reranker type ("cross-encoder" or "simple")
        threshold: Relevance threshold for quality signals (default 0.3)
        query_type: Optional query classification (for adaptive mode)
        use_adaptive_memory: Enable memory-augmented reranking (default False)
        
    Output:
        results: Reranked list with relevance_score and original_rank
        quality_signals: Evidence quality metrics
        model_used: Reranker model name
    """
    query = input_data.get("query", "")
    documents = input_data.get("documents", [])
    top_k = input_data.get("top_k")
    model_type = input_data.get("model", "cross-encoder")
    threshold = input_data.get("threshold", 0.3)
    query_type = input_data.get("query_type")
    use_adaptive = input_data.get("use_adaptive_memory", False)
    
    if not query:
        return {
            "results": [],
            "quality_signals": {},
            "model_used": "",
            "error": "Query is required",
        }
    
    if not documents:
        return {
            "results": [],
            "quality_signals": {
                "top_score": 0,
                "score_spread": 0,
                "mean_score": 0,
                "relevant_count": 0,
                "total_count": 0,
                "confidence_flags": ["no_documents"],
                "is_high_confidence": False,
            },
            "model_used": model_type,
            "error": None,
        }
    
    # Choose reranker based on mode
    if use_adaptive:
        # Use adaptive memory ranker for learning from feedback
        from src.retrieval.adaptive_memory import get_adaptive_memory
        
        memory = get_adaptive_memory()
        results, quality = memory.rerank_with_memory(
            query=query,
            documents=documents,
            query_type=query_type,
            top_k=top_k,
        )
        model_name = f"adaptive-{model_type}"
    else:
        # Standard reranking (backward compatible)
        reranker = get_reranker(model_type)
        if hasattr(reranker, "relevance_threshold"):
            reranker.relevance_threshold = threshold
        
        results, quality = reranker.rerank(query, documents, top_k=top_k)
        model_name = reranker.model_name
    
    return {
        "results": [
            {
                "id": r.id,
                "content": r.content,
                "relevance_score": r.relevance_score,
                "original_rank": r.original_rank,
                "metadata": r.metadata,
            }
            for r in results
        ],
        "quality_signals": quality.to_dict(),
        "model_used": model_name,
        "adaptive_mode": use_adaptive,
        "metadata": {
            "query_length": len(query),
            "documents_evaluated": len(documents),
        },
    }


# Add to retrieval tools registry
RERANK_TOOLS: dict[str, callable] = {
    "rerank": rerank,
}

