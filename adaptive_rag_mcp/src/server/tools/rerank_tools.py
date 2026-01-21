"""Reranking tool implementation for MCP server."""

from typing import Any

from src.retrieval.reranker import get_reranker, RERANKERS


def rerank(input_data: dict[str, Any]) -> dict[str, Any]:
    """Rerank documents by relevance to query.
    
    Input:
        query: The query to rank against
        documents: List of {id, content} dicts to rerank
        top_k: Optional limit on results
        model: Reranker type ("cross-encoder" or "simple")
        threshold: Relevance threshold for quality signals (default 0.3)
        
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
    
    # Get or create reranker
    reranker = get_reranker(model_type)
    if hasattr(reranker, "relevance_threshold"):
        reranker.relevance_threshold = threshold
    
    # Rerank
    results, quality = reranker.rerank(query, documents, top_k=top_k)
    
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
        "model_used": reranker.model_name,
        "metadata": {
            "query_length": len(query),
            "documents_evaluated": len(documents),
        },
    }


# Add to retrieval tools registry
RERANK_TOOLS: dict[str, callable] = {
    "rerank": rerank,
}
