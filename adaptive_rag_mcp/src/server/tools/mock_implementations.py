"""Mock implementations for all MCP tools - Phase 1 skeleton.

These return static placeholder data to validate the server skeleton.
Real implementations will be added in subsequent phases.
"""

from typing import Any
import random


def mock_decide_retrieval(input_data: dict[str, Any]) -> dict[str, Any]:
    """Mock decide_retrieval - returns static strategy recommendation."""
    query = input_data.get("query", "")
    
    # Simple heuristic for mock: longer queries = more complex
    word_count = len(query.split())
    if word_count <= 5:
        complexity = "simple"
        strategy = "none"
        suggested_k = 0
    elif word_count <= 15:
        complexity = "moderate"
        strategy = "single"
        suggested_k = 5
    else:
        complexity = "complex"
        strategy = "multi_step"
        suggested_k = 10
    
    return {
        "strategy": strategy,
        "complexity": complexity,
        "reasoning": f"[MOCK] Query has {word_count} words. Complexity assessed as {complexity}.",
        "suggested_k": suggested_k,
        "query_rewrites": [query] if strategy == "multi_step" else [],
        "metadata": {"mock": True, "phase": 1}
    }


def mock_embed_query(input_data: dict[str, Any]) -> dict[str, Any]:
    """Mock embed_query - returns fake embedding vector."""
    text = input_data.get("text", "")
    model = input_data.get("model", "all-MiniLM-L6-v2")
    
    # Generate deterministic fake embedding based on text length
    dimensions = 384  # Standard for MiniLM
    # Use simple hash for reproducibility
    seed = sum(ord(c) for c in text)
    random.seed(seed)
    embedding = [random.uniform(-1, 1) for _ in range(dimensions)]
    
    return {
        "embedding": embedding,
        "dimensions": dimensions,
        "model_used": model,
        "metadata": {"mock": True, "text_length": len(text)}
    }


def mock_search(input_data: dict[str, Any]) -> dict[str, Any]:
    """Mock search - returns fake search results."""
    query = input_data.get("query", "")
    k = input_data.get("k", 5)
    collection_id = input_data.get("collection_id", "default")
    
    # Generate mock results
    results = []
    for i in range(min(k, 3)):  # Return at most 3 mock results
        results.append({
            "document_id": f"doc_{i+1}",
            "chunk_id": f"doc_{i+1}_chunk_1",
            "content": f"[MOCK] This is placeholder content for result {i+1} matching query: '{query[:50]}...'",
            "score": round(0.95 - (i * 0.1), 3),
            "metadata": {"source": f"mock_file_{i+1}.txt", "page": i+1}
        })
    
    return {
        "results": results,
        "total_searched": 100,
        "metadata": {"mock": True, "collection_id": collection_id, "query_length": len(query)}
    }


def mock_rerank(input_data: dict[str, Any]) -> dict[str, Any]:
    """Mock rerank - shuffles and re-scores documents."""
    query = input_data.get("query", "")
    documents = input_data.get("documents", [])
    top_k = input_data.get("top_k")
    model = input_data.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    threshold = input_data.get("threshold", 0.0)
    
    # Mock reranking: reverse order and assign new scores
    results = []
    for i, doc in enumerate(reversed(documents)):
        relevance_score = round(0.9 - (i * 0.15), 3)
        if relevance_score >= threshold:
            results.append({
                "id": doc["id"],
                "content": doc["content"],
                "relevance_score": relevance_score,
                "original_rank": len(documents) - i
            })
    
    if top_k:
        results = results[:top_k]
    
    return {
        "results": results,
        "model_used": model,
        "metadata": {"mock": True, "original_count": len(documents)}
    }


def mock_summarize(input_data: dict[str, Any]) -> dict[str, Any]:
    """Mock summarize - returns placeholder summary."""
    context = input_data.get("context", [])
    query = input_data.get("query")
    style = input_data.get("style", "concise")
    
    source_ids = [c.get("id", f"source_{i}") for i, c in enumerate(context)]
    
    summary_prefix = {
        "concise": "[MOCK CONCISE]",
        "detailed": "[MOCK DETAILED]",
        "bullet_points": "[MOCK BULLETS]\n•",
        "executive": "[MOCK EXECUTIVE SUMMARY]"
    }
    
    return {
        "summary": f"{summary_prefix.get(style, '[MOCK]')} Summary of {len(context)} context passages. Query focus: {query or 'none'}",
        "source_ids": source_ids,
        "metadata": {"mock": True, "style": style, "context_count": len(context)}
    }


def mock_cite(input_data: dict[str, Any]) -> dict[str, Any]:
    """Mock cite - returns response with fake citations."""
    query = input_data.get("query", "")
    sources = input_data.get("sources", [])
    citation_style = input_data.get("citation_style", "inline_number")
    
    citations = []
    for i, source in enumerate(sources[:3]):  # Max 3 citations for mock
        marker = f"[{i+1}]" if citation_style == "inline_number" else f"[{source.get('title', f'Source{i+1}')}]"
        citations.append({
            "marker": marker,
            "source_id": source["id"],
            "excerpt": source["content"][:100] + "...",
            "title": source.get("title"),
            "url": source.get("url")
        })
    
    citation_markers = " ".join(c["marker"] for c in citations)
    
    return {
        "response": f"[MOCK] Based on the provided sources {citation_markers}, this is a placeholder response to: '{query}'",
        "citations": citations,
        "uncited_claims": [],
        "metadata": {"mock": True, "citation_style": citation_style}
    }


def mock_compare_documents(input_data: dict[str, Any]) -> dict[str, Any]:
    """Mock compare_documents - returns placeholder comparison."""
    documents = input_data.get("documents", [])
    focus = input_data.get("focus")
    comparison_type = input_data.get("comparison_type", "full")
    
    doc_ids = [d["id"] for d in documents]
    
    return {
        "similarities": [
            {
                "description": "[MOCK] All documents discuss related topics",
                "document_ids": doc_ids
            }
        ],
        "differences": [
            {
                "aspect": "perspective",
                "positions": [
                    {"document_id": doc_ids[0], "stance": "[MOCK] First perspective"},
                    {"document_id": doc_ids[1] if len(doc_ids) > 1 else doc_ids[0], "stance": "[MOCK] Second perspective"}
                ]
            }
        ],
        "conflicts": [],
        "summary": f"[MOCK] Comparison of {len(documents)} documents using {comparison_type} analysis. Focus: {focus or 'general'}",
        "metadata": {"mock": True, "comparison_type": comparison_type}
    }


def mock_generate_response(input_data: dict[str, Any]) -> dict[str, Any]:
    """Mock generate_response - returns placeholder response."""
    query = input_data.get("query", "")
    context = input_data.get("context", [])
    temperature = input_data.get("temperature", 0.7)
    
    context_used = len(context) > 0
    
    return {
        "response": f"[MOCK] This is a placeholder response to '{query}'. Context items used: {len(context)}. Temperature: {temperature}",
        "context_used": context_used,
        "confidence": 0.85 if context_used else 0.5,
        "metadata": {"mock": True, "context_count": len(context)}
    }


# Registry mapping tool names to their mock implementations
MOCK_TOOL_REGISTRY: dict[str, callable] = {
    "decide_retrieval": mock_decide_retrieval,
    "embed_query": mock_embed_query,
    "search": mock_search,
    "rerank": mock_rerank,
    "summarize": mock_summarize,
    "cite": mock_cite,
    "compare_documents": mock_compare_documents,
    "generate_response": mock_generate_response,
}


def execute_mock_tool(tool_name: str, input_data: dict[str, Any]) -> dict[str, Any]:
    """Execute a mock tool by name."""
    tool_fn = MOCK_TOOL_REGISTRY.get(tool_name)
    if tool_fn is None:
        raise ValueError(f"Unknown tool: {tool_name}")
    return tool_fn(input_data)
