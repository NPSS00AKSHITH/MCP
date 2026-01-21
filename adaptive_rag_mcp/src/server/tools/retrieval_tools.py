"""Real implementations for retrieval tools.

Supports dense, sparse, and hybrid search modes.
"""

from typing import Any

from src.retrieval.pipeline import get_retrieval_pipeline
from src.retrieval.embedder import get_embedder
from src.retrieval.hybrid import get_hybrid_retriever, SearchMode
from src.retrieval.sparse_retriever import get_sparse_retriever
from src.ingestion.pipeline import get_pipeline as get_ingestion_pipeline


def search(input_data: dict[str, Any]) -> dict[str, Any]:
    """Perform similarity search with mode selection.
    
    Input:
        query: Search query string
        mode: "dense" | "sparse" | "hybrid" (default: hybrid)
        k: Number of results (default 5)
        
    Output:
        results: List of search results with scores
        total_searched: Total items in index
        mode: Search mode used
    """
    query = input_data.get("query", "")
    mode_str = input_data.get("mode", "hybrid")
    k = input_data.get("k", 5)
    
    if not query:
        return {
            "results": [],
            "total_searched": 0,
            "mode": mode_str,
            "error": "Query is required",
        }
    
    # Parse mode
    try:
        mode = SearchMode(mode_str)
    except ValueError:
        mode = SearchMode.HYBRID
        mode_str = "hybrid"
    
    # Use hybrid retriever for all modes
    hybrid = get_hybrid_retriever()
    results = hybrid.search(query, k=k, mode=mode)
    
    # Get total count based on mode
    if mode == SearchMode.SPARSE:
        total = hybrid.sparse.count()
    elif mode == SearchMode.DENSE:
        total = hybrid.dense.vector_store.count()
    else:
        total = max(hybrid.sparse.count(), hybrid.dense.vector_store.count())
    
    return {
        "results": [
            {
                "document_id": r.doc_id,
                "chunk_id": r.chunk_id,
                "content": r.content,
                "score": round(r.score, 4),
                "metadata": r.metadata,
                "dense_score": round(r.dense_score, 4) if r.dense_score else None,
                "sparse_score": round(r.sparse_score, 4) if r.sparse_score else None,
                "dense_rank": r.dense_rank,
                "sparse_rank": r.sparse_rank,
            }
            for r in results
        ],
        "total_searched": total,
        "mode": mode_str,
        "metadata": {"k": k, "query_length": len(query)},
    }


def embed_query(input_data: dict[str, Any]) -> dict[str, Any]:
    """Generate embedding for text.
    
    Input:
        text: Text to embed
        model: Model name (optional)
        
    Output:
        embedding: Vector as list of floats
        dimensions: Embedding dimensions
        model_used: Model name
    """
    text = input_data.get("text", "")
    model = input_data.get("model", "all-MiniLM-L6-v2")
    
    if not text:
        return {
            "embedding": [],
            "dimensions": 0,
            "model_used": model,
            "error": "Text is required",
        }
    
    embedder = get_embedder(model)
    embedding = embedder.embed_text(text)
    
    return {
        "embedding": embedding.tolist(),
        "dimensions": len(embedding),
        "model_used": embedder.model_name,
        "metadata": {"text_length": len(text)},
    }


def index_document(input_data: dict[str, Any]) -> dict[str, Any]:
    """Index an ingested document for search (both dense and sparse).
    
    Input:
        doc_id: Document ID to index
        
    Output:
        success: Whether indexing succeeded
        chunks_indexed: Number of chunks indexed
    """
    doc_id = input_data.get("doc_id", "")
    
    if not doc_id:
        return {
            "success": False,
            "chunks_indexed": 0,
            "error": "doc_id is required",
        }
    
    # Get chunks from ingestion pipeline
    ingestion = get_ingestion_pipeline()
    chunks = ingestion.get_chunks(doc_id)
    
    if not chunks:
        return {
            "success": False,
            "doc_id": doc_id,
            "chunks_indexed": 0,
            "error": f"No chunks found for {doc_id}",
        }
    
    # Index in hybrid retriever (both dense and sparse)
    hybrid = get_hybrid_retriever()
    count = hybrid.index_chunks(chunks)
    
    return {
        "success": count > 0,
        "doc_id": doc_id,
        "chunks_indexed": count,
        "error": None,
    }


def get_retrieval_stats(input_data: dict[str, Any]) -> dict[str, Any]:
    """Get retrieval index statistics.
    
    Output:
        dense: Dense index stats
        sparse: Sparse index stats
        rrf_k: RRF constant
    """
    hybrid = get_hybrid_retriever()
    return hybrid.get_stats()


# Registry for retrieval tools
RETRIEVAL_TOOLS: dict[str, callable] = {
    "search": search,
    "embed_query": embed_query,
    "index_document": index_document,
    "get_retrieval_stats": get_retrieval_stats,
}
