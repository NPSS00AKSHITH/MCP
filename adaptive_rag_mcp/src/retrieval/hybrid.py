"""Hybrid retrieval with Reciprocal Rank Fusion (RRF).

Combines dense (vector) and sparse (BM25) retrieval results.
"""

from dataclasses import dataclass, field
from typing import List, Literal
from enum import Enum

from src.retrieval.vector_store import SearchResult as DenseResult
from src.retrieval.sparse_retriever import SparseSearchResult
from src.server.logging import get_logger

logger = get_logger(__name__)


class SearchMode(str, Enum):
    """Search mode options."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


@dataclass
class HybridSearchResult:
    """Result from hybrid search with combined score."""
    
    chunk_id: str
    doc_id: str
    content: str
    score: float  # Combined RRF score
    metadata: dict = field(default_factory=dict)
    
    # Individual scores for transparency
    dense_score: float | None = None
    sparse_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None


def reciprocal_rank_fusion(
    dense_results: List[DenseResult],
    sparse_results: List[SparseSearchResult],
    k: int = 60,
) -> List[HybridSearchResult]:
    """Combine results using Reciprocal Rank Fusion.
    
    RRF formula: score = sum(1 / (k + rank))
    where k is a constant (typically 60) and rank is 1-indexed position.
    
    Args:
        dense_results: Results from dense (vector) search.
        sparse_results: Results from sparse (BM25) search.
        k: RRF constant (default 60, as per original paper).
        
    Returns:
        Combined list of HybridSearchResult sorted by RRF score.
    """
    # Track scores and metadata by chunk_id
    chunk_data: dict[str, dict] = {}
    
    # Process dense results
    for rank, result in enumerate(dense_results, start=1):
        rrf_score = 1.0 / (k + rank)
        
        if result.chunk_id not in chunk_data:
            chunk_data[result.chunk_id] = {
                "chunk_id": result.chunk_id,
                "doc_id": result.doc_id,
                "content": result.content,
                "metadata": result.metadata,
                "rrf_score": 0.0,
                "dense_score": result.score,
                "sparse_score": None,
                "dense_rank": rank,
                "sparse_rank": None,
            }
        
        chunk_data[result.chunk_id]["rrf_score"] += rrf_score
        chunk_data[result.chunk_id]["dense_score"] = result.score
        chunk_data[result.chunk_id]["dense_rank"] = rank
    
    # Process sparse results
    for rank, result in enumerate(sparse_results, start=1):
        rrf_score = 1.0 / (k + rank)
        
        if result.chunk_id not in chunk_data:
            chunk_data[result.chunk_id] = {
                "chunk_id": result.chunk_id,
                "doc_id": result.doc_id,
                "content": result.content,
                "metadata": result.metadata,
                "rrf_score": 0.0,
                "dense_score": None,
                "sparse_score": result.score,
                "dense_rank": None,
                "sparse_rank": rank,
            }
        
        chunk_data[result.chunk_id]["rrf_score"] += rrf_score
        chunk_data[result.chunk_id]["sparse_score"] = result.score
        chunk_data[result.chunk_id]["sparse_rank"] = rank
    
    # Sort by RRF score
    sorted_chunks = sorted(
        chunk_data.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )
    
    # Convert to HybridSearchResult
    return [
        HybridSearchResult(
            chunk_id=c["chunk_id"],
            doc_id=c["doc_id"],
            content=c["content"],
            score=round(c["rrf_score"], 6),
            metadata=c["metadata"],
            dense_score=c["dense_score"],
            sparse_score=c["sparse_score"],
            dense_rank=c["dense_rank"],
            sparse_rank=c["sparse_rank"],
        )
        for c in sorted_chunks
    ]


class HybridRetriever:
    """Hybrid retriever combining dense and sparse search."""
    
    def __init__(
        self,
        dense_retriever,  # RetrievalPipeline
        sparse_retriever,  # BM25Retriever
        rrf_k: int = 60,
    ):
        """Initialize hybrid retriever.
        
        Args:
            dense_retriever: Dense (vector) retrieval pipeline.
            sparse_retriever: Sparse (BM25) retriever.
            rrf_k: RRF constant for score fusion.
        """
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.rrf_k = rrf_k
    
    def search(
        self,
        query: str,
        k: int = 5,
        mode: SearchMode = SearchMode.HYBRID,
    ) -> List[HybridSearchResult]:
        """Search with specified mode.
        
        Args:
            query: Search query.
            k: Number of results to return.
            mode: Search mode (dense, sparse, or hybrid).
            
        Returns:
            List of HybridSearchResult objects.
        """
        logger.info("hybrid_search", query_length=len(query), k=k, mode=mode.value)
        
        if mode == SearchMode.DENSE:
            # Dense only
            dense_results = self.dense.search(query, k=k)
            return [
                HybridSearchResult(
                    chunk_id=r.chunk_id,
                    doc_id=r.doc_id,
                    content=r.content,
                    score=r.score,
                    metadata=r.metadata,
                    dense_score=r.score,
                    sparse_score=None,
                    dense_rank=i + 1,
                    sparse_rank=None,
                )
                for i, r in enumerate(dense_results)
            ]
        
        elif mode == SearchMode.SPARSE:
            # Sparse only
            sparse_results = self.sparse.search(query, k=k)
            return [
                HybridSearchResult(
                    chunk_id=r.chunk_id,
                    doc_id=r.doc_id,
                    content=r.content,
                    score=r.score,
                    metadata=r.metadata,
                    dense_score=None,
                    sparse_score=r.score,
                    dense_rank=None,
                    sparse_rank=i + 1,
                )
                for i, r in enumerate(sparse_results)
            ]
        
        else:  # HYBRID
            # Get more results from each to allow for better fusion
            fetch_k = k * 2
            
            dense_results = self.dense.search(query, k=fetch_k)
            sparse_results = self.sparse.search(query, k=fetch_k)
            
            # Fuse with RRF
            fused = reciprocal_rank_fusion(
                dense_results, 
                sparse_results, 
                k=self.rrf_k
            )
            
            return fused[:k]
    
    def index_chunks(self, chunks, save: bool = True) -> int:
        """Index chunks in both dense and sparse indexes."""
        # Index in dense
        dense_count = self.dense.index_chunks(chunks, save=save)
        
        # Index in sparse
        sparse_count = self.sparse.add_batch(
            chunk_ids=[c.chunk_id for c in chunks],
            doc_ids=[c.doc_id for c in chunks],
            contents=[c.content for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        
        if save:
            self.sparse.save()
        
        logger.info(
            "hybrid_indexed",
            dense_count=dense_count,
            sparse_count=sparse_count,
        )
        
        return dense_count
    
    def get_stats(self) -> dict:
        """Get combined statistics."""
        return {
            "dense": self.dense.get_stats(),
            "sparse": self.sparse.get_stats(),
            "rrf_k": self.rrf_k,
        }


# Global hybrid retriever instance
_hybrid_retriever: HybridRetriever | None = None


def get_hybrid_retriever() -> HybridRetriever:
    """Get the global hybrid retriever instance."""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        from src.retrieval.pipeline import get_retrieval_pipeline
        from src.retrieval.sparse_retriever import get_sparse_retriever
        
        _hybrid_retriever = HybridRetriever(
            dense_retriever=get_retrieval_pipeline(),
            sparse_retriever=get_sparse_retriever(),
        )
    return _hybrid_retriever
