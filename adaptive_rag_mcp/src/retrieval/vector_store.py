"""FAISS-based vector store for dense retrieval.

Uses FAISS (Facebook AI Similarity Search) for efficient vector search.
Stores vectors in memory with optional persistence to disk.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import numpy as np

import faiss

from src.server.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Result from vector search."""
    
    chunk_id: str
    content: str
    score: float  # Similarity score (higher = more similar)
    metadata: dict
    doc_id: str


class FAISSVectorStore:
    """FAISS-based vector store for semantic search.
    
    Uses L2 distance with normalization for cosine similarity.
    Maintains a mapping from FAISS index positions to chunk IDs.
    """
    
    def __init__(
        self,
        dimensions: int = 384,
        index_path: str | Path = "./data/faiss_index",
    ):
        """Initialize vector store.
        
        Args:
            dimensions: Embedding dimensions.
            index_path: Path for persisting the index.
        """
        self.dimensions = dimensions
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # FAISS index
        self._index: Optional[faiss.IndexFlatIP] = None
        
        # Mapping from FAISS index position to chunk data
        self._id_map: List[dict] = []
        
        # Load existing index if present
        self._load_index()
    
    @property
    def index(self) -> faiss.IndexFlatIP:
        """Get or create FAISS index."""
        if self._index is None:
            # Using Inner Product (IP) for cosine similarity with normalized vectors
            self._index = faiss.IndexFlatIP(self.dimensions)
            logger.info("created_faiss_index", dimensions=self.dimensions)
        return self._index
    
    def _load_index(self) -> None:
        """Load index from disk if exists."""
        index_file = self.index_path / "index.faiss"
        map_file = self.index_path / "id_map.json"
        
        if index_file.exists() and map_file.exists():
            try:
                self._index = faiss.read_index(str(index_file))
                with open(map_file, "r") as f:
                    self._id_map = json.load(f)
                logger.info(
                    "loaded_faiss_index",
                    vectors=self._index.ntotal,
                    path=str(index_file),
                )
            except Exception as e:
                logger.error("failed_to_load_index", error=str(e))
                self._index = None
                self._id_map = []
    
    def _save_index(self) -> None:
        """Persist index to disk."""
        if self._index is None:
            return
        
        index_file = self.index_path / "index.faiss"
        map_file = self.index_path / "id_map.json"
        
        try:
            faiss.write_index(self._index, str(index_file))
            with open(map_file, "w") as f:
                json.dump(self._id_map, f)
            logger.info("saved_faiss_index", vectors=self._index.ntotal)
        except Exception as e:
            logger.error("failed_to_save_index", error=str(e))
    
    def add(
        self,
        chunk_id: str,
        doc_id: str,
        content: str,
        embedding: np.ndarray,
        metadata: dict | None = None,
    ) -> None:
        """Add a single vector to the index.
        
        Args:
            chunk_id: Unique chunk identifier.
            doc_id: Parent document ID.
            content: Text content of the chunk.
            embedding: Vector embedding.
            metadata: Optional metadata.
        """
        # Normalize for cosine similarity
        embedding = embedding.astype(np.float32)
        faiss.normalize_L2(embedding.reshape(1, -1))
        
        # Add to index
        self.index.add(embedding.reshape(1, -1))
        
        # Store mapping
        self._id_map.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "content": content,
            "metadata": metadata or {},
        })
    
    def add_batch(
        self,
        chunk_ids: List[str],
        doc_ids: List[str],
        contents: List[str],
        embeddings: np.ndarray,
        metadatas: List[dict] | None = None,
    ) -> int:
        """Add multiple vectors to the index.
        
        Args:
            chunk_ids: List of chunk identifiers.
            doc_ids: List of document IDs.
            contents: List of text contents.
            embeddings: Array of shape (n, dimensions).
            metadatas: Optional list of metadata dicts.
            
        Returns:
            Number of vectors added.
        """
        if len(chunk_ids) == 0:
            return 0
        
        metadatas = metadatas or [{} for _ in chunk_ids]
        
        # Normalize for cosine similarity
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store mappings
        for chunk_id, doc_id, content, metadata in zip(
            chunk_ids, doc_ids, contents, metadatas
        ):
            self._id_map.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "content": content,
                "metadata": metadata,
            })
        
        return len(chunk_ids)
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List[SearchResult]:
        """Search for similar vectors.
        
        Args:
            query_embedding: Query vector.
            k: Number of results to return.
            
        Returns:
            List of SearchResult objects.
        """
        if self.index.ntotal == 0:
            return []
        
        # Normalize query
        query = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        
        # Limit k to available vectors
        k = min(k, self.index.ntotal)
        
        # Search
        scores, indices = self.index.search(query, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            
            data = self._id_map[idx]
            results.append(SearchResult(
                chunk_id=data["chunk_id"],
                doc_id=data["doc_id"],
                content=data["content"],
                score=float(score),  # Cosine similarity (0-1)
                metadata=data["metadata"],
            ))
        
        return results
    
    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all vectors for a document.
        
        Note: FAISS doesn't support efficient deletion, so we rebuild the index.
        
        Args:
            doc_id: Document ID to delete.
            
        Returns:
            Number of vectors deleted.
        """
        if self.index.ntotal == 0:
            return 0
        
        # Find indices to keep
        keep_indices = [
            i for i, data in enumerate(self._id_map)
            if data["doc_id"] != doc_id
        ]
        
        deleted_count = len(self._id_map) - len(keep_indices)
        
        if deleted_count == 0:
            return 0
        
        # If deleting everything, just reset
        if len(keep_indices) == 0:
            self._index = None
            self._id_map = []
            return deleted_count
        
        # Reconstruct vectors for kept items
        all_vectors = faiss.rev_swig_ptr(
            self.index.get_xb(), self.index.ntotal * self.dimensions
        ).reshape(self.index.ntotal, self.dimensions).copy()
        
        kept_vectors = all_vectors[keep_indices]
        kept_map = [self._id_map[i] for i in keep_indices]
        
        # Rebuild index
        self._index = faiss.IndexFlatIP(self.dimensions)
        self._index.add(kept_vectors)
        self._id_map = kept_map
        
        return deleted_count
    
    def save(self) -> None:
        """Persist index to disk."""
        self._save_index()
    
    def count(self) -> int:
        """Get total number of vectors."""
        return self.index.ntotal if self._index else 0
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "total_vectors": self.count(),
            "dimensions": self.dimensions,
            "index_path": str(self.index_path),
        }


# Global vector store instance
_vector_store: FAISSVectorStore | None = None


def get_vector_store(dimensions: int = 384) -> FAISSVectorStore:
    """Get the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = FAISSVectorStore(dimensions=dimensions)
    return _vector_store
