"""BM25-based sparse retriever.

Uses term frequency-based ranking for keyword search.
"""

import re
from dataclasses import dataclass
from typing import List, Optional
import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.server.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SparseSearchResult:
    """Result from sparse search."""
    
    chunk_id: str
    content: str
    score: float
    metadata: dict
    doc_id: str


class BM25Retriever:
    """BM25-based sparse retriever.
    
    Uses Okapi BM25 for term frequency-based ranking.
    Suitable for keyword/lexical matching.
    """
    
    def __init__(self, index_path: str | Path = "./data/bm25_index"):
        """Initialize BM25 retriever.
        
        Args:
            index_path: Path for persisting the index.
        """
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # BM25 index
        self._bm25: Optional[BM25Okapi] = None
        
        # Document data
        self._documents: List[dict] = []
        self._tokenized_corpus: List[List[str]] = []
        
        # Load existing index
        self._load_index()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        # Simple tokenization: lowercase, split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def _load_index(self) -> None:
        """Load index from disk if exists."""
        data_file = self.index_path / "bm25_data.json"
        
        if data_file.exists():
            try:
                with open(data_file, "r") as f:
                    data = json.load(f)
                
                self._documents = data.get("documents", [])
                self._tokenized_corpus = [
                    self._tokenize(doc["content"]) 
                    for doc in self._documents
                ]
                
                if self._tokenized_corpus:
                    self._bm25 = BM25Okapi(self._tokenized_corpus)
                
                logger.info(
                    "loaded_bm25_index",
                    documents=len(self._documents),
                )
            except Exception as e:
                logger.error("failed_to_load_bm25_index", error=str(e))
                self._documents = []
                self._tokenized_corpus = []
                self._bm25 = None
    
    def _save_index(self) -> None:
        """Persist index to disk."""
        data_file = self.index_path / "bm25_data.json"
        
        try:
            with open(data_file, "w") as f:
                json.dump({"documents": self._documents}, f)
            logger.info("saved_bm25_index", documents=len(self._documents))
        except Exception as e:
            logger.error("failed_to_save_bm25_index", error=str(e))
    
    def add(
        self,
        chunk_id: str,
        doc_id: str,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        """Add a document to the index."""
        self._documents.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "content": content,
            "metadata": metadata or {},
        })
        self._tokenized_corpus.append(self._tokenize(content))
        
        # Rebuild BM25 index
        self._bm25 = BM25Okapi(self._tokenized_corpus)
    
    def add_batch(
        self,
        chunk_ids: List[str],
        doc_ids: List[str],
        contents: List[str],
        metadatas: List[dict] | None = None,
    ) -> int:
        """Add multiple documents to the index."""
        if not chunk_ids:
            return 0
        
        metadatas = metadatas or [{} for _ in chunk_ids]
        
        for chunk_id, doc_id, content, metadata in zip(
            chunk_ids, doc_ids, contents, metadatas
        ):
            self._documents.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "content": content,
                "metadata": metadata,
            })
            self._tokenized_corpus.append(self._tokenize(content))
        
        # Rebuild BM25 index
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        
        return len(chunk_ids)
    
    def search(self, query: str, k: int = 5) -> List[SparseSearchResult]:
        """Search for documents matching query.
        
        Args:
            query: Search query.
            k: Number of results to return.
            
        Returns:
            List of SparseSearchResult objects.
        """
        if not self._bm25 or not self._documents:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        
        # Get top k indices
        k = min(k, len(self._documents))
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                doc = self._documents[idx]
                results.append(SparseSearchResult(
                    chunk_id=doc["chunk_id"],
                    doc_id=doc["doc_id"],
                    content=doc["content"],
                    score=float(scores[idx]),
                    metadata=doc["metadata"],
                ))
        
        return results
    
    def delete_by_doc_id(self, doc_id: str) -> int:
        """Remove all documents for a doc_id."""
        original_count = len(self._documents)
        
        # Filter out documents
        keep_docs = [d for d in self._documents if d["doc_id"] != doc_id]
        deleted = original_count - len(keep_docs)
        
        if deleted > 0:
            self._documents = keep_docs
            self._tokenized_corpus = [
                self._tokenize(doc["content"]) 
                for doc in self._documents
            ]
            
            if self._tokenized_corpus:
                self._bm25 = BM25Okapi(self._tokenized_corpus)
            else:
                self._bm25 = None
        
        return deleted
    
    def save(self) -> None:
        """Persist index to disk."""
        self._save_index()
    
    def count(self) -> int:
        """Get total number of documents."""
        return len(self._documents)
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "total_documents": self.count(),
            "index_path": str(self.index_path),
        }


# Global sparse retriever instance
_sparse_retriever: BM25Retriever | None = None


def get_sparse_retriever() -> BM25Retriever:
    """Get the global sparse retriever instance."""
    global _sparse_retriever
    if _sparse_retriever is None:
        _sparse_retriever = BM25Retriever()
    return _sparse_retriever
