"""Retrieval pipeline - integrates embeddings with vector store.

Handles:
- Indexing chunks (embed and store)
- Searching (embed query and search)
"""

from typing import List

from src.retrieval.embedder import get_embedder, Embedder
from src.retrieval.vector_store import get_vector_store, FAISSVectorStore, SearchResult
from src.ingestion.pipeline import get_pipeline as get_ingestion_pipeline
from src.ingestion.chunker import Chunk
from src.server.logging import get_logger

logger = get_logger(__name__)


class RetrievalPipeline:
    """Pipeline for indexing and searching documents."""
    
    def __init__(
        self,
        embedder: Embedder | None = None,
        vector_store: FAISSVectorStore | None = None,
    ):
        """Initialize retrieval pipeline.
        
        Args:
            embedder: Embedding provider. Uses default if None.
            vector_store: Vector store. Uses default if None.
        """
        self.embedder = embedder or get_embedder()
        self.vector_store = vector_store or get_vector_store(
            dimensions=self.embedder.dimensions
        )
    
    def index_chunks(self, chunks: List[Chunk], save: bool = True) -> int:
        """Index a list of chunks.
        
        Args:
            chunks: Chunks to index.
            save: Whether to persist index after adding.
            
        Returns:
            Number of chunks indexed.
        """
        if not chunks:
            return 0
        
        # Extract texts and embed
        texts = [chunk.content for chunk in chunks]
        logger.info("embedding_chunks", count=len(chunks))
        embeddings = self.embedder.embed_batch(texts)
        
        # Add to vector store
        count = self.vector_store.add_batch(
            chunk_ids=[c.chunk_id for c in chunks],
            doc_ids=[c.doc_id for c in chunks],
            contents=texts,
            embeddings=embeddings,
            metadatas=[c.metadata for c in chunks],
        )
        
        if save:
            self.vector_store.save()
        
        logger.info("indexed_chunks", count=count)
        return count
    
    def index_document(self, doc_id: str) -> int:
        """Index all chunks from an ingested document.
        
        Args:
            doc_id: Document ID to index.
            
        Returns:
            Number of chunks indexed.
        """
        ingestion = get_ingestion_pipeline()
        chunks = ingestion.get_chunks(doc_id)
        
        if not chunks:
            logger.warning("no_chunks_found", doc_id=doc_id)
            return 0
        
        return self.index_chunks(chunks)
    
    def search(
        self,
        query: str,
        k: int = 5,
    ) -> List[SearchResult]:
        """Search for relevant chunks.
        
        Args:
            query: Natural language query.
            k: Number of results to return.
            
        Returns:
            List of SearchResult objects.
        """
        logger.info("searching", query_length=len(query), k=k)
        
        # Embed query
        query_embedding = self.embedder.embed_query(query)
        
        # Search
        results = self.vector_store.search(query_embedding, k=k)
        
        logger.info("search_complete", results=len(results))
        return results
    
    def delete_document(self, doc_id: str) -> int:
        """Remove document from vector index.
        
        Args:
            doc_id: Document ID to remove.
            
        Returns:
            Number of vectors deleted.
        """
        count = self.vector_store.delete_by_doc_id(doc_id)
        if count > 0:
            self.vector_store.save()
        return count
    
    def get_stats(self) -> dict:
        """Get retrieval statistics."""
        return {
            "embedder_model": self.embedder.model_name,
            "embedder_dimensions": self.embedder.dimensions,
            **self.vector_store.get_stats(),
        }


# Global pipeline instance
_pipeline: RetrievalPipeline | None = None


def get_retrieval_pipeline() -> RetrievalPipeline:
    """Get the global retrieval pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RetrievalPipeline()
    return _pipeline
