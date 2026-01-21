"""Embedding provider using sentence-transformers.

Uses open-source models for generating vector embeddings.
Default model: all-MiniLM-L6-v2 (fast, good quality, 384 dimensions)
"""

from functools import lru_cache
from typing import List
import numpy as np

from sentence_transformers import SentenceTransformer

from src.server.logging import get_logger
from src.config import get_settings

logger = get_logger(__name__)


class Embedder:
    """Embedding provider using sentence-transformers.
    
    Converts text into dense vector representations for semantic search.
    """
    
    # Popular open-source models
    MODELS = {
        "all-MiniLM-L6-v2": {"dim": 384, "desc": "Fast, good quality"},
        "all-mpnet-base-v2": {"dim": 768, "desc": "Higher quality, slower"},
        "paraphrase-MiniLM-L6-v2": {"dim": 384, "desc": "Good for paraphrase"},
    }
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedder with specified model.
        
        Args:
            model_name: Name of the sentence-transformers model.
        """
        self.model_name = model_name
        self._model = None
        self._dimensions = self.MODELS.get(model_name, {}).get("dim", 384)
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            logger.info("loading_embedding_model", model=self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("embedding_model_loaded", model=self.model_name)
        return self._model
    
    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        return self._dimensions
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text.
        
        Args:
            text: Text to embed.
            
        Returns:
            Numpy array of shape (dimensions,)
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed multiple texts efficiently.
        
        Args:
            texts: List of texts to embed.
            batch_size: Batch size for encoding.
            
        Returns:
            Numpy array of shape (len(texts), dimensions)
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.dimensions)
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.astype(np.float32)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query (alias for embed_text).
        
        Some models have different encoding for queries vs documents.
        This method can be overridden for such models.
        """
        return self.embed_text(query)


# Global embedder instance
_embedder: Embedder | None = None


def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> Embedder:
    """Get the global embedder instance."""
    global _embedder
    if _embedder is None or _embedder.model_name != model_name:
        _embedder = Embedder(model_name)
    return _embedder
