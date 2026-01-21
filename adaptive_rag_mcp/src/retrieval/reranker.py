"""Reranking module with pluggable rerankers and quality signals.

Rerankers take an initial set of retrieved documents and reorder them
based on more precise relevance scoring (typically using cross-encoders).

Quality Signals Computed:
========================

1. RELEVANCE SCORE: Direct cross-encoder score (0-1 normalized)
   - Measures how well document answers the query
   - Higher = more relevant

2. SCORE SPREAD: Standard deviation of top-k scores
   - High spread: Clear winner(s) among results
   - Low spread: Results are similarly relevant (or irrelevant)
   
3. LOW CONFIDENCE FLAGS:
   - "low_top_score": Best result has score < 0.3
   - "flat_distribution": Score spread < 0.1 (no clear winner)
   - "few_relevant": Less than 30% of results have score > 0.3
   - "score_drop": Large gap between #1 and #2 (potential outlier)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Protocol
import numpy as np

from src.server.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RerankResult:
    """Result from reranking."""
    
    id: str
    content: str
    relevance_score: float  # 0-1 normalized
    original_rank: int
    metadata: dict = field(default_factory=dict)


@dataclass
class QualitySignals:
    """Evidence quality signals computed from reranking."""
    
    top_score: float  # Highest relevance score
    score_spread: float  # Std dev of scores
    mean_score: float  # Average score
    score_gap: float # Gap between top 1 and top 2
    evidence_entropy: float # Shannon entropy of score distribution
    relevant_count: int  # Count of docs with score > threshold
    total_count: int  # Total docs evaluated
    confidence_flags: List[str]  # Warning flags
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if retrieval appears high quality."""
        return len(self.confidence_flags) == 0
    
    def to_dict(self) -> dict:
        return {
            "top_score": round(self.top_score, 4),
            "score_spread": round(self.score_spread, 4),
            "mean_score": round(self.mean_score, 4),
            "score_gap": round(self.score_gap, 4),
            "evidence_entropy": round(self.evidence_entropy, 4),
            "relevant_count": self.relevant_count,
            "total_count": self.total_count,
            "confidence_flags": self.confidence_flags,
            "is_high_confidence": self.is_high_confidence,
        }


class RerankerProtocol(Protocol):
    """Protocol for pluggable rerankers."""
    
    def rerank(
        self,
        query: str,
        documents: List[dict],
        top_k: int | None = None,
    ) -> tuple[List[RerankResult], QualitySignals]:
        """Rerank documents by relevance to query."""
        ...


class CrossEncoderReranker:
    """Cross-encoder based reranker using sentence-transformers.
    
    Cross-encoders score query-document pairs directly, providing
    more accurate relevance scores than bi-encoders (embedding similarity).
    """
    
    # Available cross-encoder models
    MODELS = {
        "cross-encoder/ms-marco-MiniLM-L-6-v2": {
            "desc": "Fast, good quality (recommended)",
            "max_length": 512,
        },
        "cross-encoder/ms-marco-TinyBERT-L-2-v2": {
            "desc": "Very fast, lower quality",
            "max_length": 512,
        },
        "cross-encoder/ms-marco-MiniLM-L-12-v2": {
            "desc": "Higher quality, slower",
            "max_length": 512,
        },
    }
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        relevance_threshold: float = 0.3,
    ):
        """Initialize cross-encoder reranker.
        
        Args:
            model_name: Cross-encoder model name.
            relevance_threshold: Threshold for considering a doc "relevant".
        """
        self.model_name = model_name
        self.relevance_threshold = relevance_threshold
        self._model = None
    
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info("loading_cross_encoder", model=self.model_name)
            self._model = CrossEncoder(self.model_name)
            logger.info("cross_encoder_loaded", model=self.model_name)
        return self._model
    
    def _normalize_score(self, score: float) -> float:
        """Normalize score to 0-1 range using sigmoid."""
        # Cross-encoder scores can be any range, normalize with sigmoid
        return 1 / (1 + np.exp(-score))
    
    def _compute_entropy(self, scores: np.ndarray) -> float:
        """Compute normalized Shannon entropy of score distribution."""
        if len(scores) <= 1:
            return 0.0
        
        # Normalize scores to probability distribution (sum = 1)
        # Handle cases where all scores are 0
        score_sum = scores.sum()
        if score_sum == 0:
            return 1.0 # Maximum uncertainty if all 0
            
        probs = scores / score_sum
        
        # Compute entropy: -sum(p * log(p))
        # Filter out 0 probabilities to avoid log(0)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        
        # Normalize by max possible entropy (log2(N))
        max_entropy = np.log2(len(scores))
        if max_entropy == 0:
            return 0.0
            
        return float(entropy / max_entropy)

    def _compute_quality_signals(
        self,
        scores: List[float],
    ) -> QualitySignals:
        """Compute quality signals from relevance scores."""
        if not scores:
            return QualitySignals(
                top_score=0.0,
                score_spread=0.0,
                mean_score=0.0,
                score_gap=0.0,
                evidence_entropy=0.0,
                relevant_count=0,
                total_count=0,
                confidence_flags=["no_results"],
            )
        
        scores_arr = np.array(scores)
        top_score = float(scores_arr.max())
        mean_score = float(scores_arr.mean())
        score_spread = float(scores_arr.std())
        relevant_count = int((scores_arr > self.relevance_threshold).sum())
        
        # Compute score gap (top 1 vs top 2)
        score_gap = 0.0
        if len(scores) >= 2:
            sorted_scores = sorted(scores, reverse=True)
            score_gap = sorted_scores[0] - sorted_scores[1]
            
        # Compute entropy
        evidence_entropy = self._compute_entropy(scores_arr)
        
        # Compute confidence flags
        flags = []
        
        # Low top score
        if top_score < self.relevance_threshold:
            flags.append("low_top_score")
        
        # Flat distribution (high entropy or low spread)
        if score_spread < 0.1 and len(scores) > 1:
            flags.append("flat_distribution")
            
        if evidence_entropy > 0.8 and len(scores) > 1:
             flags.append("high_entropy")

        # Few relevant results
        if len(scores) >= 3 and relevant_count / len(scores) < 0.3:
            flags.append("few_relevant")
        
        # Large score drop (potential outlier top result) - actually this is often GOOD in RAG
        # But if it's extreme, it might mean we only have one good doc.
        if score_gap > 0.4 and top_score > 0.7:
             # Just noting it, not necessarily a flag for "bad" quality, but "ragged" quality
             pass

        return QualitySignals(
            top_score=top_score,
            score_spread=score_spread,
            mean_score=mean_score,
            score_gap=score_gap,
            evidence_entropy=evidence_entropy,
            relevant_count=relevant_count,
            total_count=len(scores),
            confidence_flags=flags,
        )
    
    def rerank(
        self,
        query: str,
        documents: List[dict],
        top_k: int | None = None,
    ) -> tuple[List[RerankResult], QualitySignals]:
        """Rerank documents by relevance to query.
        
        Args:
            query: The query string.
            documents: List of dicts with 'id' and 'content' keys.
            top_k: Return only top K results (None = all).
            
        Returns:
            Tuple of (reranked results, quality signals).
        """
        if not documents:
            return [], QualitySignals(
                top_score=0.0,
                score_spread=0.0,
                mean_score=0.0,
                relevant_count=0,
                total_count=0,
                confidence_flags=["no_documents"],
            )
        
        logger.info("reranking", query_length=len(query), doc_count=len(documents))
        
        # Prepare query-document pairs
        pairs = [(query, doc.get("content", "")) for doc in documents]
        
        # Score with cross-encoder
        raw_scores = self.model.predict(pairs)
        
        # Normalize scores
        normalized_scores = [self._normalize_score(s) for s in raw_scores]
        
        # Create results with original ranks
        results = []
        for i, (doc, score) in enumerate(zip(documents, normalized_scores)):
            results.append(RerankResult(
                id=doc.get("id", f"doc_{i}"),
                content=doc.get("content", ""),
                relevance_score=round(score, 4),
                original_rank=i + 1,
                metadata=doc.get("metadata", {}),
            ))
        
        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Compute quality signals
        quality = self._compute_quality_signals(normalized_scores)
        
        # Apply top_k
        if top_k is not None:
            results = results[:top_k]
        
        logger.info(
            "reranking_complete",
            top_score=quality.top_score,
            score_gap=quality.score_gap,
            entropy=quality.evidence_entropy,
            flags=quality.confidence_flags,
        )
        
        return results, quality


class SimpleReranker:
    """Simple reranker that just sorts by existing scores.
    
    Useful as a fallback or for testing without loading ML models.
    """
    
    def __init__(self, relevance_threshold: float = 0.3):
        self.relevance_threshold = relevance_threshold
        self.model_name = "simple"
    
    def rerank(
        self,
        query: str,
        documents: List[dict],
        top_k: int | None = None,
    ) -> tuple[List[RerankResult], QualitySignals]:
        """Sort by existing score field."""
        if not documents:
            return [], QualitySignals(
                top_score=0.0,
                score_spread=0.0,
                mean_score=0.0,
                relevant_count=0,
                total_count=0,
                confidence_flags=["no_documents"],
            )
        
        # Extract scores (use 0.5 as default if no score)
        scores = [doc.get("score", 0.5) for doc in documents]
        
        results = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            results.append(RerankResult(
                id=doc.get("id", f"doc_{i}"),
                content=doc.get("content", ""),
                relevance_score=round(score, 4),
                original_rank=i + 1,
                metadata=doc.get("metadata", {}),
            ))
        
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Compute quality from existing scores
        quality = self._compute_quality_signals(scores)
        
        if top_k:
            results = results[:top_k]
        
        return results, quality
    
    def _compute_quality_signals(self, scores: List[float]) -> QualitySignals:
        """Compute quality signals from scores."""
        if not scores:
            return QualitySignals(
                top_score=0.0,
                score_spread=0.0,
                mean_score=0.0,
                relevant_count=0,
                total_count=0,
                confidence_flags=["no_results"],
            )
        
        scores_arr = np.array(scores)
        return QualitySignals(
            top_score=float(scores_arr.max()),
            score_spread=float(scores_arr.std()),
            mean_score=float(scores_arr.mean()),
            relevant_count=int((scores_arr > self.relevance_threshold).sum()),
            total_count=len(scores),
            confidence_flags=[],  # Simple reranker doesn't add flags
        )


# Reranker registry for pluggability
RERANKERS = {
    "cross-encoder": CrossEncoderReranker,
    "simple": SimpleReranker,
}


# Global reranker instance
_reranker: CrossEncoderReranker | SimpleReranker | None = None


def get_reranker(reranker_type: str = "cross-encoder") -> CrossEncoderReranker | SimpleReranker:
    """Get the global reranker instance."""
    global _reranker
    if _reranker is None:
        reranker_class = RERANKERS.get(reranker_type, CrossEncoderReranker)
        _reranker = reranker_class()
    return _reranker
