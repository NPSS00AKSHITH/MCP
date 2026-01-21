"""Self-improving memory system inspired by Microsoft NextCoder.

Learns from user feedback to boost retrieval quality over time:
- Tracks which chunks successfully answered queries
- Records query patterns and their successful retrievals
- Boosts scores of historically successful chunks
- Decays scores of unsuccessful retrievals

This transforms static retrieval into an adaptive expert system.

Design Constraints:
- Must not break existing reranker API
- Must persist memory to disk
- Must decay old feedback (recency bias)
- Must handle cold-start (no history yet)
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict

from src.server.logging import get_logger
from src.retrieval.reranker import get_reranker, RerankResult, QualitySignals

logger = get_logger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Boost factor: max score boost (20%)
DEFAULT_BOOST_FACTOR = 0.2

# Decay rate for old feedback (exponential decay)
DEFAULT_DECAY_RATE = 0.95

# Learning rate for EMA updates
DEFAULT_LEARNING_RATE = 0.1

# Neutral prior for chunks with no history
NEUTRAL_SUCCESS_RATE = 0.5

# Minimum confidence to apply boost
MIN_CONFIDENCE_FOR_BOOST = 0.3


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FeedbackRecord:
    """Record of user feedback on a retrieval result.
    
    Attributes:
        chunk_id: ID of the chunk that was retrieved
        query: The query that retrieved it
        query_type: Classification of query (from policy)
        timestamp: Unix timestamp when feedback was recorded
        accepted: Whether user found this result helpful
        relevance_score: Original reranker score
    """
    chunk_id: str
    query: str
    query_type: str
    timestamp: float
    accepted: bool
    relevance_score: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "query": self.query,
            "query_type": self.query_type,
            "timestamp": self.timestamp,
            "accepted": self.accepted,
            "relevance_score": self.relevance_score,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "FeedbackRecord":
        """Create from dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            query=data["query"],
            query_type=data["query_type"],
            timestamp=data["timestamp"],
            accepted=data["accepted"],
            relevance_score=data["relevance_score"],
        )


# =============================================================================
# ADAPTIVE MEMORY RANKER
# =============================================================================

class AdaptiveMemoryRanker:
    """Memory-augmented reranker that learns from feedback.
    
    Architecture:
    1. Base reranking: Use standard cross-encoder (unchanged)
    2. Memory boost: Adjust scores based on historical success
    3. Feedback recording: Learn from user interactions
    4. Pattern recognition: Identify query types that work
    
    Example:
        >>> memory = AdaptiveMemoryRanker()
        >>> 
        >>> # First query - no history
        >>> results = memory.rerank_with_memory(query, docs)
        >>> 
        >>> # User accepts result
        >>> memory.record_feedback(results[0].id, query, accepted=True)
        >>> 
        >>> # Second similar query - boost previously successful chunk
        >>> results = memory.rerank_with_memory(similar_query, docs)
        >>> # Results[0] gets boosted score because it worked before
    """
    
    def __init__(
        self,
        base_reranker=None,
        memory_path: Path | str = Path("./data/adaptive_memory.json"),
        boost_factor: float = DEFAULT_BOOST_FACTOR,
        decay_rate: float = DEFAULT_DECAY_RATE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
    ):
        """Initialize adaptive memory ranker.
        
        Args:
            base_reranker: Base reranker instance. Uses global if not provided.
            memory_path: Path to persist memory JSON file.
            boost_factor: Maximum score boost (0.2 = 20%).
            decay_rate: Decay rate for old feedback (0.95 = 5% decay per period).
            learning_rate: Learning rate for EMA updates.
        """
        self.base_reranker = base_reranker or get_reranker()
        self.memory_path = Path(memory_path)
        self.boost_factor = boost_factor
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        
        # Memory stores
        self.chunk_success_rates: Dict[str, float] = {}  # chunk_id -> success_rate
        self.query_patterns: Dict[str, List[str]] = defaultdict(list)  # query_type -> successful_chunk_ids
        self.feedback_history: List[FeedbackRecord] = []
        
        self._load_memory()
    
    def rerank_with_memory(
        self,
        query: str,
        documents: List[dict],
        query_type: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[List[RerankResult], QualitySignals]:
        """Rerank with memory-based score boosting.
        
        Process:
        1. Base reranking (cross-encoder)
        2. Apply memory boost to historically successful chunks
        3. Re-sort by adjusted scores
        4. Return results (same format as base reranker)
        
        Args:
            query: Search query
            documents: Documents to rerank
            query_type: Optional query classification (from policy)
            top_k: Optional limit on results
            
        Returns:
            Tuple of (reranked_results, quality_signals)
            Same format as base reranker - drop-in replacement
        """
        if not documents:
            # Return empty results with empty quality signals
            empty_quality = QualitySignals(
                top_score=0,
                score_spread=0,
                mean_score=0,
                score_gap=0,
                evidence_entropy=0,
                relevant_count=0,
                total_count=0,
                confidence_flags=["no_documents"],
            )
            return [], empty_quality
        
        # Step 1: Base reranking
        base_results, quality = self.base_reranker.rerank(query, documents, top_k=None)
        
        if not base_results:
            return base_results, quality
        
        # Step 2: Apply memory boost
        boosted_results = self._apply_memory_boost(base_results, query_type)
        
        # Step 3: Re-sort by adjusted scores
        boosted_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Step 4: Apply top_k
        if top_k:
            boosted_results = boosted_results[:top_k]
        
        # Log memory application
        original_top = base_results[0].relevance_score if base_results else 0
        boosted_top = boosted_results[0].relevance_score if boosted_results else 0
        
        logger.info(
            "memory_reranking_complete",
            original_top_score=round(original_top, 4),
            boosted_top_score=round(boosted_top, 4),
            boost_applied=abs(boosted_top - original_top) > 0.001,
            memory_chunks_tracked=len(self.chunk_success_rates),
        )
        
        return boosted_results, quality
    
    def _apply_memory_boost(
        self,
        results: List[RerankResult],
        query_type: Optional[str],
    ) -> List[RerankResult]:
        """Apply memory-based score boost to results.
        
        Boosts scores based on:
        1. Chunk's historical success rate
        2. Match with query type patterns
        
        Args:
            results: Base reranking results
            query_type: Optional query classification
            
        Returns:
            Results with adjusted scores
        """
        boosted_results = []
        
        for result in results:
            chunk_id = result.id
            score = result.relevance_score
            boost = 0.0
            
            # Check if chunk has success history
            if chunk_id in self.chunk_success_rates:
                success_rate = self.chunk_success_rates[chunk_id]
                # Only boost if success rate is above neutral
                if success_rate > NEUTRAL_SUCCESS_RATE:
                    boost = (success_rate - NEUTRAL_SUCCESS_RATE) * self.boost_factor * 2
            
            # Check query pattern match
            if query_type and query_type in self.query_patterns:
                successful_chunks = self.query_patterns[query_type]
                if chunk_id in successful_chunks:
                    # Extra boost for query type match
                    boost += self.boost_factor * 0.5
            
            # Apply boost (capped at boost_factor)
            boost = min(boost, self.boost_factor)
            adjusted_score = score * (1 + boost)
            
            # Create new result with adjusted score
            boosted_result = RerankResult(
                id=result.id,
                content=result.content,
                relevance_score=adjusted_score,
                original_rank=result.original_rank,
                metadata={
                    **result.metadata,
                    "memory_boost": round(boost, 4),
                    "original_score": round(score, 4),
                },
            )
            boosted_results.append(boosted_result)
        
        return boosted_results
    
    def record_feedback(
        self,
        chunk_id: str,
        query: str,
        query_type: str,
        accepted: bool,
        original_score: float = 0.0,
    ) -> None:
        """Record user feedback to improve future retrievals.
        
        This is called when:
        - User accepts a result (accepted=True)
        - User explicitly rejects a result (accepted=False)
        - System detects successful task completion
        
        Args:
            chunk_id: Chunk that was retrieved
            query: Query that retrieved it
            query_type: Classification of query
            accepted: Did user find this helpful?
            original_score: Original relevance score
        """
        # Create feedback record
        record = FeedbackRecord(
            chunk_id=chunk_id,
            query=query,
            query_type=query_type,
            timestamp=time.time(),
            accepted=accepted,
            relevance_score=original_score,
        )
        self.feedback_history.append(record)
        
        # Update success rate using exponential moving average (EMA)
        if chunk_id not in self.chunk_success_rates:
            self.chunk_success_rates[chunk_id] = NEUTRAL_SUCCESS_RATE
        
        # EMA update
        new_value = 1.0 if accepted else 0.0
        current = self.chunk_success_rates[chunk_id]
        self.chunk_success_rates[chunk_id] = (
            self.learning_rate * new_value + (1 - self.learning_rate) * current
        )
        
        # Update query pattern memory
        if accepted:
            if chunk_id not in self.query_patterns[query_type]:
                self.query_patterns[query_type].append(chunk_id)
        else:
            # Remove from pattern if rejected
            if chunk_id in self.query_patterns.get(query_type, []):
                self.query_patterns[query_type].remove(chunk_id)
        
        logger.info(
            "feedback_recorded",
            chunk_id=chunk_id,
            query_type=query_type,
            accepted=accepted,
            new_success_rate=round(self.chunk_success_rates[chunk_id], 4),
            total_feedback=len(self.feedback_history),
        )
        
        # Persist to disk
        self._save_memory()
    
    def apply_decay(self) -> None:
        """Apply decay to old success rates.
        
        Call periodically to reduce influence of old feedback.
        """
        for chunk_id in self.chunk_success_rates:
            # Decay towards neutral
            current = self.chunk_success_rates[chunk_id]
            self.chunk_success_rates[chunk_id] = (
                NEUTRAL_SUCCESS_RATE + 
                (current - NEUTRAL_SUCCESS_RATE) * self.decay_rate
            )
        
        logger.debug("decay_applied", chunks_affected=len(self.chunk_success_rates))
    
    def _load_memory(self) -> None:
        """Load memory from disk."""
        if not self.memory_path.exists():
            logger.debug("memory_file_not_found", path=str(self.memory_path))
            return
        
        try:
            with open(self.memory_path, 'r') as f:
                data = json.load(f)
            
            self.chunk_success_rates = data.get("chunk_success_rates", {})
            self.query_patterns = defaultdict(list, data.get("query_patterns", {}))
            self.feedback_history = [
                FeedbackRecord.from_dict(r)
                for r in data.get("feedback_history", [])
            ]
            
            logger.info(
                "memory_loaded",
                chunks_tracked=len(self.chunk_success_rates),
                query_patterns=len(self.query_patterns),
                feedback_records=len(self.feedback_history),
            )
        except Exception as e:
            logger.warning("memory_load_failed", error=str(e))
    
    def _save_memory(self) -> None:
        """Persist memory to disk."""
        try:
            # Ensure directory exists
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "chunk_success_rates": self.chunk_success_rates,
                "query_patterns": dict(self.query_patterns),
                "feedback_history": [r.to_dict() for r in self.feedback_history[-1000:]],  # Keep last 1000
                "last_updated": time.time(),
            }
            
            with open(self.memory_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug("memory_saved", path=str(self.memory_path))
        except Exception as e:
            logger.warning("memory_save_failed", error=str(e))
    
    def get_stats(self) -> dict:
        """Return memory statistics.
        
        Returns:
            Dictionary with memory stats for monitoring.
        """
        return {
            "chunks_tracked": len(self.chunk_success_rates),
            "query_patterns": {k: len(v) for k, v in self.query_patterns.items()},
            "total_feedback": len(self.feedback_history),
            "memory_path": str(self.memory_path),
            "boost_factor": self.boost_factor,
            "decay_rate": self.decay_rate,
        }
    
    def clear_memory(self) -> None:
        """Clear all memory (for testing/reset)."""
        self.chunk_success_rates = {}
        self.query_patterns = defaultdict(list)
        self.feedback_history = []
        self._save_memory()
        logger.info("memory_cleared")


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_adaptive_memory: Optional[AdaptiveMemoryRanker] = None


def get_adaptive_memory(memory_path: Optional[Path | str] = None) -> AdaptiveMemoryRanker:
    """Get or create global adaptive memory instance.
    
    Args:
        memory_path: Optional custom path for memory persistence.
        
    Returns:
        The global AdaptiveMemoryRanker instance.
    """
    global _adaptive_memory
    if _adaptive_memory is None:
        path = memory_path or Path("./data/adaptive_memory.json")
        _adaptive_memory = AdaptiveMemoryRanker(memory_path=path)
    return _adaptive_memory
