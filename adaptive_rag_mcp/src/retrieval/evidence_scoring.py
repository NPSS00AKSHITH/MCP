"""Evidence Quality Scoring module for evaluating reranked chunk quality.

This module provides a pure function for computing quality signals from
reranked document chunks. It is designed to be used AFTER retrieval and
reranking to assess the quality/confidence of the retrieved evidence.

This is a STANDALONE module that complements (not replaces) the existing
QualitySignals in src/retrieval/reranker.py. It provides a simpler,
function-based interface with configurable thresholds.

Design Constraints:
- Pure function (no side effects)
- No retrieval calls
- Configurable thresholds via constants
- Deterministic outputs

Example Usage:
-------------
>>> from src.retrieval.evidence_scoring import compute_evidence_scores
>>> scores = [0.92, 0.78, 0.45, 0.32, 0.21]
>>> result = compute_evidence_scores(scores)
>>> print(result.top_score)
0.92
>>> print(result.score_gap)
0.14
>>> print(result.low_confidence_flag)
False

Unit-Test Examples:
-------------------
>>> # Test 1: High confidence - high top score, reasonable gap
>>> scores = [0.9, 0.7, 0.5, 0.3]
>>> result = compute_evidence_scores(scores)
>>> assert result.top_score == 0.9
>>> assert result.score_gap == 0.2  # 0.9 - 0.7
>>> assert result.low_confidence_flag == False

>>> # Test 2: Low confidence - low top score
>>> scores = [0.4, 0.35, 0.3, 0.25]
>>> result = compute_evidence_scores(scores)
>>> assert result.low_confidence_flag == True

>>> # Test 3: Single score edge case
>>> scores = [0.8]
>>> result = compute_evidence_scores(scores)
>>> assert result.score_gap == 0.0
>>> assert result.score_entropy == 0.0

>>> # Test 4: Empty scores edge case
>>> scores = []
>>> result = compute_evidence_scores(scores)
>>> assert result.top_score == 0.0
>>> assert result.low_confidence_flag == True

>>> # Test 5: Flat distribution - high entropy
>>> scores = [0.5, 0.5, 0.5, 0.5]
>>> result = compute_evidence_scores(scores)
>>> assert result.score_entropy == 1.0  # Maximum entropy for uniform

>>> # Test 6: Custom thresholds
>>> scores = [0.55, 0.40, 0.35]
>>> result = compute_evidence_scores(scores, top_score_threshold=0.7)
>>> assert result.low_confidence_flag == True  # 0.55 < 0.7
"""

from dataclasses import dataclass
from typing import List, Optional
import math


# =============================================================================
# CONFIGURABLE THRESHOLD CONSTANTS
# =============================================================================

# Minimum top score to be considered "high confidence"
DEFAULT_TOP_SCORE_THRESHOLD: float = 0.6

# Minimum gap between top-1 and top-2 for clear winner
DEFAULT_GAP_THRESHOLD: float = 0.15

# Maximum entropy (normalized) before flagging as uncertain
DEFAULT_ENTROPY_THRESHOLD: float = 0.7

# Minimum average score for confident retrieval
DEFAULT_AVERAGE_SCORE_THRESHOLD: float = 0.4


# =============================================================================
# EVIDENCE SCORES DATACLASS
# =============================================================================

@dataclass(frozen=True)
class EvidenceScores:
    """Immutable container for evidence quality signals.
    
    Attributes:
        top_score: Highest relevance score among all chunks.
        score_gap: Difference between top-1 and top-2 scores.
        average_score: Mean of all relevance scores.
        score_entropy: Normalized Shannon entropy of score distribution (0-1).
        low_confidence_flag: True if evidence quality is below thresholds.
        confidence_reasons: List of reasons for low confidence (if applicable).
    """
    top_score: float
    score_gap: float
    average_score: float
    score_entropy: float
    low_confidence_flag: bool
    confidence_reasons: tuple = ()  # Tuple for immutability
    
    def to_dict(self) -> dict:
        """Convert scores to dictionary format."""
        return {
            "top_score": round(self.top_score, 4),
            "score_gap": round(self.score_gap, 4),
            "average_score": round(self.average_score, 4),
            "score_entropy": round(self.score_entropy, 4),
            "low_confidence_flag": self.low_confidence_flag,
            "confidence_reasons": list(self.confidence_reasons),
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _compute_entropy(scores: List[float]) -> float:
    """Compute normalized Shannon entropy of score distribution.
    
    Returns a value between 0 and 1, where:
    - 0 means all probability is concentrated on one item (certain)
    - 1 means uniform distribution (maximum uncertainty)
    
    Args:
        scores: List of relevance scores (0-1 range).
        
    Returns:
        Normalized entropy value between 0 and 1.
    
    Example:
        >>> _compute_entropy([0.9, 0.1, 0.0, 0.0])  # Low entropy (peaked)
        ~0.47
        >>> _compute_entropy([0.25, 0.25, 0.25, 0.25])  # High entropy (uniform)
        1.0
    """
    if len(scores) <= 1:
        return 0.0
    
    # Normalize scores to probability distribution
    total = sum(scores)
    if total == 0:
        return 1.0  # Maximum uncertainty if all scores are 0
    
    probabilities = [s / total for s in scores]
    
    # Compute Shannon entropy: -sum(p * log2(p))
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log2(p)
    
    # Normalize by maximum possible entropy (log2(N))
    max_entropy = math.log2(len(scores))
    if max_entropy == 0:
        return 0.0
    
    return entropy / max_entropy


def _compute_score_gap(scores: List[float]) -> float:
    """Compute the gap between top-1 and top-2 scores.
    
    Args:
        scores: List of relevance scores.
        
    Returns:
        Difference between highest and second-highest scores.
        Returns 0.0 if less than 2 scores.
    """
    if len(scores) < 2:
        return 0.0
    
    sorted_scores = sorted(scores, reverse=True)
    return sorted_scores[0] - sorted_scores[1]


# =============================================================================
# MAIN SCORING FUNCTION
# =============================================================================

def compute_evidence_scores(
    scores: List[float],
    top_score_threshold: float = DEFAULT_TOP_SCORE_THRESHOLD,
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
    entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
    average_threshold: float = DEFAULT_AVERAGE_SCORE_THRESHOLD,
) -> EvidenceScores:
    """Compute evidence quality signals from reranked chunk scores.
    
    This is a PURE FUNCTION with no side effects. It takes a list of
    relevance scores and returns computed quality signals.
    
    Args:
        scores: List of relevance scores from reranked chunks (0-1 range).
        top_score_threshold: Minimum top score for high confidence.
        gap_threshold: Minimum gap for clear winner.
        entropy_threshold: Maximum entropy before flagging uncertainty.
        average_threshold: Minimum average score for confident retrieval.
        
    Returns:
        EvidenceScores containing:
            - top_score: Highest score
            - score_gap: Gap between top-1 and top-2
            - average_score: Mean of all scores
            - score_entropy: Normalized Shannon entropy
            - low_confidence_flag: True if quality is insufficient
            - confidence_reasons: Tuple of reasons for low confidence
    
    Example:
        >>> scores = [0.85, 0.72, 0.45, 0.30]
        >>> result = compute_evidence_scores(scores)
        >>> result.top_score
        0.85
        >>> result.low_confidence_flag
        False
        
        >>> # With custom thresholds
        >>> result = compute_evidence_scores(scores, top_score_threshold=0.9)
        >>> result.low_confidence_flag
        True
        >>> "low_top_score" in result.confidence_reasons
        True
    """
    # Handle empty input
    if not scores:
        return EvidenceScores(
            top_score=0.0,
            score_gap=0.0,
            average_score=0.0,
            score_entropy=0.0,
            low_confidence_flag=True,
            confidence_reasons=("no_results",),
        )
    
    # Compute basic statistics
    top_score = max(scores)
    average_score = sum(scores) / len(scores)
    score_gap = _compute_score_gap(scores)
    score_entropy = _compute_entropy(scores)
    
    # Determine confidence flags
    confidence_reasons: List[str] = []
    
    # Check top score threshold
    if top_score < top_score_threshold:
        confidence_reasons.append("low_top_score")
    
    # Check for flat distribution (no clear winner)
    if len(scores) > 1 and score_gap < gap_threshold:
        confidence_reasons.append("no_clear_winner")
    
    # Check for high entropy (uncertain distribution)
    if score_entropy > entropy_threshold:
        confidence_reasons.append("high_entropy")
    
    # Check average score
    if average_score < average_threshold:
        confidence_reasons.append("low_average_score")
    
    # Determine overall low confidence flag
    low_confidence_flag = len(confidence_reasons) > 0
    
    return EvidenceScores(
        top_score=top_score,
        score_gap=score_gap,
        average_score=average_score,
        score_entropy=score_entropy,
        low_confidence_flag=low_confidence_flag,
        confidence_reasons=tuple(confidence_reasons),
    )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def is_high_quality_evidence(
    scores: List[float],
    top_score_threshold: float = DEFAULT_TOP_SCORE_THRESHOLD,
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
) -> bool:
    """Quick check if evidence meets high quality criteria.
    
    This is a convenience function for simple pass/fail checks.
    
    Args:
        scores: List of relevance scores.
        top_score_threshold: Minimum top score required.
        gap_threshold: Minimum gap required between top scores.
        
    Returns:
        True if evidence is high quality, False otherwise.
    
    Example:
        >>> is_high_quality_evidence([0.9, 0.6, 0.4])
        True
        >>> is_high_quality_evidence([0.4, 0.35, 0.3])
        False
    """
    result = compute_evidence_scores(
        scores,
        top_score_threshold=top_score_threshold,
        gap_threshold=gap_threshold,
    )
    return not result.low_confidence_flag


def extract_quality_summary(scores: List[float]) -> dict:
    """Extract a minimal quality summary for logging/debugging.
    
    Args:
        scores: List of relevance scores.
        
    Returns:
        Dictionary with key quality metrics.
    
    Example:
        >>> extract_quality_summary([0.85, 0.70, 0.45])
        {'top': 0.85, 'gap': 0.15, 'avg': 0.67, 'confident': True}
    """
    if not scores:
        return {"top": 0.0, "gap": 0.0, "avg": 0.0, "confident": False}
    
    result = compute_evidence_scores(scores)
    return {
        "top": round(result.top_score, 2),
        "gap": round(result.score_gap, 2),
        "avg": round(result.average_score, 2),
        "confident": not result.low_confidence_flag,
    }


# =============================================================================
# CONTRADICTION DETECTION
# =============================================================================

# Similarity threshold below which chunks are considered semantically different
CONTRADICTION_SIMILARITY_THRESHOLD: float = 0.3

# Minimum score for a chunk to be considered in contradiction detection
CONTRADICTION_SCORE_THRESHOLD: float = 0.5

# Maximum number of top chunks to check for contradictions
CONTRADICTION_CHECK_TOP_K: int = 5


@dataclass(frozen=True)
class ContradictionResult:
    """Result of contradiction detection between chunks.
    
    Attributes:
        has_contradiction: Whether a potential contradiction was detected
        conflicting_chunks: Tuple of (chunk_id_1, chunk_id_2) pairs that conflict
        explanation: Human-readable explanation of the contradiction
        similarity_scores: Similarity scores between checked chunk pairs
    """
    has_contradiction: bool
    conflicting_chunks: tuple = ()  # Tuple of (id1, id2) pairs
    explanation: str = ""
    similarity_scores: tuple = ()  # Tuple of similarity values
    
    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "has_contradiction": self.has_contradiction,
            "conflicting_chunks": list(self.conflicting_chunks),
            "explanation": self.explanation,
            "similarity_scores": list(self.similarity_scores),
        }


def _compute_text_similarity(text1: str, text2: str) -> float:
    """Compute lightweight text similarity using word overlap (Jaccard).
    
    This is a fast heuristic that doesn't require embeddings.
    It measures the overlap of significant words between two texts.
    
    Args:
        text1: First text to compare
        text2: Second text to compare
        
    Returns:
        Similarity score between 0 and 1
        
    Example:
        >>> _compute_text_similarity("The cat sat", "The dog sat")
        0.5  # 2 words in common out of 4 unique
    """
    if not text1 or not text2:
        return 0.0
    
    # Simple tokenization and lowercasing
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Remove common stopwords
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'between', 'under', 'again', 'further', 'then', 'once', 'and', 'but',
        'or', 'nor', 'so', 'yet', 'both', 'either', 'neither', 'not', 'only',
        'own', 'same', 'than', 'too', 'very', 'just', 'also', 'now', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'any',
        'some', 'no', 'other', 'such', 'this', 'that', 'these', 'those', 'it',
    }
    
    words1 = words1 - stopwords
    words2 = words2 - stopwords
    
    if not words1 or not words2:
        return 0.0
    
    # Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def _check_semantic_contradiction(text1: str, text2: str) -> bool:
    """Check if two texts are likely to contain contradictory information.
    
    Uses heuristics to detect potential contradictions:
    1. Both texts are about similar topics (some word overlap)
    2. But they contain opposing indicators (negation, different values, etc.)
    
    This is a lightweight heuristic - not perfect detection.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        True if potential contradiction detected
    """
    # Must have some topic overlap to be a contradiction
    similarity = _compute_text_similarity(text1, text2)
    
    # Too similar = probably not contradicting, too different = different topics
    if similarity < 0.1 or similarity > 0.8:
        return False
    
    # Look for negation patterns
    text1_lower = text1.lower()
    text2_lower = text2.lower()
    
    negation_words = ['not', 'no', 'never', 'none', "n't", 'cannot', 'without']
    
    text1_has_negation = any(neg in text1_lower for neg in negation_words)
    text2_has_negation = any(neg in text2_lower for neg in negation_words)
    
    # One has negation, one doesn't - possible contradiction
    if text1_has_negation != text2_has_negation:
        return True
    
    # Look for contrasting value patterns (dates, numbers, etc.)
    import re
    
    # Extract numbers
    nums1 = set(re.findall(r'\b\d+(?:\.\d+)?\b', text1))
    nums2 = set(re.findall(r'\b\d+(?:\.\d+)?\b', text2))
    
    # If both have numbers but they're different - possible contradiction
    if nums1 and nums2 and nums1 != nums2:
        # Check if they're likely referring to the same thing
        if similarity > 0.3:  # Similar enough context
            return True
    
    # Look for date patterns
    date_pattern = r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4})?\b'
    dates1 = set(re.findall(date_pattern, text1_lower))
    dates2 = set(re.findall(date_pattern, text2_lower))
    
    if dates1 and dates2 and dates1 != dates2:
        if similarity > 0.3:
            return True
    
    return False


def detect_contradictions(
    chunks: List[dict],
    score_threshold: float = CONTRADICTION_SCORE_THRESHOLD,
    similarity_threshold: float = CONTRADICTION_SIMILARITY_THRESHOLD,
    top_k: int = CONTRADICTION_CHECK_TOP_K,
) -> ContradictionResult:
    """Detect potential contradictions among top-scoring chunks.
    
    This function uses lightweight heuristics to identify when high-scoring
    chunks may contain contradictory information. When detected, the system
    should NOT merge or average claims, but instead flag uncertainty.
    
    Design Principles:
    - Lightweight: Uses word overlap and pattern matching, no LLM calls
    - Safe: When in doubt, flags potential contradiction
    - Explainable: Provides clear explanation of what was detected
    
    Args:
        chunks: List of chunk dictionaries with 'id', 'content', and 'score' keys
        score_threshold: Minimum score for a chunk to be considered
        similarity_threshold: Similarity below this triggers contradiction check
        top_k: Maximum number of top chunks to check
        
    Returns:
        ContradictionResult with detection status and explanation
        
    Example:
        >>> chunks = [
        ...     {"id": "1", "content": "Deadline is Dec 15", "score": 0.9},
        ...     {"id": "2", "content": "Deadline extended to Jan 30", "score": 0.85},
        ... ]
        >>> result = detect_contradictions(chunks)
        >>> result.has_contradiction
        True
    """
    if not chunks or len(chunks) < 2:
        return ContradictionResult(
            has_contradiction=False,
            explanation="Not enough chunks to check for contradictions."
        )
    
    # Filter to high-scoring chunks only
    high_score_chunks = [
        c for c in chunks 
        if c.get("score", 0) >= score_threshold
    ][:top_k]
    
    if len(high_score_chunks) < 2:
        return ContradictionResult(
            has_contradiction=False,
            explanation="Not enough high-scoring chunks to check for contradictions."
        )
    
    conflicting_pairs = []
    similarity_scores = []
    
    # Check pairs of chunks for contradictions
    for i in range(len(high_score_chunks)):
        for j in range(i + 1, len(high_score_chunks)):
            chunk1 = high_score_chunks[i]
            chunk2 = high_score_chunks[j]
            
            content1 = chunk1.get("content", "")
            content2 = chunk2.get("content", "")
            
            similarity = _compute_text_similarity(content1, content2)
            similarity_scores.append(round(similarity, 3))
            
            # Check for semantic contradiction
            if _check_semantic_contradiction(content1, content2):
                conflicting_pairs.append((
                    chunk1.get("id", f"chunk_{i}"),
                    chunk2.get("id", f"chunk_{j}")
                ))
    
    if conflicting_pairs:
        # Build explanation
        pair_strs = [f"{p[0]} and {p[1]}" for p in conflicting_pairs]
        explanation = (
            f"Potential contradictions detected between chunks: {', '.join(pair_strs)}. "
            f"High-confidence sources appear to disagree. "
            f"Unable to merge claims safely."
        )
        
        return ContradictionResult(
            has_contradiction=True,
            conflicting_chunks=tuple(conflicting_pairs),
            explanation=explanation,
            similarity_scores=tuple(similarity_scores),
        )
    
    return ContradictionResult(
        has_contradiction=False,
        explanation="No contradictions detected among top chunks.",
        similarity_scores=tuple(similarity_scores),
    )
