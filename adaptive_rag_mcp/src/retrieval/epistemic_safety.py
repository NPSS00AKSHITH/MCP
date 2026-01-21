"""Epistemic Safety Module for Explicit Outcomes and Stopping Rules.

This module provides the core epistemic safety infrastructure:
1. Configurable stopping rule constants
2. Explicit outcome taxonomy with confidence levels
3. Stopping decision evaluation with explanations
4. Structured logging for all decisions

Design Principles:
- All outcomes are explicit and explained
- No silent stopping - every decision has a reason
- Prefer refusing over guessing
- Confidence levels are transparent

Example Usage:
-------------
>>> from src.retrieval.epistemic_safety import (
...     StoppingRuleEvaluator,
...     EpistemicOutcome,
...     ConfidenceLevel,
... )
>>> evaluator = StoppingRuleEvaluator()
>>> decision = evaluator.evaluate(
...     top_score=0.85,
...     score_gap=0.20,
...     score_entropy=0.3,
...     iteration=1,
...     has_contradiction=False,
... )
>>> print(decision.stopped)
True
>>> print(decision.triggered_rule)
'HIGH_CONFIDENCE'
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from src.server.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# CONFIGURABLE STOPPING RULE CONSTANTS
# =============================================================================

# High confidence threshold - top score must exceed this for answer_ready
HIGH_CONFIDENCE_THRESHOLD: float = 0.7

# Score gap threshold - gap between top-1 and top-2 for clear winner
SCORE_GAP_THRESHOLD: float = 0.15

# Entropy threshold - maximum normalized entropy before flagging uncertainty
ENTROPY_THRESHOLD: float = 0.7

# Maximum iterations before forcing stop
MAX_ITERATIONS: int = 3

# Thresholds for confidence level determination
MEDIUM_CONFIDENCE_THRESHOLD: float = 0.5  # Above this is medium, below is low


# =============================================================================
# CONFIDENCE LEVEL ENUM
# =============================================================================

class ConfidenceLevel(Enum):
    """Confidence levels for epistemic outcomes.
    
    HIGH: Strong evidence, clear winner, no contradictions
    MEDIUM: Moderate evidence, some uncertainty
    LOW: Weak evidence, high uncertainty, or contradictions
    """
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# STOPPING RULE ENUM
# =============================================================================

class StoppingRule(Enum):
    """Named stopping rules with semantic meaning."""
    HIGH_CONFIDENCE = "high_confidence"
    SCORE_GAP_CLEAR = "score_gap_clear"
    ENTROPY_LOW = "entropy_low"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    CONTRADICTION_DETECTED = "contradiction_detected"
    POLICY_SKIP = "policy_skip"
    NO_RESULTS = "no_results"
    ERROR = "error"
    HYPOTHETICAL_SUCCESS = "hypothetical_success"
    NOT_STOPPED = "not_stopped"


# =============================================================================
# EPISTEMIC OUTCOME DATACLASS
# =============================================================================

@dataclass(frozen=True)
class EpistemicOutcome:
    """Structured outcome type for MCP responses.
    
    Every retrieval operation MUST return an EpistemicOutcome that includes:
    - outcome_type: What kind of result this is
    - explanation_reason: WHY this outcome was chosen (human-readable)
    - confidence_level: How confident the system is in this outcome
    
    Attributes:
        outcome_type: One of answer_ready, partial_answer, insufficient_evidence, clarification_needed
        explanation_reason: Human-readable explanation for WHY this outcome was chosen
        confidence_level: high, medium, or low
        triggered_rules: List of stopping rules that contributed to this outcome
    """
    outcome_type: str
    explanation_reason: str
    confidence_level: ConfidenceLevel
    triggered_rules: Tuple[str, ...] = ()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for MCP response."""
        return {
            "outcome_type": self.outcome_type,
            "explanation_reason": self.explanation_reason,
            "confidence_level": self.confidence_level.value,
            "triggered_rules": list(self.triggered_rules),
        }


# =============================================================================
# STOPPING DECISION DATACLASS
# =============================================================================

@dataclass(frozen=True)
class StoppingDecision:
    """Result of evaluating stopping rules.
    
    Attributes:
        stopped: Whether retrieval should stop
        reason: Short reason code
        triggered_rule: Which rule caused the stop (if any)
        explanation: Human-readable explanation
        confidence_level: Determined confidence level
        should_retry: Whether to retry with different parameters
    """
    stopped: bool
    reason: str
    triggered_rule: StoppingRule
    explanation: str
    confidence_level: ConfidenceLevel
    should_retry: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "stopped": self.stopped,
            "reason": self.reason,
            "triggered_rule": self.triggered_rule.value,
            "explanation": self.explanation,
            "confidence_level": self.confidence_level.value,
            "should_retry": self.should_retry,
        }


# =============================================================================
# STOPPING RULE EVALUATOR
# =============================================================================

class StoppingRuleEvaluator:
    """Evaluates stopping conditions with explicit explanations.
    
    This evaluator checks all stopping conditions and provides:
    - Clear decision on whether to stop
    - Explanation of WHY the decision was made
    - Confidence level based on evidence quality
    
    All decisions are logged for transparency.
    
    Example:
        >>> evaluator = StoppingRuleEvaluator()
        >>> decision = evaluator.evaluate(
        ...     top_score=0.85,
        ...     score_gap=0.20,
        ...     score_entropy=0.3,
        ...     iteration=1,
        ...     max_iterations=3,
        ... )
        >>> print(decision.stopped)
        True
        >>> print(decision.triggered_rule)
        StoppingRule.HIGH_CONFIDENCE
    """
    
    def __init__(
        self,
        high_confidence_threshold: float = HIGH_CONFIDENCE_THRESHOLD,
        score_gap_threshold: float = SCORE_GAP_THRESHOLD,
        entropy_threshold: float = ENTROPY_THRESHOLD,
        medium_confidence_threshold: float = MEDIUM_CONFIDENCE_THRESHOLD,
    ):
        """Initialize the evaluator with configurable thresholds.
        
        Args:
            high_confidence_threshold: Minimum top score for high confidence
            score_gap_threshold: Minimum gap for clear winner
            entropy_threshold: Maximum entropy for low uncertainty
            medium_confidence_threshold: Threshold for medium vs low confidence
        """
        self.high_confidence_threshold = high_confidence_threshold
        self.score_gap_threshold = score_gap_threshold
        self.entropy_threshold = entropy_threshold
        self.medium_confidence_threshold = medium_confidence_threshold
    
    def _determine_confidence_level(
        self,
        top_score: float,
        score_gap: float,
        has_contradiction: bool,
    ) -> ConfidenceLevel:
        """Determine confidence level based on evidence quality.
        
        HIGH: top_score >= 0.7 AND score_gap >= 0.15 AND no contradictions
        MEDIUM: top_score >= 0.5 AND top_score < 0.7
        LOW: top_score < 0.5 OR contradictions detected
        
        Args:
            top_score: Highest relevance score
            score_gap: Gap between top-1 and top-2
            has_contradiction: Whether contradictions were detected
            
        Returns:
            ConfidenceLevel enum value
        """
        if has_contradiction:
            return ConfidenceLevel.LOW
        
        if (
            top_score >= self.high_confidence_threshold
            and score_gap >= self.score_gap_threshold
        ):
            return ConfidenceLevel.HIGH
        
        if top_score >= self.medium_confidence_threshold:
            return ConfidenceLevel.MEDIUM
        
        return ConfidenceLevel.LOW
    
    def evaluate(
        self,
        top_score: float,
        score_gap: float,
        score_entropy: float,
        iteration: int,
        max_iterations: int = MAX_ITERATIONS,
        has_contradiction: bool = False,
        has_results: bool = True,
        policy_skip: bool = False,
        error_occurred: bool = False,
        hypothetical_improved: bool = False,
    ) -> StoppingDecision:
        """Evaluate all stopping conditions and return decision.
        
        This method checks conditions in priority order:
        1. Error occurred
        2. Policy skip
        3. No results
        4. Contradiction detected
        5. High confidence achieved
        6. Hypothetical reasoning improved results
        7. Max iterations reached
        8. Continue (not stopped)
        
        Args:
            top_score: Highest relevance score
            score_gap: Gap between top-1 and top-2
            score_entropy: Normalized Shannon entropy
            iteration: Current iteration number
            max_iterations: Maximum allowed iterations
            has_contradiction: Whether contradictions were detected
            has_results: Whether any results were retrieved
            policy_skip: Whether policy decided to skip retrieval
            error_occurred: Whether an error occurred
            hypothetical_improved: Whether HyDE improved results
            
        Returns:
            StoppingDecision with full explanation
        """
        confidence_level = self._determine_confidence_level(
            top_score, score_gap, has_contradiction
        )
        
        # Priority 1: Error
        if error_occurred:
            decision = StoppingDecision(
                stopped=True,
                reason="error",
                triggered_rule=StoppingRule.ERROR,
                explanation="An error occurred during retrieval.",
                confidence_level=ConfidenceLevel.LOW,
                should_retry=False,
            )
            self._log_decision(decision, iteration)
            return decision
        
        # Priority 2: Policy skip
        if policy_skip:
            decision = StoppingDecision(
                stopped=True,
                reason="policy_skip",
                triggered_rule=StoppingRule.POLICY_SKIP,
                explanation="Policy engine decided to skip retrieval.",
                confidence_level=ConfidenceLevel.LOW,
                should_retry=False,
            )
            self._log_decision(decision, iteration)
            return decision
        
        # Priority 3: No results
        if not has_results:
            decision = StoppingDecision(
                stopped=True,
                reason="no_results",
                triggered_rule=StoppingRule.NO_RESULTS,
                explanation="No documents found matching the query.",
                confidence_level=ConfidenceLevel.LOW,
                should_retry=False,
            )
            self._log_decision(decision, iteration)
            return decision
        
        # Priority 4: Contradiction detected (stop but with explanation)
        if has_contradiction:
            decision = StoppingDecision(
                stopped=True,
                reason="contradiction",
                triggered_rule=StoppingRule.CONTRADICTION_DETECTED,
                explanation=(
                    f"Conflicting sources detected. Top score is {top_score:.2f} "
                    f"but evidence contains contradictions. "
                    f"Returning partial results with caution."
                ),
                confidence_level=ConfidenceLevel.LOW,
                should_retry=False,
            )
            self._log_decision(decision, iteration)
            return decision
        
        # Priority 5: High confidence
        if (
            top_score >= self.high_confidence_threshold
            and score_gap >= self.score_gap_threshold
            and score_entropy <= self.entropy_threshold
        ):
            decision = StoppingDecision(
                stopped=True,
                reason="high_confidence",
                triggered_rule=StoppingRule.HIGH_CONFIDENCE,
                explanation=(
                    f"High confidence evidence found. "
                    f"Top score: {top_score:.2f}, score gap: {score_gap:.2f}, "
                    f"entropy: {score_entropy:.2f}. No contradictions detected."
                ),
                confidence_level=ConfidenceLevel.HIGH,
                should_retry=False,
            )
            self._log_decision(decision, iteration)
            return decision
        
        # Priority 6: Hypothetical reasoning improved results
        if hypothetical_improved and top_score >= self.medium_confidence_threshold:
            decision = StoppingDecision(
                stopped=True,
                reason="hypothetical_success",
                triggered_rule=StoppingRule.HYPOTHETICAL_SUCCESS,
                explanation=(
                    f"Found relevant documents via hypothetical expansion. "
                    f"Top score improved to {top_score:.2f}."
                ),
                confidence_level=confidence_level,
                should_retry=False,
            )
            self._log_decision(decision, iteration)
            return decision
        
        # Priority 7: Max iterations reached
        if iteration >= max_iterations:
            decision = StoppingDecision(
                stopped=True,
                reason="max_iterations",
                triggered_rule=StoppingRule.MAX_ITERATIONS_REACHED,
                explanation=(
                    f"Max iterations ({max_iterations}) reached. "
                    f"Best top score: {top_score:.2f}. "
                    f"Confidence: {confidence_level.value}."
                ),
                confidence_level=confidence_level,
                should_retry=False,
            )
            self._log_decision(decision, iteration)
            return decision
        
        # Priority 8: Continue (not stopped)
        decision = StoppingDecision(
            stopped=False,
            reason="continue",
            triggered_rule=StoppingRule.NOT_STOPPED,
            explanation=(
                f"Confidence not yet sufficient. "
                f"Top score: {top_score:.2f} (need >= {self.high_confidence_threshold}). "
                f"Will retry with adjusted parameters."
            ),
            confidence_level=confidence_level,
            should_retry=True,
        )
        self._log_decision(decision, iteration)
        return decision
    
    def _log_decision(self, decision: StoppingDecision, iteration: int) -> None:
        """Log the stopping decision with full context."""
        logger.info(
            "stopping_rule_evaluated",
            iteration=iteration,
            stopped=decision.stopped,
            triggered_rule=decision.triggered_rule.value,
            reason=decision.reason,
            confidence_level=decision.confidence_level.value,
            explanation=decision.explanation,
        )


# =============================================================================
# OUTCOME FACTORY FUNCTIONS
# =============================================================================

def create_answer_ready_outcome(
    top_score: float,
    triggered_rules: Optional[List[str]] = None,
) -> EpistemicOutcome:
    """Create an answer_ready outcome with high confidence.
    
    Args:
        top_score: The top relevance score achieved
        triggered_rules: List of rules that contributed to this outcome
        
    Returns:
        EpistemicOutcome with answer_ready type
    """
    outcome = EpistemicOutcome(
        outcome_type="answer_ready",
        explanation_reason=(
            f"High confidence evidence found with top_score={top_score:.2f}. "
            f"Evidence is consistent and no contradictions detected."
        ),
        confidence_level=ConfidenceLevel.HIGH,
        triggered_rules=tuple(triggered_rules or ["HIGH_CONFIDENCE"]),
    )
    
    logger.info(
        "outcome_created",
        outcome_type=outcome.outcome_type,
        confidence_level=outcome.confidence_level.value,
    )
    
    return outcome


def create_partial_answer_outcome(
    top_score: float,
    reason: str,
    has_contradiction: bool = False,
    triggered_rules: Optional[List[str]] = None,
) -> EpistemicOutcome:
    """Create a partial_answer outcome with medium confidence.
    
    Args:
        top_score: The top relevance score achieved
        reason: Specific reason for partial answer
        has_contradiction: Whether contradictions contributed
        triggered_rules: List of rules that contributed to this outcome
        
    Returns:
        EpistemicOutcome with partial_answer type
    """
    confidence = ConfidenceLevel.LOW if has_contradiction else ConfidenceLevel.MEDIUM
    
    if has_contradiction:
        explanation = (
            f"Conflicting sources detected. Top score is {top_score:.2f} "
            f"but evidence contains contradictions. "
            f"Unable to provide definitive answer."
        )
    else:
        explanation = (
            f"Some evidence found but confidence is moderate. "
            f"Top score: {top_score:.2f}. {reason}"
        )
    
    outcome = EpistemicOutcome(
        outcome_type="partial_answer",
        explanation_reason=explanation,
        confidence_level=confidence,
        triggered_rules=tuple(triggered_rules or []),
    )
    
    logger.info(
        "outcome_created",
        outcome_type=outcome.outcome_type,
        confidence_level=outcome.confidence_level.value,
        has_contradiction=has_contradiction,
    )
    
    return outcome


def create_insufficient_evidence_outcome(
    top_score: float,
    reason: str,
    triggered_rules: Optional[List[str]] = None,
) -> EpistemicOutcome:
    """Create an insufficient_evidence outcome with low confidence.
    
    Args:
        top_score: The top relevance score achieved
        reason: Specific reason for insufficient evidence
        triggered_rules: List of rules that contributed to this outcome
        
    Returns:
        EpistemicOutcome with insufficient_evidence type
    """
    outcome = EpistemicOutcome(
        outcome_type="insufficient_evidence",
        explanation_reason=(
            f"Unable to find sufficient evidence to answer confidently. "
            f"Top score: {top_score:.2f}. {reason}"
        ),
        confidence_level=ConfidenceLevel.LOW,
        triggered_rules=tuple(triggered_rules or []),
    )
    
    logger.info(
        "outcome_created",
        outcome_type=outcome.outcome_type,
        confidence_level=outcome.confidence_level.value,
    )
    
    return outcome


def create_clarification_needed_outcome(
    reason: str,
    triggered_rules: Optional[List[str]] = None,
) -> EpistemicOutcome:
    """Create a clarification_needed outcome.
    
    Args:
        reason: Specific reason why clarification is needed
        triggered_rules: List of rules that contributed to this outcome
        
    Returns:
        EpistemicOutcome with clarification_needed type
    """
    outcome = EpistemicOutcome(
        outcome_type="clarification_needed",
        explanation_reason=reason,
        confidence_level=ConfidenceLevel.LOW,
        triggered_rules=tuple(triggered_rules or ["POLICY_SKIP"]),
    )
    
    logger.info(
        "outcome_created",
        outcome_type=outcome.outcome_type,
        confidence_level=outcome.confidence_level.value,
    )
    
    return outcome
