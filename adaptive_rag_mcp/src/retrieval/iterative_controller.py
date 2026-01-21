"""Iterative Retrieval Controller for orchestrating retrieve → rerank → score loops.

This module provides a controller that orchestrates the full retrieval flow:
1. Policy decision (should we retrieve? what mode?)
2. Retrieval (dense/sparse/hybrid)
3. Reranking (cross-encoder)
4. Evidence scoring (confidence check)
5. Hypothetical reasoning (HyDE fallback if low confidence)
6. Stop or retry based on confidence

Uses Phase 1 modules:
- AdaptivePolicyEngine (src/policy/engine.py)
- compute_evidence_scores (src/retrieval/evidence_scoring.py)

Design Constraints:
- Does NOT modify retrieval internals
- Bounded loop (respects max_iterations)
- Structured logging for each iteration
- Hypothetical text is NEVER returned in outputs

Example Usage:
-------------
>>> from src.retrieval.iterative_controller import IterativeController
>>> controller = IterativeController()
>>> result = controller.retrieve("Compare the Q1 and Q2 reports")
>>> print(result.final_chunks)      # Reranked chunks
>>> print(result.outcome_type)      # "answer_ready", "partial_answer", etc.
>>> print(result.explanation_reason) # Why this outcome

Example Execution Traces:
-------------------------

[Trace 1: answer_ready]
Query: "What is the budget allocation for Q3?"
Iteration 1: hybrid, k=5, top_score=0.85, confident=True
Outcome: answer_ready
Reason: "High confidence evidence found with top_score=0.85"

[Trace 2: partial_answer]
Query: "Summarize the project milestones"
Iteration 1: hybrid, k=5, top_score=0.55, confident=False
Iteration 2: dense, k=8, top_score=0.62, confident=True (after HyDE)
Outcome: partial_answer
Reason: "Found relevant documents via hypothetical expansion"

[Trace 3: insufficient_evidence]
Query: "What is the CEO's favorite color?"
Iteration 1: hybrid, k=5, top_score=0.25, confident=False
Iteration 2: HyDE, k=8, top_score=0.30, confident=False
Iteration 3: hybrid, k=12, top_score=0.28, confident=False
Outcome: insufficient_evidence
Reason: "Max iterations reached with low confidence (top_score=0.28)"

[Trace 4: clarification_needed]
Query: "the bank"
Policy: ambiguous query, skip retrieval
Outcome: clarification_needed
Reason: "Query is ambiguous - clarification may be needed."
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional
from enum import Enum

from src.server.logging import get_logger
from src.policy.engine import AdaptivePolicyEngine, PolicyDecision
from src.retrieval.evidence_scoring import (
    compute_evidence_scores,
    EvidenceScores,
    DEFAULT_TOP_SCORE_THRESHOLD,
    DEFAULT_ENTROPY_THRESHOLD,
    detect_contradictions,
    ContradictionResult,
)
from src.retrieval.epistemic_safety import (
    ConfidenceLevel,
    StoppingRuleEvaluator,
    HIGH_CONFIDENCE_THRESHOLD,
    SCORE_GAP_THRESHOLD,
    ENTROPY_THRESHOLD,
)
from src.retrieval.hybrid import (
    HybridRetriever,
    HybridSearchResult,
    SearchMode,
    get_hybrid_retriever,
)
from src.retrieval.reranker import (
    get_reranker,
    RerankResult,
    QualitySignals,
)

logger = get_logger(__name__)


# =============================================================================
# CONFIGURABLE CONSTANTS
# =============================================================================

# Default maximum iterations for the retrieval loop
DEFAULT_MAX_ITERATIONS: int = 3

# Default number of results to retrieve per iteration
DEFAULT_K: int = 5

# K multiplier for retry attempts (fetch more on retry)
RETRY_K_MULTIPLIER: float = 1.5

# Enable/disable hypothetical reasoning
ENABLE_HYPOTHETICAL_REASONING: bool = True

# Hypothetical reasoning prompt template (HyDE-style)
HYDE_PROMPT_TEMPLATE = """Given the following question, write a short paragraph that would answer this question if it existed in a document. Write only the hypothetical answer, no preamble or explanation. Keep it factual and concise (2-3 sentences).

Question: {query}

Hypothetical answer:"""


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class StopReason(Enum):
    """Reasons for stopping the retrieval loop."""
    HIGH_CONFIDENCE = "high_confidence"      # Evidence meets confidence threshold
    MAX_ITERATIONS = "max_iterations"        # Reached iteration limit
    POLICY_SKIP = "policy_skip"              # Policy decided not to retrieve
    NO_RESULTS = "no_results"                # Retrieval returned no results
    HYPOTHETICAL_SUCCESS = "hypothetical_success"  # HyDE improved results
    CONTRADICTION_DETECTED = "contradiction_detected"  # Conflicting sources found
    ERROR = "error"                          # An error occurred


class OutcomeType(Enum):
    """Explicit outcome types for MCP responses."""
    ANSWER_READY = "answer_ready"              # High confidence, ready to answer
    PARTIAL_ANSWER = "partial_answer"          # Some evidence found, may be incomplete
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"  # Not enough evidence to answer
    CLARIFICATION_NEEDED = "clarification_needed"    # Query is ambiguous


@dataclass
class IterationStep:
    """Record of a single iteration in the retrieval loop."""
    iteration: int
    retrieval_mode: str
    k: int
    chunks_retrieved: int
    chunks_after_rerank: int
    top_score: float
    score_gap: float
    score_entropy: float
    is_confident: bool
    used_hypothetical: bool = False  # Flag if HyDE was used this iteration
    

@dataclass
class RetrievalResult:
    """Final result from the iterative retrieval controller."""
    query: str
    final_chunks: List[RerankResult]
    iterations: int
    stop_reason: StopReason
    policy_decision: PolicyDecision
    evidence_scores: Optional[EvidenceScores]
    outcome_type: OutcomeType
    explanation_reason: str
    confidence_level: str = "low"  # high, medium, or low
    contradiction_result: Optional[ContradictionResult] = None
    iteration_history: List[IterationStep] = field(default_factory=list)
    error_message: Optional[str] = None
    # Note: hypothetical_query is intentionally NOT stored to prevent exposure
    
    def to_dict(self) -> dict:
        """Convert result to dictionary format."""
        return {
            "query": self.query,
            "chunk_count": len(self.final_chunks),
            "iterations": self.iterations,
            "stop_reason": self.stop_reason.value,
            "intent": self.policy_decision.intent,
            "retrieval_mode": self.policy_decision.retrieval_mode,
            "evidence": self.evidence_scores.to_dict() if self.evidence_scores else None,
            "outcome_type": self.outcome_type.value,
            "explanation_reason": self.explanation_reason,
            "confidence_level": self.confidence_level,
            "contradiction": self.contradiction_result.to_dict() if self.contradiction_result else None,
            "error": self.error_message,
        }


# =============================================================================
# HYPOTHETICAL DOCUMENT GENERATOR
# =============================================================================

class HypotheticalGenerator:
    """Generator for hypothetical document embeddings (HyDE-style).
    
    HyDE (Hypothetical Document Embeddings) generates a hypothetical answer
    to a query, then uses that hypothetical answer for retrieval instead of
    the original query. This can improve recall for questions that don't
    have exact keyword matches in the corpus.
    
    IMPORTANT: The hypothetical text is NEVER returned in outputs.
    It is used internally only for improved retrieval.
    """
    
    def __init__(self):
        self._llm = None
    
    @property
    def llm(self):
        """Lazy-load the LLM client."""
        if self._llm is None:
            from src.server.llm import get_llm_client
            self._llm = get_llm_client()
        return self._llm
    
    def generate_hypothetical_query(self, original_query: str) -> Optional[str]:
        """Generate a hypothetical document that would answer the query.
        
        Args:
            original_query: The original user query.
            
        Returns:
            A hypothetical answer text, or None if generation fails.
            This text is for INTERNAL USE ONLY and must not be exposed.
        """
        if not self.llm or not self.llm.model:
            logger.warning("hyde_llm_unavailable", reason="LLM not configured")
            return None
        
        try:
            prompt = HYDE_PROMPT_TEMPLATE.format(query=original_query)
            hypothetical = self.llm.generate_text(prompt)
            
            # Clean up the response
            hypothetical = hypothetical.strip()
            
            # Sanity check - should be reasonable length
            if len(hypothetical) < 10 or len(hypothetical) > 1000:
                logger.warning(
                    "hyde_response_invalid",
                    length=len(hypothetical),
                    reason="Response too short or too long",
                )
                return None
            
            logger.info(
                "hyde_generated",
                original_query_length=len(original_query),
                hypothetical_length=len(hypothetical),
            )
            
            return hypothetical
            
        except Exception as e:
            logger.error("hyde_generation_failed", error=str(e))
            return None


# =============================================================================
# OUTCOME DETERMINATION
# =============================================================================

def determine_outcome(
    stop_reason: StopReason,
    evidence: Optional[EvidenceScores],
    policy: PolicyDecision,
    used_hypothetical: bool,
    contradiction_result: Optional[ContradictionResult] = None,
) -> tuple[OutcomeType, str, str]:
    """Determine the explicit outcome type based on retrieval results.
    
    Args:
        stop_reason: Why the retrieval loop stopped.
        evidence: Final evidence scores (if any).
        policy: Policy decision for the query.
        used_hypothetical: Whether HyDE was used.
        contradiction_result: Result of contradiction detection (if any).
        
    Returns:
        Tuple of (OutcomeType, explanation_reason, confidence_level).
    
    IMPORTANT RULE:
    If the system cannot clearly explain WHY an answer is correct,
    it must NOT return answer_ready.
    
    Example Log Messages Per Outcome:
    ---------------------------------
    answer_ready:
        "outcome_determined", outcome="answer_ready", confidence="high"
        reason="High confidence evidence found with top_score=0.85"
    
    partial_answer:
        "outcome_determined", outcome="partial_answer", confidence="medium"
        reason="Found relevant documents via hypothetical expansion"
    
    insufficient_evidence:
        "outcome_determined", outcome="insufficient_evidence", confidence="low"
        reason="Max iterations reached with low confidence (top_score=0.28)"
    
    clarification_needed:
        "outcome_determined", outcome="clarification_needed", confidence="low"
        reason="Query is ambiguous - clarification may be needed."
    """
    
    # Check for contradictions first - this downgrades confidence
    has_contradiction = contradiction_result and contradiction_result.has_contradiction
    
    # Policy skip cases
    if stop_reason == StopReason.POLICY_SKIP:
        intent = policy.intent
        if intent == "ambiguous":
            return (
                OutcomeType.CLARIFICATION_NEEDED,
                policy.decision_reason,
                "low",
            )
        elif intent == "general_knowledge":
            return (
                OutcomeType.ANSWER_READY,
                "General knowledge query - can answer without retrieval.",
                "high",
            )
        else:
            return (
                OutcomeType.CLARIFICATION_NEEDED,
                policy.decision_reason,
                "low",
            )
    
    # No results
    if stop_reason == StopReason.NO_RESULTS:
        return (
            OutcomeType.INSUFFICIENT_EVIDENCE,
            "No documents found matching the query.",
            "low",
        )
    
    # Error case
    if stop_reason == StopReason.ERROR:
        return (
            OutcomeType.INSUFFICIENT_EVIDENCE,
            "An error occurred during retrieval.",
            "low",
        )
    
    # Evidence-based outcomes
    if evidence is None:
        return (
            OutcomeType.INSUFFICIENT_EVIDENCE,
            "No evidence available.",
            "low",
        )
    
    top_score = evidence.top_score
    score_gap = evidence.score_gap
    
    # Contradiction detected - never return answer_ready
    if stop_reason == StopReason.CONTRADICTION_DETECTED or has_contradiction:
        contradiction_explanation = ""
        if contradiction_result:
            contradiction_explanation = f" {contradiction_result.explanation}"
        return (
            OutcomeType.PARTIAL_ANSWER,
            f"Conflicting sources detected (top_score={top_score:.2f}).{contradiction_explanation}",
            "low",
        )
    
    # Determine confidence level based on scores
    if top_score >= HIGH_CONFIDENCE_THRESHOLD and score_gap >= SCORE_GAP_THRESHOLD:
        confidence_level = "high"
    elif top_score >= 0.5:
        confidence_level = "medium"
    else:
        confidence_level = "low"
    
    # High confidence (normal retrieval or HyDE success)
    if stop_reason == StopReason.HIGH_CONFIDENCE:
        # Only return answer_ready if we can explain WHY it's correct
        if confidence_level == "high":
            return (
                OutcomeType.ANSWER_READY,
                f"High confidence evidence found. Top score: {top_score:.2f}, score gap: {score_gap:.2f}. No contradictions detected.",
                "high",
            )
        else:
            # Downgrade to partial_answer if confidence not truly high
            return (
                OutcomeType.PARTIAL_ANSWER,
                f"Evidence found but confidence is only {confidence_level} (top_score={top_score:.2f}).",
                confidence_level,
            )
    
    # HyDE success (found better results via hypothetical)
    if stop_reason == StopReason.HYPOTHETICAL_SUCCESS:
        return (
            OutcomeType.PARTIAL_ANSWER,
            f"Found relevant documents via hypothetical expansion (top_score={top_score:.2f}).",
            confidence_level,
        )
    
    # Max iterations reached - check quality
    if stop_reason == StopReason.MAX_ITERATIONS:
        if top_score >= 0.5:
            return (
                OutcomeType.PARTIAL_ANSWER,
                f"Some evidence found but confidence is {confidence_level} (top_score={top_score:.2f}).",
                confidence_level,
            )
        else:
            return (
                OutcomeType.INSUFFICIENT_EVIDENCE,
                f"Max iterations reached with low confidence (top_score={top_score:.2f}).",
                "low",
            )
    
    # Fallback
    return (
        OutcomeType.INSUFFICIENT_EVIDENCE,
        "Unable to determine outcome.",
        "low",
    )


# =============================================================================
# ITERATIVE CONTROLLER
# =============================================================================

class IterativeController:
    """Controller for iterative retrieve → rerank → score loops with HyDE fallback.
    
    This controller orchestrates the full retrieval flow:
    1. Policy decision (should we retrieve? what mode?)
    2. Retrieval (dense/sparse/hybrid via HybridRetriever)
    3. Reranking (cross-encoder)
    4. Evidence scoring (confidence check)
    5. HyDE fallback (if low confidence and iterations remain)
    6. Stop or retry based on confidence
    
    Example:
        >>> controller = IterativeController()
        >>> result = controller.retrieve("Find the budget report")
        >>> result.outcome_type
        OutcomeType.ANSWER_READY
    """
    
    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        policy_engine: Optional[AdaptivePolicyEngine] = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        default_k: int = DEFAULT_K,
        confidence_top_threshold: float = DEFAULT_TOP_SCORE_THRESHOLD,
        confidence_entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
        enable_hypothetical: bool = ENABLE_HYPOTHETICAL_REASONING,
    ):
        """Initialize the iterative controller.
        
        Args:
            retriever: HybridRetriever instance. Uses global if None.
            policy_engine: AdaptivePolicyEngine instance. Creates new if None.
            max_iterations: Maximum iterations before stopping.
            default_k: Default number of results to retrieve.
            confidence_top_threshold: Minimum top score for high confidence.
            confidence_entropy_threshold: Maximum entropy for high confidence.
            enable_hypothetical: Whether to enable HyDE fallback.
        """
        self._retriever = retriever
        self._policy_engine = policy_engine or AdaptivePolicyEngine()
        self.max_iterations = max_iterations
        self.default_k = default_k
        self.confidence_top_threshold = confidence_top_threshold
        self.confidence_entropy_threshold = confidence_entropy_threshold
        self.enable_hypothetical = enable_hypothetical
        self._reranker = None
        self._hyde_generator = None
    
    @property
    def retriever(self) -> HybridRetriever:
        """Lazy-load the hybrid retriever."""
        if self._retriever is None:
            self._retriever = get_hybrid_retriever()
        return self._retriever
    
    @property
    def reranker(self):
        """Lazy-load the reranker."""
        if self._reranker is None:
            self._reranker = get_reranker()
        return self._reranker
    
    @property
    def hyde_generator(self) -> HypotheticalGenerator:
        """Lazy-load the HyDE generator."""
        if self._hyde_generator is None:
            self._hyde_generator = HypotheticalGenerator()
        return self._hyde_generator
    
    def _map_mode_to_search_mode(self, mode: str) -> SearchMode:
        """Map policy retrieval mode string to SearchMode enum."""
        mode_map = {
            "dense": SearchMode.DENSE,
            "sparse": SearchMode.SPARSE,
            "hybrid": SearchMode.HYBRID,
        }
        return mode_map.get(mode, SearchMode.HYBRID)
    
    def _is_confident(self, evidence: EvidenceScores) -> bool:
        """Check if evidence meets confidence thresholds."""
        return (
            evidence.top_score >= self.confidence_top_threshold and
            evidence.score_entropy <= self.confidence_entropy_threshold and
            not evidence.low_confidence_flag
        )
    
    def _get_retry_params(self, iteration: int, last_mode: str, last_k: int) -> tuple:
        """Get adjusted parameters for retry attempt.
        
        Strategy:
        - Iteration 2: Switch to dense, increase k
        - Iteration 3: Switch to hybrid with higher k
        
        Args:
            iteration: Current iteration number (1-indexed).
            last_mode: Mode used in last iteration.
            last_k: K used in last iteration.
            
        Returns:
            Tuple of (new_mode, new_k).
        """
        new_k = int(last_k * RETRY_K_MULTIPLIER)
        
        if iteration == 2:
            # Try dense mode with more results
            return ("dense", new_k)
        elif iteration == 3:
            # Try hybrid with even more results
            return ("hybrid", new_k)
        else:
            # Continue with same mode, more results
            return (last_mode, new_k)
    
    def _should_try_hypothetical(
        self,
        iteration: int,
        max_iters: int,
        evidence: EvidenceScores,
        already_used: bool,
    ) -> bool:
        """Determine if we should try HyDE fallback.
        
        Trigger conditions:
        - HyDE is enabled
        - Evidence is low confidence
        - Iterations remain
        - Haven't already used HyDE this session
        
        Args:
            iteration: Current iteration number.
            max_iters: Maximum iterations allowed.
            evidence: Current evidence scores.
            already_used: Whether HyDE was already used.
            
        Returns:
            True if should try HyDE.
        """
        if not self.enable_hypothetical:
            return False
        
        if already_used:
            return False
        
        if iteration >= max_iters:
            return False
        
        if self._is_confident(evidence):
            return False
        
        # Low confidence - try HyDE
        return True
    
    def _execute_retrieval_iteration(
        self,
        query: str,
        mode: str,
        k: int,
        iteration: int,
        used_hypothetical: bool = False,
    ) -> tuple[List[RerankResult], EvidenceScores, List[HybridSearchResult]]:
        """Execute a single retrieval → rerank → score iteration.
        
        Args:
            query: The search query (may be original or hypothetical).
            mode: Retrieval mode.
            k: Number of results.
            iteration: Iteration number.
            used_hypothetical: Whether this uses a hypothetical query.
            
        Returns:
            Tuple of (reranked_results, evidence_scores, raw_results).
        """
        search_mode = self._map_mode_to_search_mode(mode)
        raw_results = self.retriever.search(
            query=query,
            k=k,
            mode=search_mode,
        )
        
        if not raw_results:
            return [], compute_evidence_scores([]), []
        
        # Convert to reranker input format
        # IMPORTANT: Use ORIGINAL query for reranking, not hypothetical
        # The hypothetical is only for retrieval
        docs_for_rerank = [
            {
                "id": r.chunk_id,
                "content": r.content,
                "metadata": r.metadata,
                "score": r.score,
            }
            for r in raw_results
        ]
        
        reranked, _ = self.reranker.rerank(
            query=query,  # Use the query provided (original for rerank)
            documents=docs_for_rerank,
        )
        
        scores = [r.relevance_score for r in reranked]
        evidence = compute_evidence_scores(scores)
        
        return reranked, evidence, raw_results
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        max_iterations: Optional[int] = None,
    ) -> RetrievalResult:
        """Execute iterative retrieval for a query.
        
        This is the main entry point. It:
        1. Gets policy decision
        2. Runs bounded retrieve → rerank → score loop
        3. Optionally tries HyDE if low confidence
        4. Returns results with explicit outcome type
        
        Args:
            query: The search query.
            k: Number of results to retrieve (overrides default).
            max_iterations: Maximum iterations (overrides instance default).
            
        Returns:
            RetrievalResult with final chunks, outcome type, and metadata.
        """
        k = k or self.default_k
        max_iters = max_iterations or self.max_iterations
        
        logger.info(
            "iterative_retrieval_start",
            query_length=len(query),
            k=k,
            max_iterations=max_iters,
            enable_hypothetical=self.enable_hypothetical,
        )
        
        # Step 1: Get policy decision
        policy = self._policy_engine.decide(query)
        
        logger.info(
            "policy_decision",
            should_retrieve=policy.should_retrieve,
            mode=policy.retrieval_mode,
            intent=policy.intent,
            reason=policy.decision_reason,
        )
        
        # Check if policy says skip retrieval
        if not policy.should_retrieve:
            logger.info(
                "retrieval_skipped",
                reason=policy.decision_reason,
            )
            outcome_type, explanation, confidence_level = determine_outcome(
                StopReason.POLICY_SKIP, None, policy, False
            )
            
            logger.info(
                "outcome_determined",
                outcome=outcome_type.value,
                confidence_level=confidence_level,
                reason=explanation,
            )
            
            return RetrievalResult(
                query=query,
                final_chunks=[],
                iterations=0,
                stop_reason=StopReason.POLICY_SKIP,
                policy_decision=policy,
                evidence_scores=None,
                outcome_type=outcome_type,
                explanation_reason=explanation,
                confidence_level=confidence_level,
            )
        
        # Step 2: Iterative retrieval loop
        iteration_history: List[IterationStep] = []
        current_mode = policy.retrieval_mode
        current_k = k
        final_chunks: List[RerankResult] = []
        final_evidence: Optional[EvidenceScores] = None
        final_contradiction: Optional[ContradictionResult] = None
        stop_reason = StopReason.MAX_ITERATIONS
        hyde_used_this_session = False
        
        for iteration in range(1, max_iters + 1):
            logger.info(
                "iteration_start",
                iteration=iteration,
                mode=current_mode,
                k=current_k,
            )
            
            try:
                # Execute retrieval iteration
                reranked, evidence, raw_results = self._execute_retrieval_iteration(
                    query=query,
                    mode=current_mode,
                    k=current_k,
                    iteration=iteration,
                )
                
                logger.info(
                    "retrieval_complete",
                    iteration=iteration,
                    chunks_retrieved=len(raw_results),
                    chunks_reranked=len(reranked),
                    top_score=evidence.top_score,
                    score_gap=evidence.score_gap,
                    entropy=evidence.score_entropy,
                )
                
                # Handle no results
                if not raw_results:
                    logger.warning(
                        "no_results_retrieved",
                        iteration=iteration,
                    )
                    stop_reason = StopReason.NO_RESULTS
                    break
                
                # Record iteration
                step = IterationStep(
                    iteration=iteration,
                    retrieval_mode=current_mode,
                    k=current_k,
                    chunks_retrieved=len(raw_results),
                    chunks_after_rerank=len(reranked),
                    top_score=evidence.top_score,
                    score_gap=evidence.score_gap,
                    score_entropy=evidence.score_entropy,
                    is_confident=self._is_confident(evidence),
                    used_hypothetical=False,
                )
                iteration_history.append(step)
                
                # Update final results
                final_chunks = reranked
                final_evidence = evidence
                
                # Check for contradictions among top chunks
                chunks_for_contradiction = [
                    {
                        "id": r.id,
                        "content": r.content,
                        "score": r.relevance_score,
                    }
                    for r in reranked
                ]
                contradiction_result = detect_contradictions(chunks_for_contradiction)
                
                if contradiction_result.has_contradiction:
                    logger.warning(
                        "contradiction_detected",
                        iteration=iteration,
                        conflicting_chunks=contradiction_result.conflicting_chunks,
                        explanation=contradiction_result.explanation,
                    )
                    final_contradiction = contradiction_result
                    stop_reason = StopReason.CONTRADICTION_DETECTED
                    break
                
                # Check stopping condition - high confidence
                if self._is_confident(evidence):
                    logger.info(
                        "stopping_condition_triggered",
                        iteration=iteration,
                        rule="HIGH_CONFIDENCE",
                        top_score=evidence.top_score,
                        entropy=evidence.score_entropy,
                    )
                    stop_reason = StopReason.HIGH_CONFIDENCE
                    break
                
                # Check if we should try HyDE fallback
                if self._should_try_hypothetical(
                    iteration, max_iters, evidence, hyde_used_this_session
                ):
                    logger.info(
                        "hypothetical_reasoning_triggered",
                        iteration=iteration,
                        top_score=evidence.top_score,
                        reason="Low confidence evidence, trying HyDE",
                    )
                    
                    # Generate hypothetical query
                    hypothetical_query = self.hyde_generator.generate_hypothetical_query(query)
                    
                    if hypothetical_query:
                        hyde_used_this_session = True
                        
                        # Execute retrieval with hypothetical query
                        # But rerank against ORIGINAL query
                        search_mode = self._map_mode_to_search_mode("dense")
                        hyde_k = int(current_k * RETRY_K_MULTIPLIER)
                        
                        hyde_raw_results = self.retriever.search(
                            query=hypothetical_query,  # Search with hypothetical
                            k=hyde_k,
                            mode=search_mode,
                        )
                        
                        if hyde_raw_results:
                            # Rerank against ORIGINAL query (not hypothetical)
                            docs_for_rerank = [
                                {
                                    "id": r.chunk_id,
                                    "content": r.content,
                                    "metadata": r.metadata,
                                    "score": r.score,
                                }
                                for r in hyde_raw_results
                            ]
                            
                            hyde_reranked, _ = self.reranker.rerank(
                                query=query,  # Original query for reranking
                                documents=docs_for_rerank,
                            )
                            
                            hyde_scores = [r.relevance_score for r in hyde_reranked]
                            hyde_evidence = compute_evidence_scores(hyde_scores)
                            
                            logger.info(
                                "hypothetical_retrieval_complete",
                                iteration=iteration,
                                chunks_retrieved=len(hyde_raw_results),
                                chunks_reranked=len(hyde_reranked),
                                top_score=hyde_evidence.top_score,
                            )
                            
                            # Record HyDE iteration
                            hyde_step = IterationStep(
                                iteration=iteration,
                                retrieval_mode="dense_hyde",
                                k=hyde_k,
                                chunks_retrieved=len(hyde_raw_results),
                                chunks_after_rerank=len(hyde_reranked),
                                top_score=hyde_evidence.top_score,
                                score_gap=hyde_evidence.score_gap,
                                score_entropy=hyde_evidence.score_entropy,
                                is_confident=self._is_confident(hyde_evidence),
                                used_hypothetical=True,
                            )
                            
                            # If HyDE improved results, use them
                            if hyde_evidence.top_score > evidence.top_score:
                                logger.info(
                                    "hypothetical_improvement",
                                    original_score=evidence.top_score,
                                    hyde_score=hyde_evidence.top_score,
                                )
                                final_chunks = hyde_reranked
                                final_evidence = hyde_evidence
                                iteration_history.append(hyde_step)
                                
                                if self._is_confident(hyde_evidence):
                                    stop_reason = StopReason.HYPOTHETICAL_SUCCESS
                                    break
                
                # Not confident - prepare for retry if iterations remain
                if iteration < max_iters:
                    current_mode, current_k = self._get_retry_params(
                        iteration + 1,
                        current_mode,
                        current_k,
                    )
                    logger.info(
                        "low_confidence_retry",
                        next_iteration=iteration + 1,
                        next_mode=current_mode,
                        next_k=current_k,
                    )
                else:
                    logger.info(
                        "max_iterations_reached",
                        iteration=iteration,
                    )
                    stop_reason = StopReason.MAX_ITERATIONS
                    
            except Exception as e:
                logger.error(
                    "iteration_error",
                    iteration=iteration,
                    error=str(e),
                )
                stop_reason = StopReason.ERROR
                outcome_type, explanation, confidence_level = determine_outcome(
                    stop_reason, final_evidence, policy, hyde_used_this_session, final_contradiction
                )
                return RetrievalResult(
                    query=query,
                    final_chunks=final_chunks,
                    iterations=iteration,
                    stop_reason=stop_reason,
                    policy_decision=policy,
                    evidence_scores=final_evidence,
                    outcome_type=outcome_type,
                    explanation_reason=explanation,
                    confidence_level=confidence_level,
                    contradiction_result=final_contradiction,
                    iteration_history=iteration_history,
                    error_message=str(e),
                )
        
        # Determine final outcome
        outcome_type, explanation, confidence_level = determine_outcome(
            stop_reason, final_evidence, policy, hyde_used_this_session, final_contradiction
        )
        
        # Log final result with explainability
        logger.info(
            "iterative_retrieval_complete",
            query_length=len(query),
            iterations=len(iteration_history),
            stop_reason=stop_reason.value,
            final_chunk_count=len(final_chunks),
            final_top_score=final_evidence.top_score if final_evidence else 0.0,
            outcome=outcome_type.value,
            confidence_level=confidence_level,
            used_hypothetical=hyde_used_this_session,
            has_contradiction=final_contradiction.has_contradiction if final_contradiction else False,
        )
        
        logger.info(
            "outcome_selected",
            outcome_type=outcome_type.value,
            confidence_level=confidence_level,
            explanation=explanation,
            why="System chose this outcome based on evidence scores and stopping rules.",
        )
        
        return RetrievalResult(
            query=query,
            final_chunks=final_chunks,
            iterations=len(iteration_history),
            stop_reason=stop_reason,
            policy_decision=policy,
            evidence_scores=final_evidence,
            outcome_type=outcome_type,
            explanation_reason=explanation,
            confidence_level=confidence_level,
            contradiction_result=final_contradiction,
            iteration_history=iteration_history,
        )


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def iterative_retrieve(
    query: str,
    k: int = DEFAULT_K,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    enable_hypothetical: bool = ENABLE_HYPOTHETICAL_REASONING,
) -> RetrievalResult:
    """Convenience function for iterative retrieval.
    
    Args:
        query: The search query.
        k: Number of results to retrieve.
        max_iterations: Maximum iterations.
        enable_hypothetical: Whether to enable HyDE fallback.
        
    Returns:
        RetrievalResult with final chunks, outcome type, and metadata.
    
    Example:
        >>> from src.retrieval.iterative_controller import iterative_retrieve
        >>> result = iterative_retrieve("Find the Q3 budget report")
        >>> print(f"Outcome: {result.outcome_type.value}")
        >>> print(f"Reason: {result.explanation_reason}")
    """
    controller = IterativeController(
        max_iterations=max_iterations,
        default_k=k,
        enable_hypothetical=enable_hypothetical,
    )
    return controller.retrieve(query)
