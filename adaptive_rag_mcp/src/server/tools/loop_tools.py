"""MCP tool for Adaptive Iterative Retrieval.

This module provides the MCP tool interface for the iterative retrieval controller.
It uses the Phase 2/3 IterativeController with hypothetical reasoning and explicit outcomes.

Backward Compatibility:
- Tool name and schema unchanged
- Output format extended with outcome fields
"""

from typing import Any, Dict
from src.server.logging import get_logger

logger = get_logger(__name__)


def adaptive_retrieve(
    input_data: dict[str, Any],
    request_id: str | None = None
) -> dict[str, Any]:
    """Execute adaptive retrieval loop for a query.
    
    Uses the new IterativeController which includes:
    - Policy-driven retrieval decisions
    - Iterative retrieve → rerank → score loops
    - HyDE-style hypothetical reasoning (gated fallback)
    - Explicit outcome types
    
    Input:
        query: User query to answer (required)
        max_iterations: Maximum retry attempts (default 3)
        confidence_threshold: Score threshold to stop early (default 0.6)
        
    Output:
        results: List of retrieved results
        trace: List of iteration steps
        final_status: Success/reason/iterations
        outcome: Explicit outcome type and explanation
    """
    # Bind request_id to logger context so it appears in all loop logs
    if request_id:
        from src.server.logging import bind_logger_context
        bind_logger_context(request_id=request_id)
    
    query = input_data.get("query", "")
    max_iters = input_data.get("max_iterations", 3)
    threshold = input_data.get("confidence_threshold", 0.6)
    
    if not query:
        return {
            "results": [],
            "trace": [],
            "final_status": {
                "success": False,
                "reason": "Query is required",
                "iterations": 0
            },
            "outcome": {
                "type": "insufficient_evidence",
                "explanation": "No query provided.",
                "confidence_level": "low",
            }
        }
    
    try:
        # Use the new iterative controller
        from src.retrieval.iterative_controller import (
            IterativeController,
            OutcomeType,
        )
        
        controller = IterativeController(
            max_iterations=max_iters,
            confidence_top_threshold=threshold,
        )
        
        result = controller.retrieve(query)
        
        # Map outcome to simple success flag
        success = result.outcome_type in (
            OutcomeType.ANSWER_READY,
            OutcomeType.PARTIAL_ANSWER,
        )
        
        # Build response (maintaining backward compatibility)
        response = {
            # Original fields (backward compatible)
            "results": [
                {
                    "id": chunk.id,
                    "content": chunk.content,
                    "relevance_score": round(chunk.relevance_score, 4),
                    "original_rank": chunk.original_rank,
                    "metadata": chunk.metadata,
                }
                for chunk in result.final_chunks
            ],
            "trace": [
                {
                    "step": step.iteration,
                    "strategy": step.retrieval_mode,
                    "count": step.chunks_retrieved,
                    "score": round(step.top_score, 4),
                    "confident": step.is_confident,
                    "used_hypothetical": step.used_hypothetical,
                }
                for step in result.iteration_history
            ],
            "final_status": {
                "success": success,
                "reason": result.stop_reason.value,
                "iterations": result.iterations
            },
            # New Phase 3 fields (explicit outcomes with epistemic safety)
            "outcome": {
                "type": result.outcome_type.value,
                "explanation": result.explanation_reason,
                "confidence_level": result.confidence_level,
            },
            # Contradiction detection results (if any)
            "contradiction": result.contradiction_result.to_dict() if result.contradiction_result else None,
            # Additional context
            "policy": {
                "intent": result.policy_decision.intent,
                "retrieval_mode": result.policy_decision.retrieval_mode,
                "should_retrieve": result.policy_decision.should_retrieve,
            },
            "evidence": result.evidence_scores.to_dict() if result.evidence_scores else None,
        }
        
        logger.info(
            "adaptive_retrieve_complete",
            query_length=len(query),
            result_count=len(result.final_chunks),
            outcome=result.outcome_type.value,
            confidence_level=result.confidence_level,
            iterations=result.iterations,
            has_contradiction=result.contradiction_result.has_contradiction if result.contradiction_result else False,
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "adaptive_retrieve_error",
            query_length=len(query),
            error=str(e),
        )
        return {
            "results": [],
            "trace": [],
            "final_status": {
                "success": False,
                "reason": f"Error: {str(e)}",
                "iterations": 0
            },
            "outcome": {
                "type": "insufficient_evidence",
                "explanation": f"An error occurred: {str(e)}",
                "confidence_level": "low",
            }
        }


# Registry
LOOP_TOOLS = {
    "adaptive_retrieve": adaptive_retrieve
}
