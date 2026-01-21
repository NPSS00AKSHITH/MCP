"""Iterative adaptive retrieval controller.

Orchestrates the retrieve -> rerank -> evaluate -> retry loop.
"""

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from src.server.policy import PolicyEngine, RetrievalDecision, QueryType
from src.retrieval.hybrid import get_hybrid_retriever, SearchMode
from src.retrieval.reranker import get_reranker
from src.server.logging import get_logger

logger = get_logger(__name__)

@dataclass
class LoopStep:
    """Record of a single iteration in the loop."""
    step_number: int
    strategy: str  # e.g., "initial", "expanded_query", "dense_only"
    retrieved_count: int
    top_score: float
    is_confident: bool
    used_query: str

@dataclass
class LoopResult:
    """Final result of the iterative process."""
    final_docs: List[dict]
    steps: List[LoopStep]
    total_iterations: int
    success: bool
    reason: str

class RetrievalLoop:
    """Controller for iterative retrieval."""

    def __init__(self, max_iterations: int = 3, confidence_threshold: float = 0.6):
        self.policy = PolicyEngine()
        self.retriever = get_hybrid_retriever()
        # Use cross-encoder for high quality signal
        self.reranker = get_reranker("cross-encoder")
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold

    def run(self, query: str) -> LoopResult:
        """Execute the retrieval loop."""
        steps: List[LoopStep] = []
        current_query = query
        
        # Initial policy decision
        decision = self.policy.decide(query)
        if not decision.should_retrieve:
            return LoopResult([], steps, 0, True, "Policy decided to skip retrieval")

        # Initial parameters
        # Normalize mode string to Enum
        try:
            next_mode = SearchMode(decision.search_mode)
        except ValueError:
            next_mode = SearchMode.HYBRID
        next_k = decision.max_k

        # Start loop
        for i in range(1, self.max_iterations + 1):
            logger.info("loop_iteration_start", iteration=i, query=current_query)
            
            # 1. Set strategy for this iteration
            mode = next_mode
            k = next_k
            
            logger.info("retrieval_step_start", iteration=i, mode=mode.value, k=k)
             
            # 2. Retrieve
            retrieved_chunks = self.retriever.search(current_query, k=k, mode=mode)
            
            # Convert chunks to dicts for reranker
            docs_for_rerank = [
                {"id": c.chunk_id, "content": c.content, "metadata": c.metadata}
                for c in retrieved_chunks
            ]
            
            # 3. Rerank & Evaluate
            # Even if 0 chunks, we should run rerank logic to get a cleaner empty QualitySignals object if needed,
            # or handle it explicitly. 
            if not docs_for_rerank:
                reranked, quality = [], None # Will handle empty below
            else:
                reranked, quality = self.reranker.rerank(query, docs_for_rerank, top_k=5)
            
            if quality:
                top_score = quality.top_score
                # Use Policy to evaluate evidence
                is_confident = self.policy.evaluate_evidence(quality)
            else:
                top_score = 0.0
                is_confident = False
            
            step = LoopStep(
                step_number=i,
                strategy=f"{mode.value} (k={k})",
                retrieved_count=len(retrieved_chunks),
                top_score=top_score,
                is_confident=is_confident,
                used_query=current_query
            )
            steps.append(step)
            
            context_size = sum(len(c.content) for c in retrieved_chunks)
            
            logger.info(
                "loop_iteration_eval",
                iteration=i,
                top_score=top_score,
                confident=is_confident,
                context_size=context_size
            )

            # 4. Check stopping condition
            if is_confident:
                return LoopResult(
                    final_docs=[
                        {"id": r.id, "content": r.content, "score": r.relevance_score, "metadata": r.metadata} 
                        for r in reranked
                    ],
                    steps=steps,
                    total_iterations=i,
                    success=True,
                    reason="High confidence achieved"
                )
            
            # 5. Adapt for next iteration (if not last)
            if i < self.max_iterations:
                # Ask policy for next strategy
                retry_params = self.policy.determine_retry_strategy(steps, current_query)
                
                if retry_params:
                    # Update params for next loop
                    try:
                        next_mode = SearchMode(retry_params.get("search_mode", "hybrid"))
                    except ValueError:
                        next_mode = SearchMode.HYBRID
                        
                    next_k = retry_params.get("max_k", k + 5)
                    current_query = retry_params.get("query", current_query)
                    
                    logger.info("adapting_strategy", next_mode=next_mode.value, next_k=next_k)
                else:
                    logger.info("policy_stopped_retries")
                    break

        # Fallback if loop finishes without high confidence
        # Return best results from last iteration (if any)
        final_results = []
        if 'reranked' in locals() and reranked:
             final_results = [
                {"id": r.id, "content": r.content, "score": r.relevance_score, "metadata": r.metadata} 
                for r in reranked
            ]
        
        return LoopResult(
            final_docs=final_results,
            steps=steps,
            total_iterations=len(steps),
            success=False,
            reason="Max iterations reached without high confidence"
        )
