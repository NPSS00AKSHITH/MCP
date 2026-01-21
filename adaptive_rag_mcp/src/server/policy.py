from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import json
import re

from src.server.logging import get_logger
from src.server.llm import get_llm_client
from src.retrieval.reranker import QualitySignals

logger = get_logger(__name__)

# Evidence Thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.7
GAP_THRESHOLD = 0.15
ENTROPY_THRESHOLD = 0.6 # Normalized entropy threshold

class QueryType(Enum):
    """Classification of the user query."""
    GENERAL_KNOWLEDGE = "general_knowledge"
    DOC_SPECIFIC = "doc_specific"
    MULTI_DOC = "multi_doc"
    COMPARISON = "comparison"
    AMBIGUOUS = "ambiguous"
    UNCERTAIN = "uncertain"

@dataclass
class RetrievalDecision:
    """Decision on how to handle the query."""
    should_retrieve: bool
    query_type: QueryType
    reason: str
    search_mode: str = "hybrid" # "dense", "sparse", "hybrid"
    max_k: int = 5
    filters: Optional[dict] = None
    suggested_rewrite: Optional[str] = None

class PolicyEngine:
    """Engine for making retrieval decisions."""

    def __init__(self):
        self.llm = get_llm_client()
        
    def decide(self, query: str) -> RetrievalDecision:
        """Make a retrieval decision for the given query."""
        if not query:
            return RetrievalDecision(
                should_retrieve=False,
                query_type=QueryType.UNCERTAIN,
                reason="Empty query"
            )

        query_type = self._classify_query_llm(query)
        decision = self._determine_strategy(query, query_type)
        
        logger.info(
            "policy_decision",
            query_type=decision.query_type.value,
            should_retrieve=decision.should_retrieve,
            mode=decision.search_mode,
            k=decision.max_k,
            reason=decision.reason
        )
        return decision

    def _classify_query_llm(self, query: str) -> QueryType:
        """Classify the query using LLM."""
        if not self.llm or not self.llm.model:
            logger.warning("llm_not_configured_for_policy", msg="Falling back to heuristics")
            return self._classify_query_heuristic(query)

        prompt = f"""Classify the following query into one of these categories:
- general_knowledge: Questions about world facts, definitions, or history not specific to internal documents.
- doc_specific: Questions asking about specific files, reports, or internal data.
- multi_doc: Questions requiring synthesis across multiple documents.
- comparison: Questions comparing two or more entities or documents.
- ambiguous: Vague queries where intent is unclear (e.g., "the bank").

Query: "{query}"

Output ONLY the category name.
"""
        try:
            response = self.llm.generate_text(prompt).strip().lower()
            # Clean response
            response = re.sub(r'[^a-z_]', '', response)
            
            # Map to Enum
            for qt in QueryType:
                if qt.value == response:
                    return qt
            
            logger.warning("llm_classification_unknown", response=response)
            return QueryType.DOC_SPECIFIC # Default safe
            
        except Exception as e:
            logger.error("llm_classification_failed", error=str(e))
            return self._classify_query_heuristic(query)

    def _classify_query_heuristic(self, query: str) -> QueryType:
        """Fallback heuristic classification."""
        query_lower = query.lower()
        
        doc_keywords = ["document", "file", "report", "paper", "memo", "contract", "summarize"]
        comparison_keywords = ["compare", "difference", "vs", "versus"]
        multi_doc_keywords = ["all", "across", "multiple"]
        general_keywords = ["what is", "define", "history of"]

        if any(kw in query_lower for kw in comparison_keywords):
            return QueryType.COMPARISON
        if any(kw in query_lower for kw in multi_doc_keywords) and any(kw in query_lower for kw in doc_keywords):
            return QueryType.MULTI_DOC
        if any(kw in query_lower for kw in doc_keywords):
            return QueryType.DOC_SPECIFIC
        if any(kw in query_lower for kw in general_keywords):
            return QueryType.GENERAL_KNOWLEDGE
            
        return QueryType.DOC_SPECIFIC

    def _determine_strategy(self, query: str, query_type: QueryType) -> RetrievalDecision:
        """Determine retrieval strategy based on type."""
        
        if query_type == QueryType.GENERAL_KNOWLEDGE:
            # Skip retrieval for general knowledge to avoid noise,
            # UNLESS user explicitly asks to check docs? 
            # For now, let's say we skip if it's purely general.
            return RetrievalDecision(
                should_retrieve=False,
                query_type=query_type,
                reason="General knowledge query - skipping retrieval.",
                search_mode="none",
                max_k=0
            )

        elif query_type == QueryType.AMBIGUOUS:
            return RetrievalDecision(
                should_retrieve=False,
                query_type=query_type,
                reason="Query is ambiguous - requesting clarification.",
                search_mode="none",
                max_k=0
            )

        elif query_type == QueryType.COMPARISON:
            return RetrievalDecision(
                should_retrieve=True,
                query_type=query_type,
                reason="Comparison query detected.",
                search_mode="hybrid",
                max_k=8
            )

        elif query_type == QueryType.MULTI_DOC:
            return RetrievalDecision(
                should_retrieve=True,
                query_type=query_type,
                reason="Multi-document query detected.",
                search_mode="dense",
                max_k=10
            )

        else: # DOC_SPECIFIC or UNCERTAIN
            return RetrievalDecision(
                should_retrieve=True,
                query_type=query_type,
                reason="Standard document query.",
                search_mode="hybrid",
                max_k=5
            )

    def evaluate_evidence(self, signals: QualitySignals) -> bool:
        """Determine if retrieved evidence is sufficient."""
        # Check high confidence criteria
        
        # 1. High absolute score
        if signals.top_score >= HIGH_CONFIDENCE_THRESHOLD:
            # 2. Reasonable gap (not a tie) OR very high score
            if signals.score_gap >= GAP_THRESHOLD or signals.top_score > 0.85:
                # 3. Entropy check (optional, but good for stability)
                if signals.evidence_entropy <= ENTROPY_THRESHOLD:
                    return True
        
        return False

    def determine_retry_strategy(self, history: List[Any], last_query: str) -> dict:
        """Determine parameters for the next retry attempt."""
        iteration = len(history) + 1 # Next iteration index
        
        # Strategy breakdown
        # Iteration 1 (Initial): Hybrid k=5
        # Iteration 2: Expand Query (if low score) or Switch Mode
        
        # Simple progressive strategy for now:
        if iteration == 1:
            # Should have been handled by initial decide(), but if we are here,
            # it means we are re-deciding/adapting.
            pass

        if iteration == 2:
            # Try Dense with higher K
            return {
                "search_mode": "dense", 
                "max_k": 10,
                "query": last_query # Could add " ... detailed" or similar
            }
            
        if iteration == 3:
            # Try Hybrid with very high K (Recall focus)
            return {
                "search_mode": "hybrid",
                "max_k": 15,
                "query": last_query
            }
            
        return None # Stop retrying
