"""Adaptive Policy Engine for query intent classification and retrieval decisions.

This module provides a lightweight, rule-based policy engine that classifies
query intent and decides whether retrieval is needed, what mode to use, and
how many iterations to allow.

This is a STANDALONE module that does not modify existing PolicyEngine in
src/server/policy.py. It can be used independently or alongside existing logic.

Design Constraints:
- Rule-based or lightweight heuristics only (no LLM calls)
- No retriever calls
- No side effects
- Deterministic outputs

Example Usage:
-------------
>>> from src.policy.engine import AdaptivePolicyEngine
>>> engine = AdaptivePolicyEngine()
>>> decision = engine.decide("Compare the Q1 and Q2 sales reports")
>>> print(decision.should_retrieve)
True
>>> print(decision.retrieval_mode)
'hybrid'
>>> print(decision.intent)
'comparison'

Unit-Test Examples:
-------------------
>>> # Test 1: General knowledge query should NOT retrieve
>>> engine = AdaptivePolicyEngine()
>>> d = engine.decide("What is the capital of France?")
>>> assert d.should_retrieve == False
>>> assert d.intent == "general_knowledge"

>>> # Test 2: Document-specific query SHOULD retrieve
>>> d = engine.decide("Summarize the Q3 financial report")
>>> assert d.should_retrieve == True
>>> assert d.intent == "doc_specific"

>>> # Test 3: Comparison query should use hybrid mode
>>> d = engine.decide("Compare revenue between 2023 and 2024")
>>> assert d.should_retrieve == True
>>> assert d.retrieval_mode == "hybrid"
>>> assert d.intent == "comparison"

>>> # Test 4: Multi-document queries should use dense mode with higher iterations
>>> d = engine.decide("Find all mentions of budget across all documents")
>>> assert d.should_retrieve == True
>>> assert d.intent == "multi_document"
>>> assert d.retrieval_mode == "dense"
"""

from dataclasses import dataclass
from typing import Literal
import re


# =============================================================================
# CONFIGURABLE CONSTANTS
# =============================================================================

# Query intent types
IntentType = Literal[
    "general_knowledge",
    "doc_specific",
    "multi_document",
    "comparison",
    "ambiguous"
]

# Retrieval modes
RetrievalMode = Literal["dense", "sparse", "hybrid"]


# =============================================================================
# DECISION DATACLASS
# =============================================================================

@dataclass(frozen=True)
class PolicyDecision:
    """Immutable decision from the Adaptive Policy Engine.
    
    Attributes:
        should_retrieve: Whether retrieval should be performed.
        retrieval_mode: Preferred retrieval mode ("dense", "sparse", "hybrid").
        max_iterations: Maximum retrieval iterations (default 1).
        decision_reason: Human-readable explanation of the decision.
        intent: Classified query intent type.
    """
    should_retrieve: bool
    retrieval_mode: RetrievalMode
    max_iterations: int
    decision_reason: str
    intent: IntentType
    
    def to_dict(self) -> dict:
        """Convert decision to dictionary format."""
        return {
            "should_retrieve": self.should_retrieve,
            "retrieval_mode": self.retrieval_mode,
            "max_iterations": self.max_iterations,
            "decision_reason": self.decision_reason,
            "intent": self.intent,
        }


# =============================================================================
# KEYWORD PATTERNS FOR CLASSIFICATION
# =============================================================================

# Patterns indicating general knowledge queries (no retrieval needed)
GENERAL_KNOWLEDGE_PATTERNS = [
    r"\bwhat is\b",
    r"\bdefine\b",
    r"\bdefinition of\b",
    r"\bwho is\b",
    r"\bwho was\b",
    r"\bhistory of\b",
    r"\bwhen did\b",
    r"\bwhere is\b",
    r"\bhow does .* work\b",
    r"\bexplain the concept\b",
    r"\bcapital of\b",
]

# Patterns indicating document-specific queries
DOC_SPECIFIC_PATTERNS = [
    r"\bdocument\b",
    r"\bfile\b",
    r"\breport\b",
    r"\bpaper\b",
    r"\bmemo\b",
    r"\bcontract\b",
    r"\bsummarize\b",
    r"\bsummary\b",
    r"\bextract\b",
    r"\bin the\b.{0,30}\b(doc|file|report)\b",
    r"\bfrom the\b.{0,30}\b(doc|file|report)\b",
    r"\baccording to\b",
]

# Patterns indicating multi-document queries
MULTI_DOC_PATTERNS = [
    r"\ball\b.{0,20}\bdocument",
    r"\bacross\b.{0,20}\b(doc|file|report)",
    r"\bmultiple\b.{0,20}\b(doc|file|report)",
    r"\bevery\b.{0,15}\b(doc|file|report)",
    r"\ball\b.{0,10}\bmentions\b",
    r"\beverywhere\b",
    r"\bthroughout\b",
]

# Patterns indicating comparison queries
COMPARISON_PATTERNS = [
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bdifference between\b",
    r"\bdifferences\b",
    r"\bvs\.?\b",
    r"\bversus\b",
    r"\bcontrast\b",
    r"\bhow does .* differ\b",
    r"\bbetter than\b",
    r"\bworse than\b",
]

# Patterns indicating ambiguous queries
AMBIGUOUS_PATTERNS = [
    r"^the\s+\w+$",  # Just "the bank", "the company"
    r"^it$",
    r"^this$",
    r"^that$",
]


# =============================================================================
# ADAPTIVE POLICY ENGINE
# =============================================================================

class AdaptivePolicyEngine:
    """Rule-based policy engine for query classification and retrieval decisions.
    
    This engine uses pattern matching and keyword detection to classify
    query intent and make retrieval decisions. It is designed to be:
    - Fast and lightweight (no ML models or LLM calls)
    - Deterministic (same input always produces same output)
    - Side-effect free (pure functions)
    
    Example:
        >>> engine = AdaptivePolicyEngine()
        >>> decision = engine.decide("Show me the Q3 report")
        >>> decision.should_retrieve
        True
        >>> decision.retrieval_mode
        'hybrid'
    """
    
    def __init__(self):
        """Initialize the policy engine with compiled regex patterns."""
        # Pre-compile patterns for efficiency
        self._general_patterns = [re.compile(p, re.IGNORECASE) for p in GENERAL_KNOWLEDGE_PATTERNS]
        self._doc_patterns = [re.compile(p, re.IGNORECASE) for p in DOC_SPECIFIC_PATTERNS]
        self._multi_doc_patterns = [re.compile(p, re.IGNORECASE) for p in MULTI_DOC_PATTERNS]
        self._comparison_patterns = [re.compile(p, re.IGNORECASE) for p in COMPARISON_PATTERNS]
        self._ambiguous_patterns = [re.compile(p, re.IGNORECASE) for p in AMBIGUOUS_PATTERNS]
    
    def classify_intent(self, query: str) -> IntentType:
        """Classify the intent of a query using rule-based pattern matching.
        
        Args:
            query: The user query string.
            
        Returns:
            One of: "general_knowledge", "doc_specific", "multi_document",
                    "comparison", "ambiguous"
        
        Example:
            >>> engine = AdaptivePolicyEngine()
            >>> engine.classify_intent("What is machine learning?")
            'general_knowledge'
            >>> engine.classify_intent("Compare sales in Q1 vs Q2")
            'comparison'
        """
        if not query or not query.strip():
            return "ambiguous"
        
        query = query.strip()
        
        # Check patterns in order of specificity
        # (more specific patterns should be checked first)
        
        # 1. Check for ambiguous queries (very short, unclear)
        if len(query.split()) <= 2:
            if any(p.search(query) for p in self._ambiguous_patterns):
                return "ambiguous"
        
        # 2. Check for comparison queries
        if any(p.search(query) for p in self._comparison_patterns):
            return "comparison"
        
        # 3. Check for multi-document queries
        if any(p.search(query) for p in self._multi_doc_patterns):
            return "multi_document"
        
        # 4. Check for document-specific queries
        if any(p.search(query) for p in self._doc_patterns):
            return "doc_specific"
        
        # 5. Check for general knowledge queries
        if any(p.search(query) for p in self._general_patterns):
            return "general_knowledge"
        
        # Default: Assume document-specific (safe default for RAG systems)
        return "doc_specific"
    
    def decide(self, query: str) -> PolicyDecision:
        """Make a retrieval decision for the given query.
        
        This method classifies the query intent and returns a structured
        decision about whether to retrieve, what mode to use, and the
        reasoning behind the decision.
        
        Args:
            query: The user query string.
            
        Returns:
            PolicyDecision containing:
                - should_retrieve: bool
                - retrieval_mode: "dense" | "sparse" | "hybrid"
                - max_iterations: int (default 1)
                - decision_reason: str
                - intent: str
        
        Example:
            >>> engine = AdaptivePolicyEngine()
            >>> d = engine.decide("What is the capital of France?")
            >>> d.should_retrieve
            False
            >>> d.decision_reason
            'General knowledge query - retrieval not needed.'
        """
        intent = self.classify_intent(query)
        
        # Decision logic based on intent
        if intent == "general_knowledge":
            return PolicyDecision(
                should_retrieve=False,
                retrieval_mode="hybrid",  # Not used, but provides default
                max_iterations=1,
                decision_reason="General knowledge query - retrieval not needed.",
                intent=intent,
            )
        
        elif intent == "ambiguous":
            return PolicyDecision(
                should_retrieve=False,
                retrieval_mode="hybrid",
                max_iterations=1,
                decision_reason="Query is ambiguous - clarification may be needed.",
                intent=intent,
            )
        
        elif intent == "comparison":
            return PolicyDecision(
                should_retrieve=True,
                retrieval_mode="hybrid",  # Hybrid works best for comparisons
                max_iterations=1,
                decision_reason="Comparison query - hybrid retrieval for balanced recall.",
                intent=intent,
            )
        
        elif intent == "multi_document":
            return PolicyDecision(
                should_retrieve=True,
                retrieval_mode="dense",  # Dense for broad semantic matching
                max_iterations=1,
                decision_reason="Multi-document query - dense retrieval for broad coverage.",
                intent=intent,
            )
        
        else:  # doc_specific
            return PolicyDecision(
                should_retrieve=True,
                retrieval_mode="hybrid",
                max_iterations=1,
                decision_reason="Document-specific query - standard hybrid retrieval.",
                intent=intent,
            )


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def make_policy_decision(query: str) -> PolicyDecision:
    """Convenience function to make a policy decision without instantiating engine.
    
    Args:
        query: The user query string.
        
    Returns:
        PolicyDecision with retrieval guidance.
    
    Example:
        >>> from src.policy.engine import make_policy_decision
        >>> decision = make_policy_decision("Summarize the budget report")
        >>> decision.should_retrieve
        True
    """
    engine = AdaptivePolicyEngine()
    return engine.decide(query)
