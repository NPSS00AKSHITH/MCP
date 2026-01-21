"""LangGraph-based state machine for retrieval workflow.

Replaces procedural loop with explicit state machine:
- Nodes = Actions (route, retrieve, grade, generate)
- Edges = Transitions (conditional based on state)
- State = Shared context (query, docs, confidence, iteration)

Benefits:
- Visualizable (can render as graph diagram)
- Debuggable (can inspect state at each step)
- Testable (can test nodes in isolation)
- Maintainable (clear control flow)

Note: This module requires langgraph to be installed:
    pip install langgraph

If langgraph is not installed, the module will gracefully degrade
and provide a fallback to the existing IterativeController.
"""

from typing import TypedDict, List, Literal, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from src.server.logging import get_logger

logger = get_logger(__name__)

# Check if langgraph is available
try:
    from langgraph.graph import StateGraph, END
    import operator
    from typing import Annotated
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("langgraph_not_installed", msg="Install with: pip install langgraph")


# =============================================================================
# STATE DEFINITION
# =============================================================================

class RAGState(TypedDict, total=False):
    """Shared state for the RAG workflow.
    
    This state is passed between all nodes in the graph.
    Each node reads from state and writes updates.
    
    Attributes:
        query: Original user query
        max_iterations: Maximum iterations allowed
        should_retrieve: Whether retrieval should happen
        retrieval_mode: Mode for retrieval (dense/sparse/hybrid)
        query_complexity: Classified complexity of query
        retrieved_docs: Documents from retrieval
        reranked_results: Results after reranking
        confidence_score: Current confidence score
        is_confident: Whether we have sufficient confidence
        contradiction_detected: Whether contradictions were found
        iteration: Current iteration number
        stop_reason: Reason for stopping
        final_answer: Generated answer (if any)
        error: Error message (if any)
    """
    # Input
    query: str
    max_iterations: int
    
    # Policy decision
    should_retrieve: bool
    retrieval_mode: str
    query_complexity: str
    
    # Retrieval results
    retrieved_docs: List[dict]
    reranked_results: List[dict]
    
    # Quality assessment
    confidence_score: float
    is_confident: bool
    contradiction_detected: bool
    
    # Loop control
    iteration: int
    stop_reason: str
    
    # Output
    final_answer: Optional[str]
    error: Optional[str]


# =============================================================================
# NODE FUNCTIONS
# =============================================================================

def route_query(state: RAGState) -> RAGState:
    """Node: Analyze query and decide retrieval strategy.
    
    Uses Policy Engine to classify query and determine if retrieval needed.
    """
    from src.policy.engine import AdaptivePolicyEngine
    
    query = state.get("query", "")
    
    try:
        policy = AdaptivePolicyEngine()
        decision = policy.decide(query)
        
        logger.info(
            "route_query",
            query_length=len(query),
            should_retrieve=decision.should_retrieve,
            mode=decision.retrieval_mode,
            intent=decision.intent,
        )
        
        return {
            **state,
            "should_retrieve": decision.should_retrieve,
            "retrieval_mode": decision.retrieval_mode,
            "query_complexity": decision.intent,
        }
    except Exception as e:
        logger.error("route_query_error", error=str(e))
        return {
            **state,
            "should_retrieve": True,  # Default to retrieval
            "retrieval_mode": "hybrid",
            "query_complexity": "unknown",
            "error": str(e),
        }


def retrieve_documents(state: RAGState) -> RAGState:
    """Node: Execute retrieval using selected mode."""
    from src.retrieval.hybrid import get_hybrid_retriever, SearchMode
    
    query = state.get("query", "")
    mode_str = state.get("retrieval_mode", "hybrid")
    iteration = state.get("iteration", 0) + 1
    
    try:
        retriever = get_hybrid_retriever()
        mode = SearchMode(mode_str)
        
        results = retriever.search(
            query=query,
            k=5,
            mode=mode,
        )
        
        # Convert to dict format
        docs = [
            {
                "id": r.chunk_id,
                "content": r.content,
                "score": r.score,
                "metadata": r.metadata,
            }
            for r in results
        ]
        
        logger.info(
            "retrieve_documents",
            mode=mode_str,
            iteration=iteration,
            retrieved_count=len(docs),
        )
        
        return {
            **state,
            "retrieved_docs": docs,
            "iteration": iteration,
        }
    except Exception as e:
        logger.error("retrieve_documents_error", error=str(e))
        return {
            **state,
            "retrieved_docs": [],
            "iteration": iteration,
            "error": str(e),
        }


def grade_relevance(state: RAGState) -> RAGState:
    """Node: Rerank and assess evidence quality."""
    from src.retrieval.adaptive_memory import get_adaptive_memory
    from src.retrieval.evidence_scoring import compute_evidence_scores
    
    query = state.get("query", "")
    docs = state.get("retrieved_docs", [])
    query_type = state.get("query_complexity", "unknown")
    
    if not docs:
        return {
            **state,
            "reranked_results": [],
            "confidence_score": 0.0,
            "is_confident": False,
        }
    
    try:
        # Use adaptive memory for reranking
        memory = get_adaptive_memory()
        reranked, quality = memory.rerank_with_memory(
            query=query,
            documents=docs,
            query_type=query_type,
        )
        
        # Extract top scores for evidence assessment
        scores = [r.relevance_score for r in reranked]
        evidence = compute_evidence_scores(scores)
        
        logger.info(
            "grade_relevance",
            top_score=round(evidence.top_score, 4),
            is_confident=not evidence.low_confidence_flag,
            iteration=state.get("iteration", 0),
        )
        
        return {
            **state,
            "reranked_results": [
                {
                    "id": r.id,
                    "content": r.content,
                    "score": r.relevance_score,
                    "metadata": r.metadata,
                }
                for r in reranked
            ],
            "confidence_score": evidence.top_score,
            "is_confident": not evidence.low_confidence_flag,
            "contradiction_detected": evidence.contradiction_flag if hasattr(evidence, 'contradiction_flag') else False,
        }
    except Exception as e:
        logger.error("grade_relevance_error", error=str(e))
        return {
            **state,
            "reranked_results": [],
            "confidence_score": 0.0,
            "is_confident": False,
            "error": str(e),
        }


def generate_answer(state: RAGState) -> RAGState:
    """Node: Mark retrieval as complete with final state."""
    logger.info(
        "generate_answer",
        is_confident=state.get("is_confident", False),
        result_count=len(state.get("reranked_results", [])),
    )
    
    return {
        **state,
        "stop_reason": "answer_generated",
    }


def mark_insufficient(state: RAGState) -> RAGState:
    """Node: Mark as insufficient evidence."""
    return {
        **state,
        "stop_reason": "insufficient_evidence",
    }


# =============================================================================
# CONDITIONAL EDGES
# =============================================================================

def should_retrieve(state: RAGState) -> Literal["retrieve", "skip"]:
    """Conditional edge: Should we retrieve documents?"""
    if state.get("should_retrieve", True):
        return "retrieve"
    return "skip"


def should_retry(state: RAGState) -> Literal["retrieve", "generate", "insufficient"]:
    """Conditional edge: Should we retry retrieval or proceed?"""
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)
    is_confident = state.get("is_confident", False)
    
    # Confident - proceed to generate
    if is_confident:
        return "generate"
    
    # Max iterations reached
    if iteration >= max_iterations:
        # Check if we have any results
        if state.get("reranked_results"):
            return "generate"  # Try to answer with what we have
        return "insufficient"
    
    # Retry with different strategy
    return "retrieve"


# =============================================================================
# GRAPH BUILDER
# =============================================================================

def build_rag_graph():
    """Build the complete RAG workflow as a LangGraph state machine.
    
    Graph structure:
        START → route → [should_retrieve?]
                         ├─ yes → retrieve → grade → [should_retry?]
                         │                            ├─ confident → generate → END
                         │                            ├─ retry → retrieve (loop)
                         │                            └─ insufficient → mark_insufficient → END
                         └─ no → mark_insufficient → END
    
    Returns:
        Compiled StateGraph or None if langgraph not available.
    """
    if not LANGGRAPH_AVAILABLE:
        logger.warning("build_rag_graph_skipped", reason="langgraph not installed")
        return None
    
    # Create graph
    graph = StateGraph(RAGState)
    
    # Add nodes
    graph.add_node("route", route_query)
    graph.add_node("retrieve", retrieve_documents)
    graph.add_node("grade", grade_relevance)
    graph.add_node("generate", generate_answer)
    graph.add_node("insufficient", mark_insufficient)
    
    # Set entry point
    graph.set_entry_point("route")
    
    # Add conditional edges
    graph.add_conditional_edges(
        "route",
        should_retrieve,
        {
            "retrieve": "retrieve",
            "skip": "insufficient",
        }
    )
    
    graph.add_edge("retrieve", "grade")
    
    graph.add_conditional_edges(
        "grade",
        should_retry,
        {
            "retrieve": "retrieve",  # Loop back for retry
            "generate": "generate",
            "insufficient": "insufficient",
        }
    )
    
    graph.add_edge("generate", END)
    graph.add_edge("insufficient", END)
    
    return graph.compile()


# =============================================================================
# CONTROLLER CLASS
# =============================================================================

class GraphBasedController:
    """Controller using LangGraph state machine instead of procedural loop.
    
    Falls back to IterativeController if langgraph is not installed.
    """
    
    def __init__(self):
        """Initialize the graph-based controller."""
        self.graph = build_rag_graph()
        self._fallback_controller = None
    
    @property
    def fallback_controller(self):
        """Lazy-load fallback controller."""
        if self._fallback_controller is None:
            from src.retrieval.iterative_controller import IterativeController
            self._fallback_controller = IterativeController()
        return self._fallback_controller
    
    def retrieve(self, query: str, max_iterations: int = 3) -> dict:
        """Execute retrieval using state machine.
        
        This replaces IterativeController.retrieve() with graph-based execution.
        Falls back to IterativeController if langgraph not available.
        
        Args:
            query: User query
            max_iterations: Maximum retrieval iterations
            
        Returns:
            Final state dictionary with results
        """
        # Fallback if langgraph not available
        if self.graph is None:
            logger.info("using_fallback_controller", reason="langgraph not available")
            result = self.fallback_controller.retrieve(query)
            return self._convert_result_to_state(result)
        
        # Initialize state
        initial_state: RAGState = {
            "query": query,
            "max_iterations": max_iterations,
            "iteration": 0,
            "retrieved_docs": [],
            "reranked_results": [],
        }
        
        logger.info("graph_execution_start", query_length=len(query))
        
        try:
            # Execute graph
            final_state = self.graph.invoke(initial_state)
            
            logger.info(
                "graph_execution_complete",
                stop_reason=final_state.get("stop_reason", "unknown"),
                iterations=final_state.get("iteration", 0),
                is_confident=final_state.get("is_confident", False),
            )
            
            return dict(final_state)
        except Exception as e:
            logger.error("graph_execution_error", error=str(e))
            # Fallback on error
            result = self.fallback_controller.retrieve(query)
            return self._convert_result_to_state(result)
    
    def _convert_result_to_state(self, result) -> dict:
        """Convert IterativeController result to state format."""
        return {
            "query": result.query,
            "iteration": result.iterations,
            "reranked_results": [
                {"id": c.id, "content": c.content, "score": c.relevance_score}
                for c in result.final_chunks
            ],
            "is_confident": result.outcome_type.value in ("answer_ready", "partial_answer"),
            "stop_reason": result.stop_reason.value,
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

_graph_controller: Optional[GraphBasedController] = None


def get_graph_controller() -> GraphBasedController:
    """Get or create graph-based controller instance."""
    global _graph_controller
    if _graph_controller is None:
        _graph_controller = GraphBasedController()
    return _graph_controller
