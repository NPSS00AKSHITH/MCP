"""MCP tools for Adaptive Retrieval Policy."""

from typing import Any
from src.server.policy import PolicyEngine, QueryType

# Global instance
_ENGINE = PolicyEngine()

def decide_retrieval(input_data: dict[str, Any]) -> dict[str, Any]:
    """Decide whether and how to retrieve information for a query.
    
    analyzes the query to determine if it requires retrieval, what type of query it is
    (general, specific, comparison, etc.), and recommends retrieval parameters.
    
    Input:
        query: The user query string.
        
    Output:
        decision: logic for the decision (retrieve/skip).
        query_type: classification of the query.
        parameters: suggested retrieval parameters (mode, k).
        reason: explanation of the decision.
    """
    query = input_data.get("query", "")
    
    decision = _ENGINE.decide(query)
    
    return {
        "decision": "retrieve" if decision.should_retrieve else "skip",
        "query_type": decision.query_type.value,
        "parameters": {
            "mode": decision.search_mode,
            "max_chunks": decision.max_k,
            "filters": decision.filters
        },
        "reason": decision.reason,
        "suggested_rewrite": decision.suggested_rewrite
    }

# Registry for policy tools
POLICY_TOOLS: dict[str, callable] = {
    "decide_retrieval": decide_retrieval,
}
