"""Generation tools using LLM."""

import json
from typing import Any, List, Optional

from src.server.llm import get_llm_client

def summarize(input_data: dict[str, Any]) -> dict[str, Any]:
    """Summarize the provided context.
    
    Input:
        context: List of {id, content} dicts
        query: Optional focus query
        style: concise | detailed (default: concise)
    """
    context = input_data.get("context", [])
    query = input_data.get("query")
    style = input_data.get("style", "concise")
    
    if not context:
        return {"error": "No context provided for summarization"}
        
    client = get_llm_client()
    
    # Prepare prompt
    context_str = "\n\n".join([f"[{c.get('id', 'unknown')}] {c.get('content', '')}" for c in context])
    
    prompt = f"""You are a helpful assistant. Summarize the following information in a {style} style.
    
    Context:
    {context_str}
    
    """
    
    if query:
        prompt += f"Focus the summary on answering this query: '{query}'\n"
        
    prompt += "\nOutput ONLY the summary."
    
    summary = client.generate_text(prompt)
    
    return {
        "summary": summary,
        "source_ids": [c.get("id") for c in context if c.get("id")],
        "metadata": {"style": style}
    }

def compare_documents(input_data: dict[str, Any]) -> dict[str, Any]:
    """Compare multiple documents.
    
    Input:
        documents: List of {id, content, label}
        focus: Optional aspect to focus on
    """
    documents = input_data.get("documents", [])
    focus = input_data.get("focus")
    
    if len(documents) < 2:
        return {"error": "At least 2 documents required for comparison"}
        
    client = get_llm_client()
    
    doc_str = "\n\n".join([f"Document '{d.get('label', d.get('id'))}' (ID: {d.get('id')}):\n{d.get('content')}" for d in documents])
    
    prompt = f"""Compare the following documents. Identify similarities, differences, and any conflicts.
    
    {doc_str}
    
    """
    
    if focus:
        prompt += f"Focus the comparison on: {focus}\n"
        
    prompt += "\nProvide a structured comparison."
    
    comparison = client.generate_text(prompt)
    
    # In a real implementation, we might ask for JSON output and parse it.
    # For now, return the text.
    return {
        "summary": comparison,
        "similarities": [], # Placeholder for structured parsing
        "differences": [],
        "conflicts": []
    }

def cite(input_data: dict[str, Any]) -> dict[str, Any]:
    """Generate an answer with citations.
    
    Input:
        query: User question
        sources: List of {id, content}
        citation_style: inline_number | inline_name
    """
    query = input_data.get("query")
    sources = input_data.get("sources", [])
    
    if not query or not sources:
        return {"error": "Query and sources are required"}
        
    client = get_llm_client()
    
    source_str = "\n\n".join([f"Source ID: {s.get('id')}\nContent: {s.get('content')}" for s in sources])
    
    prompt = f"""Answer the user's question using ONLY the provided sources.
    You MUST cite your statements using the Source IDs in brackets, e.g., [doc1], [chunk_3].
    Do not hallucinate info not in sources.
    
    Question: {query}
    
    Sources:
    {source_str}
    
    Answer:
    """
    
    response = client.generate_text(prompt)
    
    return {
        "response": response,
        "citations": [], # We could parse the response to extract citations
        "uncited_claims": []
    }

# Registry
GENERATION_TOOLS = {
    "summarize": summarize,
    "compare_documents": compare_documents,
    "cite": cite
}
