"""Python SDK for Adaptive RAG MCP Server.

Provides a simple interface for interacting with the MCP server.
"""

import os
import requests
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

class MCPError(Exception):
    """Base class for MCP errors."""
    pass

@dataclass
class MCPClient:
    """Client for interacting with the Adaptive RAG MCP Server."""
    
    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    namespace: str = "default"
    
    def __post_init__(self):
        self.api_key = self.api_key or os.getenv("ADAPTIVE_RAG_API_KEY")
        
    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "X-Namespace": self.namespace
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools."""
        try:
            resp = requests.get(f"{self.base_url}/tools", headers=self._get_headers())
            resp.raise_for_status()
            return resp.json()["tools"]
        except requests.exceptions.RequestException as e:
            raise MCPError(f"Failed to list tools: {e}")

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific tool."""
        payload = {
            "name": tool_name,
            "arguments": arguments
        }
        
        try:
            resp = requests.post(
                f"{self.base_url}/tools/execute",
                headers=self._get_headers(),
                json=payload
            )
            resp.raise_for_status()
            
            data = resp.json()
            if data.get("isError"):
                raise MCPError(f"Tool execution failed: {data.get('content')}")
                
            return data
        except requests.exceptions.RequestException as e:
            raise MCPError(f"Failed to call tool '{tool_name}': {e}")

# Example Usage
if __name__ == "__main__":
    client = MCPClient(namespace="sdk-test")
    try:
        tools = client.list_tools()
        print(f"Found {len(tools)} tools.")
        
        # Example: Decide retrieval
        decision = client.call_tool("decide_retrieval", {"query": "What is RAG?"})
        print("\nDecision:", decision)
        
    except Exception as e:
        print(f"Error: {e}")
