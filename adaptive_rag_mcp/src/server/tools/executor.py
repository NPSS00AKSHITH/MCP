"""Tool executor with validation and logging."""

import time
import uuid
from typing import Any

import jsonschema
from jsonschema import ValidationError as JsonSchemaValidationError

from src.exceptions import ToolExecutionError, ToolNotFoundError, ValidationError
from src.server.logging import log_tool_call, get_logger
from src.server.schemas import get_tool_input_schema, get_all_tool_names, TOOL_SCHEMAS
from src.server.tools.mock_implementations import execute_mock_tool
from src.server.tools.ingestion_tools import INGESTION_TOOLS
from src.server.tools.retrieval_tools import RETRIEVAL_TOOLS
from src.server.tools.rerank_tools import RERANK_TOOLS
from src.server.tools.policy_tools import POLICY_TOOLS
from src.server.tools.loop_tools import LOOP_TOOLS
from src.server.tools.generation_tools import GENERATION_TOOLS

logger = get_logger(__name__)


def validate_input(tool_name: str, input_data: dict[str, Any]) -> None:
    """
    Validate input data against the tool's JSON schema.
    
    Args:
        tool_name: Name of the tool
        input_data: Input data to validate
        
    Raises:
        ToolNotFoundError: If tool doesn't exist
        ValidationError: If input doesn't match schema
    """
    schema = get_tool_input_schema(tool_name)
    if schema is None:
        raise ToolNotFoundError(f"Tool '{tool_name}' not found")
    
    try:
        jsonschema.validate(instance=input_data, schema=schema)
    except JsonSchemaValidationError as e:
        raise ValidationError(f"Input validation failed: {e.message}")


def _execute_tool_impl(tool_name: str, input_data: dict[str, Any]) -> dict[str, Any]:
    """Execute the appropriate tool implementation.
    
    Routes to real implementations for ingestion, retrieval, and rerank tools,
    mock implementations for everything else.
    """
    # Check if this is an ingestion tool
    if tool_name in INGESTION_TOOLS:
        return INGESTION_TOOLS[tool_name](input_data)
    
    # Check if this is a retrieval tool
    if tool_name in RETRIEVAL_TOOLS:
        return RETRIEVAL_TOOLS[tool_name](input_data)
    
    # Check if this is a rerank tool
    if tool_name in RERANK_TOOLS:
        return RERANK_TOOLS[tool_name](input_data)
    
    # Check if this is a policy tool
    if tool_name in POLICY_TOOLS:
        return POLICY_TOOLS[tool_name](input_data)

    # Check if this is a loop tool
    if tool_name in LOOP_TOOLS:
        return LOOP_TOOLS[tool_name](input_data)

    # Check if this is a generation tool
    if tool_name in GENERATION_TOOLS:
        return GENERATION_TOOLS[tool_name](input_data)
    
    # Fall back to mock implementations
    return execute_mock_tool(tool_name, input_data)


def execute_tool(tool_name: str, input_data: dict[str, Any]) -> dict[str, Any]:
    """
    Execute a tool with validation and logging.
    
    Args:
        tool_name: Name of the tool to execute
        input_data: Input data for the tool
        
    Returns:
        Tool output dictionary
        
    Raises:
        ToolNotFoundError: If tool doesn't exist
        ValidationError: If input validation fails
        ToolExecutionError: If tool execution fails
    """
    request_id = str(uuid.uuid4())[:8]
    
    # Log start
    log_tool_call(tool_name, request_id, input_data)
    
    start_time = time.perf_counter()
    
    try:
        # Validate input
        validate_input(tool_name, input_data)
        
        # Execute tool (routes to real or mock implementation)
        result = _execute_tool_impl(tool_name, input_data)
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Log success
        log_tool_call(tool_name, request_id, input_data, result, duration_ms=duration_ms)
        
        return result
        
    except (ToolNotFoundError, ValidationError):
        duration_ms = (time.perf_counter() - start_time) * 1000
        raise
        
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        log_tool_call(tool_name, request_id, input_data, error=str(e), duration_ms=duration_ms)
        raise ToolExecutionError(f"Tool execution failed: {e}") from e


def list_tools(query_context: str | None = None, top_k: int = 5) -> list[dict[str, Any]]:
    """
    List available tools with their schemas, optionally filtered by query context.
    
    Uses TURA-style semantic discovery when query_context is provided.
    This prevents "prompt bloat" by only loading relevant tools per query.
    
    Args:
        query_context: Optional query to filter tools semantically.
        top_k: Number of tools to return when filtering (default: 5).
        
    Returns:
        List of tool definitions
    """
    if query_context:
        # TURA-style: only return relevant tools
        from src.policy.tool_discovery import get_tool_registry
        registry = get_tool_registry()
        relevant_tool_names = registry.discover_tools(query_context, top_k=top_k)
        return [
            {
                "name": name,
                "description": TOOL_SCHEMAS[name]["description"],
                "inputSchema": TOOL_SCHEMAS[name]["inputSchema"],
            }
            for name in relevant_tool_names
            if name in TOOL_SCHEMAS
        ]
    
    # No context: return all (backward compatible)
    return [
        {
            "name": name,
            "description": schema["description"],
            "inputSchema": schema["inputSchema"],
        }
        for name, schema in TOOL_SCHEMAS.items()
    ]
