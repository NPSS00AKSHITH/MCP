"""FastAPI MCP Server - Main entry point.

This server exposes MCP tools via HTTP REST endpoints.
Tools are registered based on the Phase 0 JSON schemas.

How tools are registered:
1. Tool schemas are defined in schemas.py (from Phase 0 contract)
2. Mock implementations are in tools/mock_implementations.py
3. The executor.py validates inputs and calls the appropriate mock
4. This main.py exposes /tools/{tool_name} endpoints
"""

from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src import __version__
from src.config import get_settings
from src.exceptions import ToolExecutionError, ToolNotFoundError, ValidationError
from src.server.auth import verify_api_key
from src.server.logging import setup_logging, get_logger
from src.server.tools.executor import execute_tool, list_tools
from src.server.schemas import get_all_tool_names, TOOL_SCHEMAS
from src.server.middleware import ContextMiddleware, RateLimitMiddleware


# Setup logging on module load
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    settings = get_settings()
    logger.info(
        "server_starting",
        version=__version__,
        host=settings.host,
        port=settings.port,
        tools_count=len(TOOL_SCHEMAS),
    )
    yield
    logger.info("server_stopping")


# FastAPI application
app = FastAPI(
    title="Adaptive RAG MCP Server",
    description="Production-grade MCP server implementing Adaptive Retrieval-Augmented Generation",
    version=__version__,
    lifespan=lifespan,
)

# Add Middleware
app.add_middleware(RateLimitMiddleware, limit=100, window=60)
app.add_middleware(ContextMiddleware)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ToolRequest(BaseModel):
    """Generic tool request body."""
    # Using dict for flexibility - validation happens via JSON schema
    class Config:
        extra = "allow"


class ToolResponse(BaseModel):
    """Generic tool response wrapper."""
    success: bool
    tool_name: str
    result: dict[str, Any] | None = None
    error: str | None = None


class ToolListItem(BaseModel):
    """Tool listing item."""
    name: str
    description: str
    inputSchema: dict[str, Any]


# Health check endpoint (no auth)
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": __version__}


# List tools endpoint
@app.get("/tools", response_model=list[ToolListItem])
async def get_tools(api_key: str = Depends(verify_api_key)):
    """
    List all available MCP tools.
    
    Returns tool names, descriptions, and input schemas.
    """
    return list_tools()


# Execute tool endpoint
@app.post("/tools/{tool_name}", response_model=ToolResponse)
async def call_tool(
    tool_name: str,
    request: dict[str, Any],
    api_key: str = Depends(verify_api_key),
):
    """
    Execute an MCP tool by name.
    
    The request body should match the tool's inputSchema.
    """
    try:
        result = execute_tool(tool_name, request)
        return ToolResponse(
            success=True,
            tool_name=tool_name,
            result=result,
        )
    except ToolNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except ToolExecutionError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# Tool schema endpoint
@app.get("/tools/{tool_name}/schema")
async def get_tool_schema(
    tool_name: str,
    api_key: str = Depends(verify_api_key),
):
    """Get the full schema for a specific tool."""
    if tool_name not in TOOL_SCHEMAS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found",
        )
    return TOOL_SCHEMAS[tool_name]


# Feedback endpoint for adaptive learning
class FeedbackRequest(BaseModel):
    """Request body for recording feedback."""
    chunk_id: str
    query: str
    query_type: str = "unknown"
    accepted: bool
    original_score: float = 0.0


@app.post("/feedback")
async def record_feedback(
    feedback: FeedbackRequest,
    api_key: str = Depends(verify_api_key),
):
    """Record user feedback for adaptive learning.
    
    This endpoint records whether a retrieved chunk was helpful,
    allowing the adaptive memory system to improve over time.
    """
    from src.retrieval.adaptive_memory import get_adaptive_memory
    
    memory = get_adaptive_memory()
    memory.record_feedback(
        chunk_id=feedback.chunk_id,
        query=feedback.query,
        query_type=feedback.query_type,
        accepted=feedback.accepted,
        original_score=feedback.original_score,
    )
    
    return {"status": "recorded", "chunk_id": feedback.chunk_id}


@app.get("/memory/stats")
async def get_memory_stats(api_key: str = Depends(verify_api_key)):
    """Get adaptive memory statistics."""
    from src.retrieval.adaptive_memory import get_adaptive_memory
    
    memory = get_adaptive_memory()
    return memory.get_stats()


def main():
    """Run the server."""
    settings = get_settings()
    uvicorn.run(
        "src.server.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
