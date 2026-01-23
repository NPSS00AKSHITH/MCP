"""Custom exceptions for the Adaptive RAG MCP Server.

Enhanced with structured error codes and actionable suggestions per MCP best practices.
"""

from typing import Any


class AdaptiveRAGError(Exception):
    """Base exception for all Adaptive RAG errors.
    
    Attributes:
        message: Human-readable error message
        error_code: Machine-readable error code for categorization
        suggestion: Actionable suggestion for resolving the error
        details: Additional context about the error
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "ADAPTIVE_RAG_ERROR",
        suggestion: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.error_code = error_code
        self.suggestion = suggestion
        self.details = details or {}
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format error message with suggestion if available."""
        msg = f"[{self.error_code}] {self.message}"
        if self.suggestion:
            msg += f" Suggestion: {self.suggestion}"
        return msg
    
    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary for JSON response."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "suggestion": self.suggestion,
            "details": self.details,
        }


class AuthenticationError(AdaptiveRAGError):
    """Raised when API key authentication fails."""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        suggestion: str = "Provide a valid X-API-Key header.",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="AUTH_ERROR",
            suggestion=suggestion,
            details=details,
        )


class ValidationError(AdaptiveRAGError):
    """Raised when input validation fails."""
    
    def __init__(
        self,
        message: str,
        field: str | None = None,
        suggestion: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.field = field
        if suggestion is None:
            suggestion = f"Check the '{field}' parameter format." if field else "Verify input parameters match the schema."
        
        error_details = details or {}
        if field:
            error_details["field"] = field
            
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            suggestion=suggestion,
            details=error_details,
        )


class ToolNotFoundError(AdaptiveRAGError):
    """Raised when a requested tool does not exist."""
    
    def __init__(
        self,
        tool_name: str,
        available_tools: list[str] | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.available_tools = available_tools or []
        
        # Generate helpful suggestion based on available tools
        if available_tools:
            similar = [t for t in available_tools if tool_name.lower() in t.lower() or t.lower() in tool_name.lower()]
            if similar:
                suggestion = f"Did you mean: {', '.join(similar[:3])}? Use GET /tools to list all available tools."
            else:
                suggestion = f"Use GET /tools to list all available tools. Available: {', '.join(available_tools[:5])}..."
        else:
            suggestion = "Use GET /tools to list all available tools."
        
        super().__init__(
            message=f"Tool '{tool_name}' not found",
            error_code="TOOL_NOT_FOUND",
            suggestion=suggestion,
            details={"requested_tool": tool_name, "available_count": len(available_tools)},
        )


class ToolExecutionError(AdaptiveRAGError):
    """Raised when a tool fails during execution."""
    
    def __init__(
        self,
        message: str,
        tool_name: str,
        suggestion: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.tool_name = tool_name
        
        if suggestion is None:
            suggestion = "Try again with different parameters or check the input format."
        
        error_details = details or {}
        error_details["tool_name"] = tool_name
        
        super().__init__(
            message=message,
            error_code="TOOL_EXECUTION_ERROR",
            suggestion=suggestion,
            details=error_details,
        )


class DocumentNotFoundError(AdaptiveRAGError):
    """Raised when a requested document does not exist."""
    
    def __init__(
        self,
        doc_id: str,
        suggestion: str = "Use list_documents to see available documents.",
    ) -> None:
        self.doc_id = doc_id
        super().__init__(
            message=f"Document '{doc_id}' not found",
            error_code="DOCUMENT_NOT_FOUND",
            suggestion=suggestion,
            details={"doc_id": doc_id},
        )


class IndexNotReadyError(AdaptiveRAGError):
    """Raised when vector index is not ready for queries."""
    
    def __init__(
        self,
        message: str = "Vector index not ready",
        suggestion: str = "Ingest and index documents first using ingest_document and index_document tools.",
    ) -> None:
        super().__init__(
            message=message,
            error_code="INDEX_NOT_READY",
            suggestion=suggestion,
        )


class RateLimitError(AdaptiveRAGError):
    """Raised when rate limit is exceeded."""
    
    def __init__(
        self,
        limit: int,
        window_seconds: int,
        retry_after: int | None = None,
    ) -> None:
        suggestion = f"Wait {retry_after} seconds before retrying." if retry_after else "Please wait before making more requests."
        super().__init__(
            message=f"Rate limit exceeded: {limit} requests per {window_seconds} seconds",
            error_code="RATE_LIMIT_EXCEEDED",
            suggestion=suggestion,
            details={"limit": limit, "window_seconds": window_seconds, "retry_after": retry_after},
        )

