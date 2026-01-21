"""Custom exceptions for the Adaptive RAG MCP Server."""


class AdaptiveRAGError(Exception):
    """Base exception for all Adaptive RAG errors."""
    pass


class AuthenticationError(AdaptiveRAGError):
    """Raised when API key authentication fails."""
    pass


class ValidationError(AdaptiveRAGError):
    """Raised when input validation fails."""
    pass


class ToolNotFoundError(AdaptiveRAGError):
    """Raised when a requested tool does not exist."""
    pass


class ToolExecutionError(AdaptiveRAGError):
    """Raised when a tool fails during execution."""
    pass
