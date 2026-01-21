"""Request context management."""

from contextvars import ContextVar
from typing import Optional
from dataclasses import dataclass

@dataclass
class RequestContext:
    request_id: str
    namespace: str = "default"
    user_id: Optional[str] = None

# Global context variable
_request_context: ContextVar[Optional[RequestContext]] = ContextVar("request_context", default=None)

def get_context() -> Optional[RequestContext]:
    """Get current request context."""
    return _request_context.get()

def set_context(ctx: RequestContext):
    """Set current request context."""
    _request_context.set(ctx)
