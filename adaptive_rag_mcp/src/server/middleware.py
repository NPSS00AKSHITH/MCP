"""API Middleware for Rate Limiting and Context."""

import time
from collections import defaultdict
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.server.context import RequestContext, set_context
from src.server.logging import get_logger

logger = get_logger(__name__)

class ContextMiddleware(BaseHTTPMiddleware):
    """Middleware to initialize request context from headers."""
    
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(time.time()))
        namespace = request.headers.get("X-Namespace", "default")
        user_id = request.headers.get("X-User-ID")
        
        ctx = RequestContext(request_id=request_id, namespace=namespace, user_id=user_id)
        set_context(ctx)
        
        response = await call_next(request)
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter."""
    
    def __init__(self, app: ASGIApp, limit: int = 60, window: int = 60):
        super().__init__(app)
        self.limit = limit
        self.window = window
        self.requests = defaultdict(list)
        
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        
        # Clean old requests
        self.requests[client_ip] = [t for t in self.requests[client_ip] if now - t < self.window]
        
        if len(self.requests[client_ip]) >= self.limit:
            logger.warning("rate_limit_exceeded", client_ip=client_ip)
            return Response(status_code=429, content="Rate limit exceeded")
            
        self.requests[client_ip].append(now)
        
        response = await call_next(request)
        return response
