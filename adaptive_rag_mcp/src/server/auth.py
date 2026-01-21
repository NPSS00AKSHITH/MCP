"""API key authentication for the MCP server."""

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from src.config import get_settings
from src.exceptions import AuthenticationError

# API key header definition
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(api_key_header)) -> str:
    """
    Verify the API key from the request header.
    
    Args:
        api_key: The API key from the X-API-Key header
        
    Returns:
        The validated API key
        
    Raises:
        HTTPException: If API key is missing or invalid
    """
    settings = get_settings()
    
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
        )
    
    if api_key != settings.adaptive_rag_api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )
    
    return api_key
