"""Authentication utilities for DSX Hub server."""

from typing import Optional
from fastapi import Header, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config import get_env

API_KEY: str = get_env("DSX_API_KEY", "default-secret")
security = HTTPBearer()


def verify_api_key(authorization: Optional[str] = Header(None)) -> None:
    """
    FastAPI dependency function to verify API key from Authorization header.
    
    Checks for Bearer token in Authorization header and validates it against
    the configured API_KEY environment variable.
    
    Args:
        authorization: Authorization header value (Bearer token).
    
    Raises:
        HTTPException: 401 Unauthorized if authentication fails.
    """
    if authorization is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header"
        )
    
    # Extract token from "Bearer <token>" format
    try:
        scheme, token = authorization.split(" ", 1)
        if scheme.lower() != "bearer":
            raise ValueError("Invalid authorization scheme")
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Expected: Bearer <token>"
        )
    
    if token != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> None:
    """
    FastAPI dependency using HTTPBearer for authentication.
    
    Alternative authentication method using FastAPI's HTTPBearer security scheme.
    This is the recommended approach for Bearer token authentication.
    
    Args:
        credentials: HTTPBearer credentials from request (injected by FastAPI).
    
    Raises:
        HTTPException: 401 Unauthorized if authentication fails.
    """
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

