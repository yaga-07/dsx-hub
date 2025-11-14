"""DSX Hub server implementation with FastAPI and Pydantic models."""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import FileResponse
from pathlib import Path
import json
import sys
from typing import List, Dict, Any, Optional

# Add server directory to path for imports
server_dir = Path(__file__).parent
if str(server_dir) not in sys.path:
    sys.path.insert(0, str(server_dir))

from config import get_env
from auth import verify_token, security
from utils import compute_manifest
from models import (
    ManifestEntry,
    ErrorResponse,
    HealthResponse,
)

BASE_PATH: Path = Path(get_env("DSX_DATASET_PATH", "/path/to/datasets"))
MANIFEST_PATH: Path = BASE_PATH / "_manifest_cache"

app: FastAPI = FastAPI(
    title="DSX Hub API",
    description="A lightweight data streaming framework for efficient dataset distribution",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get(
    "/datasets",
    response_model=List[str],
    summary="List all available datasets",
    description="Returns a list of all available dataset names on the server.",
    tags=["datasets"],
    responses={
        200: {"description": "List of dataset names"},
        500: {"model": ErrorResponse, "description": "Server error"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def list_datasets(
    _: None = Depends(verify_token)
) -> List[str]:
    """
    List all available datasets.
    
    Requires authentication via Bearer token in Authorization header.
    
    Returns:
        List of dataset names available on the server.
    
    Raises:
        HTTPException: 500 if dataset base path does not exist.
    """
    
    if not BASE_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Dataset base path does not exist"
        )
    
    datasets: List[str] = [
        f.name for f in BASE_PATH.iterdir() 
        if f.is_dir() and not f.name.startswith("_")
    ]
    return datasets


@app.get(
    "/manifest/{dataset}",
    response_model=List[ManifestEntry],
    summary="Get dataset manifest",
    description="Get manifest for a dataset (cached). Returns list of files with metadata.",
    tags=["datasets"],
    responses={
        200: {"description": "List of manifest entries"},
        404: {"model": ErrorResponse, "description": "Dataset not found"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def manifest(
    dataset: str,
    _: None = Depends(verify_token)
) -> List[ManifestEntry]:
    """
    Get manifest for a dataset (cached).
    
    The manifest contains metadata about all files in the dataset including
    path, size, and hash. Results are cached for performance.
    
    Args:
        dataset: Name of the dataset to get manifest for.
    
    Returns:
        List of manifest entries, each containing path, size, and hash.
    
    Raises:
        HTTPException: 404 if dataset is not found.
    """
    
    dataset_path: Path = BASE_PATH / dataset
    cache_file: Path = MANIFEST_PATH / f"{dataset}.json"
    MANIFEST_PATH.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )

    if cache_file.exists():
        with open(cache_file) as f:
            cached_data: List[Dict[str, Any]] = json.load(f)
            # Validate cached data with Pydantic
            manifest_entries: List[ManifestEntry] = [
                ManifestEntry(**entry) for entry in cached_data
            ]
            return manifest_entries

    manifest_data: List[Dict[str, Any]] = compute_manifest(dataset_path)
    # Validate manifest data with Pydantic
    manifest_entries = [ManifestEntry(**entry) for entry in manifest_data]
    
    with open(cache_file, "w") as f:
        json.dump([entry.model_dump() for entry in manifest_entries], f, indent=2)
    
    return manifest_entries


@app.get(
    "/file/{dataset}/{subpath:path}",
    summary="Download file from dataset",
    description="Download a file from a dataset. Returns the file content.",
    tags=["files"],
    responses={
        200: {"description": "File content"},
        400: {"model": ErrorResponse, "description": "Invalid file path"},
        404: {"model": ErrorResponse, "description": "File not found"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def get_file(
    dataset: str,
    subpath: str,
    _: None = Depends(verify_token)
) -> FileResponse:
    """
    Download a file from a dataset.
    
    Args:
        dataset: Name of the dataset.
        subpath: Path to the file relative to the dataset root.
    
    Returns:
        FileResponse with the file content.
    
    Raises:
        HTTPException: 400 if file path is invalid, 404 if file not found.
    """
    
    file_path: Path = BASE_PATH / dataset / subpath
    
    # Security check: ensure file is within dataset directory
    try:
        file_path.resolve().relative_to(BASE_PATH.resolve())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file path"
        )
    
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )
    
    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="application/octet-stream"
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Health check endpoint (no authentication required).",
    tags=["system"],
    responses={
        200: {"description": "Service is healthy"},
    },
)
async def health() -> HealthResponse:
    """
    Health check endpoint (no auth required).
    
    Returns:
        HealthResponse indicating service status.
    """
    return HealthResponse(status="ok")


if __name__ == "__main__":
    import uvicorn
    port: int = int(get_env("DSX_PORT", "8000"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

