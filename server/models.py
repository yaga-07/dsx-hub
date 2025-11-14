"""Pydantic models for DSX Hub server API."""

from typing import List
from pydantic import BaseModel, Field, RootModel


class ManifestEntry(BaseModel):
    """Represents a single file entry in a dataset manifest."""
    
    path: str = Field(..., description="Relative path to the file within the dataset")
    size: int = Field(..., ge=0, description="File size in bytes")
    hash: str = Field(..., description="MD5 hash of the first 1024 bytes of the file")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "path": "images/train/image001.jpg",
                "size": 123456,
                "hash": "a1b2c3d4e5f6..."
            }
        }
    }


class ManifestResponse(RootModel[List[ManifestEntry]]):
    """Response model for dataset manifest endpoint."""
    
    root: List[ManifestEntry] = Field(..., description="List of manifest entries")
    
    model_config = {
        "json_schema_extra": {
            "example": [
                {
                    "path": "images/train/image001.jpg",
                    "size": 123456,
                    "hash": "a1b2c3d4e5f6..."
                }
            ]
        }
    }


class DatasetsResponse(RootModel[List[str]]):
    """Response model for datasets list endpoint."""
    
    root: List[str] = Field(..., description="List of available dataset names")
    
    model_config = {
        "json_schema_extra": {
            "example": ["imagenet", "cifar10", "mnist"]
        }
    }


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error message")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "Dataset not found"
            }
        }
    }


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(default="ok", description="Service status")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok"
            }
        }
    }

