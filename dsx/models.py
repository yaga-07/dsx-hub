"""Pydantic models for DSX Hub SDK."""

from typing import List
from pydantic import BaseModel, Field


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

