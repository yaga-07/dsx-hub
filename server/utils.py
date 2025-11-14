"""Utility functions for DSX Hub server."""

from pathlib import Path
import os
import json
import hashlib
from typing import List, Dict, Any, Union

from models import ManifestEntry


def compute_manifest(dataset_path: Union[Path, str]) -> List[Dict[str, Any]]:
    """
    Compute manifest for a dataset directory.
    
    Walks through all files in the dataset directory and creates manifest entries
    with file path, size, and a partial MD5 hash (first 1024 bytes).
    
    Args:
        dataset_path: Path to the dataset directory (Path object or string).
    
    Returns:
        List of dictionaries containing manifest entry data. Each entry has:
        - path: Relative path to file within dataset
        - size: File size in bytes
        - hash: MD5 hash of first 1024 bytes
    
    Note:
        Files that cannot be read (permission errors, etc.) are silently skipped.
    """
    manifest: List[Dict[str, Any]] = []
    dataset_path_obj: Path = Path(dataset_path)
    
    for root, dirs, files in os.walk(dataset_path_obj):
        for f in files:
            fp: Path = Path(root) / f
            rel: Path = fp.relative_to(dataset_path_obj)
            try:
                file_size: int = fp.stat().st_size
                file_bytes: bytes = fp.read_bytes()[:1024]
                file_hash: str = hashlib.md5(file_bytes).hexdigest()
                
                manifest.append({
                    "path": str(rel),
                    "size": file_size,
                    "hash": file_hash
                })
            except (OSError, IOError) as e:
                # Skip files that can't be read
                continue
    return manifest

