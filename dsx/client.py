"""DSX Hub SDK client with Pydantic models and type annotations."""

import requests
from pathlib import Path
import json
import os
import shutil
from typing import Optional, List, Dict, Any

from .models import ManifestEntry


class DatasetHubClient:
    """
    Synchronous client for DSX Hub.
    
    Provides methods to interact with a DSX Hub server, including listing datasets,
    fetching manifests, downloading files, and managing local cache.
    """
    
    def __init__(
        self, 
        base_url: str, 
        api_key: str, 
        cache_dir: Optional[str] = "./cache"
    ) -> None:
        """
        Initialize DSX Hub client.
        
        Args:
            base_url: Base URL of the DSX Hub server (e.g., "http://localhost:8000")
            api_key: API key for authentication
            cache_dir: Directory for caching manifests and files (optional).
                     Set to None to disable all caching.
        """
        self.base_url: str = base_url.rstrip("/")
        self.cache_dir: Optional[Path] = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.headers: Dict[str, str] = {"Authorization": f"Bearer {api_key}"}

    def list_datasets(self) -> List[str]:
        """
        List all available datasets.
        
        Returns:
            List of dataset names available on the server.
        
        Raises:
            requests.HTTPError: If the request fails.
        """
        r: requests.Response = requests.get(
            f"{self.base_url}/datasets", 
            headers=self.headers
        )
        r.raise_for_status()
        return r.json()

    def get_manifest(self, dataset: str) -> List[Dict[str, Any]]:
        """
        Get manifest for a dataset (cached locally if cache_dir is set).
        
        Args:
            dataset: Name of the dataset to get manifest for.
        
        Returns:
            List of manifest entries, each containing path, size, and hash.
            Can be validated with ManifestEntry Pydantic model.
        
        Raises:
            requests.HTTPError: If the request fails.
        """
        # Check cache if caching is enabled
        if self.cache_dir:
            cache_path: Path = self.cache_dir / f"{dataset}_manifest.json"
            if cache_path.exists():
                with open(cache_path) as f:
                    cached_data: List[Dict[str, Any]] = json.load(f)
                    return cached_data

        # Fetch from server
        r: requests.Response = requests.get(
            f"{self.base_url}/manifest/{dataset}", 
            headers=self.headers
        )
        r.raise_for_status()
        data: List[Dict[str, Any]] = r.json()
        
        # Save to cache if caching is enabled
        if self.cache_dir:
            cache_path = self.cache_dir / f"{dataset}_manifest.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2)
        
        return data

    def get_file(self, dataset: str, subpath: str) -> Path:
        """
        Download a file from a dataset (cached locally if cache_dir is set).
        
        WARNING: This method saves files to disk. If you don't want to fill up
        storage, use get_file_bytes() instead for in-memory processing.
        
        Args:
            dataset: Dataset name
            subpath: Path to file relative to dataset root
        
        Returns:
            Path to local cached file.
        
        Raises:
            ValueError: If cache_dir is None.
            requests.HTTPError: If the request fails.
        """
        if not self.cache_dir:
            raise ValueError(
                "Cannot use get_file() when cache_dir is None. "
                "Use get_file_bytes() for in-memory processing instead."
            )
        
        local_path: Path = self.cache_dir / dataset / subpath
        if local_path.exists():
            return local_path
        
        local_path.parent.mkdir(parents=True, exist_ok=True)
        url: str = f"{self.base_url}/file/{dataset}/{subpath}"
        r: requests.Response = requests.get(
            url, 
            headers=self.headers, 
            stream=True
        )
        r.raise_for_status()
        
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        return local_path
    
    def get_file_bytes(self, dataset: str, subpath: str) -> bytes:
        """
        Get file content as bytes directly from server (no disk write).
        
        This method fetches the file from the server and returns it as bytes
        without saving to disk. Perfect for in-memory processing and training
        loops where you don't want to fill up local storage.
        
        Args:
            dataset: Dataset name
            subpath: Path to file relative to dataset root
        
        Returns:
            File content as bytes.
        
        Raises:
            requests.HTTPError: If the request fails.
        """
        url: str = f"{self.base_url}/file/{dataset}/{subpath}"
        r: requests.Response = requests.get(
            url, 
            headers=self.headers, 
            stream=True
        )
        r.raise_for_status()
        return r.content

    def download_dataset(
        self, 
        dataset: str, 
        file_list: Optional[List[str]] = None
    ) -> List[Path]:
        """
        Download multiple files from a dataset.
        
        WARNING: This method saves files to disk. Consider using get_file_bytes()
        in a loop if you want in-memory processing.
        
        Args:
            dataset: Dataset name
            file_list: List of file paths relative to dataset root.
                      If None, downloads all files from manifest.
        
        Returns:
            List of paths to downloaded files.
        
        Raises:
            ValueError: If cache_dir is None.
            requests.HTTPError: If the request fails.
        """
        if file_list is None:
            manifest: List[Dict[str, Any]] = self.get_manifest(dataset)
            file_list = [item["path"] for item in manifest]
        
        downloaded: List[Path] = []
        for subpath in file_list:
            local_path: Path = self.get_file(dataset, subpath)
            downloaded.append(local_path)
        
        return downloaded
    
    def clear_cache(self, dataset: Optional[str] = None) -> int:
        """
        Clear cache for a dataset or all datasets.
        
        Args:
            dataset: Dataset name to clear cache for. If None, clears all cache.
        
        Returns:
            Number of bytes freed.
        """
        if not self.cache_dir:
            return 0
        
        if dataset:
            # Clear specific dataset
            dataset_cache: Path = self.cache_dir / dataset
            manifest_cache: Path = self.cache_dir / f"{dataset}_manifest.json"
            
            size_freed: int = 0
            if dataset_cache.exists():
                size_freed = sum(
                    f.stat().st_size 
                    for f in dataset_cache.rglob("*") 
                    if f.is_file()
                )
                shutil.rmtree(dataset_cache)
            if manifest_cache.exists():
                size_freed += manifest_cache.stat().st_size
                manifest_cache.unlink()
            
            return size_freed
        else:
            # Clear all cache
            if not self.cache_dir.exists():
                return 0
            
            size_freed: int = sum(
                f.stat().st_size 
                for f in self.cache_dir.rglob("*") 
                if f.is_file()
            )
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            return size_freed
    
    def get_cache_size(self, dataset: Optional[str] = None) -> int:
        """
        Get size of cache in bytes.
        
        Args:
            dataset: Dataset name. If None, returns total cache size.
        
        Returns:
            Size in bytes.
        """
        if not self.cache_dir or not self.cache_dir.exists():
            return 0
        
        if dataset:
            dataset_cache: Path = self.cache_dir / dataset
            manifest_cache: Path = self.cache_dir / f"{dataset}_manifest.json"
            
            size: int = 0
            if dataset_cache.exists():
                size += sum(
                    f.stat().st_size 
                    for f in dataset_cache.rglob("*") 
                    if f.is_file()
                )
            if manifest_cache.exists():
                size += manifest_cache.stat().st_size
            
            return size
        else:
            return sum(
                f.stat().st_size 
                for f in self.cache_dir.rglob("*") 
                if f.is_file()
            )

