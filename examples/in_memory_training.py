"""
Example: In-Memory Training with DSX Hub

This example demonstrates how to train a model using data streamed directly
into memory from the server, without saving any files to disk. Perfect for
when you have limited local storage or want to keep everything in memory.
"""

import io
import threading
import queue
from typing import Tuple, Optional
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent / ".env")

# Import DSX Hub SDK
sys.path.insert(0, str(Path(__file__).parent.parent))

from dsx import DatasetHubClient

# Configuration
HUB_URL = os.getenv("DSX_HUB_URL", "http://localhost:8000")
API_KEY = os.getenv("DSX_API_KEY", "default-secret")
DATASET_NAME = "imagenet"
BATCH_SIZE = 8


class InMemoryBatchLoader:
    """Loads batches into memory with prefetching (no disk writes)."""
    
    def __init__(self, hub_client: DatasetHubClient, dataset_name: str, 
                 file_list: list, batch_size: int = 8, transform=None, 
                 prefetch_batches: int = 2):
        """
        Initialize in-memory batch loader.
        
        Args:
            hub_client: DatasetHubClient instance
            dataset_name: Name of the dataset
            file_list: List of file paths relative to dataset root
            batch_size: Number of samples per batch
            transform: Optional image transform
            prefetch_batches: Number of batches to prefetch ahead
        """
        self.hub_client = hub_client
        self.dataset_name = dataset_name
        self.file_list = file_list
        self.batch_size = batch_size
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.prefetch_batches = prefetch_batches
        
        # In-memory batch storage: queue of (images, labels) tensors
        self.batch_queue = queue.Queue(maxsize=prefetch_batches)
        self.stop_prefetch = False
        self.prefetch_thread = None
        
        self.num_batches = (len(file_list) + batch_size - 1) // batch_size
    
    def _load_batch(self, batch_indices: list) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Load a batch of images into memory (no disk writes)."""
        images = []
        labels = []
        
        for idx in batch_indices:
            if idx >= len(self.file_list):
                break
            
            subpath = self.file_list[idx]
            
            # Get file bytes directly from server (no disk write!)
            file_bytes = self.hub_client.get_file_bytes(self.dataset_name, subpath)
            
            # Load image from bytes in memory
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            
            # Apply transform
            if self.transform:
                image = self.transform(image)
            
            images.append(image)
            labels.append(0)  # Dummy label - replace with actual label extraction
        
        # Stack into batch tensors
        if images:
            images_tensor = torch.stack(images)
            labels_tensor = torch.tensor(labels)
            return images_tensor, labels_tensor
        return None
    
    def _prefetch_worker(self):
        """Worker thread that prefetches batches."""
        batch_idx = 0
        
        while not self.stop_prefetch and batch_idx < self.num_batches:
            # Calculate indices for this batch
            start_idx = batch_idx * self.batch_size
            end_idx = start_idx + self.batch_size
            batch_indices = list(range(start_idx, end_idx))
            
            # Load batch into memory
            batch = self._load_batch(batch_indices)
            
            if batch is not None:
                # Put batch in queue (blocks if queue is full)
                self.batch_queue.put(batch)
                batch_idx += 1
    
    def start_prefetching(self):
        """Start prefetching batches in background thread."""
        self.stop_prefetch = False
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()
    
    def get_batch(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get next batch from queue (blocks until available)."""
        try:
            return self.batch_queue.get(timeout=1.0)
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop prefetching."""
        self.stop_prefetch = True
        if self.prefetch_thread:
            self.prefetch_thread.join(timeout=5.0)
    
    def __iter__(self):
        """Make this iterable."""
        self.start_prefetching()
        return self
    
    def __next__(self):
        """Get next batch."""
        batch = self.get_batch()
        if batch is None:
            self.stop()
            raise StopIteration
        return batch
    
    def __len__(self):
        """Return number of batches."""
        return self.num_batches


def train_with_in_memory_batches():
    """Train model using in-memory batch loading (no disk writes)."""
    print("=" * 60)
    print("In-Memory Training Example")
    print("=" * 60)
    print("Data is streamed directly into memory - no files saved to disk!")
    print()
    
    # Initialize hub client (cache_dir not used for in-memory mode)
    hub = DatasetHubClient(HUB_URL, API_KEY, cache_dir="./cache")
    
    # Get dataset manifest
    print("Fetching dataset manifest...")
    manifest = hub.get_manifest(DATASET_NAME)
    print(f"Dataset has {len(manifest)} files")
    
    # Filter image files
    image_files = [
        item["path"] for item in manifest 
        if item["path"].lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    print(f"Found {len(image_files)} image files")
    
    # Limit to first 100 images for demo
    image_files = image_files[:100]
    print(f"Using {len(image_files)} images for training")
    
    # Create in-memory batch loader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    batch_loader = InMemoryBatchLoader(
        hub_client=hub,
        dataset_name=DATASET_NAME,
        file_list=image_files,
        batch_size=BATCH_SIZE,
        transform=transform,
        prefetch_batches=2  # Prefetch 2 batches ahead
    )
    
    print(f"Created batch loader: {len(batch_loader)} batches")
    print("\nKey features:")
    print("✓ Files streamed directly into memory (no disk writes)")
    print("✓ Next batch prefetched while processing current batch")
    print("✓ Zero local storage usage")
    
    # Simple model
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 10)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop with in-memory batches
    print("\nStarting training...")
    
    num_epochs = 2
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)
        
        # Start prefetching
        batch_loader.start_prefetching()
        
        batch_idx = 0
        total_loss = 0.0
        
        try:
            while True:
                # Get batch from memory (prefetched in background)
                batch = batch_loader.get_batch()
                if batch is None:
                    break
                
                images, labels = batch
                
                # Training step
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 5 == 0:
                    print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")
                
                batch_idx += 1
        
        finally:
            batch_loader.stop()
        
        avg_loss = total_loss / batch_idx if batch_idx > 0 else 0.0
        print(f"\nEpoch {epoch + 1} complete: Average Loss = {avg_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


def simple_example():
    """Simple example showing in-memory file loading."""
    print("\n" + "=" * 60)
    print("Simple In-Memory File Loading Example")
    print("=" * 60)
    
    hub = DatasetHubClient(HUB_URL, API_KEY, cache_dir="./cache")
    manifest = hub.get_manifest(DATASET_NAME)
    
    # Get first image file
    image_files = [
        item["path"] for item in manifest[:5]
        if item["path"].lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    
    if not image_files:
        print("No image files found!")
        return
    
    print(f"Loading {len(image_files)} files into memory...")
    
    for subpath in image_files:
        # Get file bytes directly (no disk write!)
        file_bytes = hub.get_file_bytes(DATASET_NAME, subpath)
        
        # Process in memory
        image = Image.open(io.BytesIO(file_bytes))
        print(f"{subpath}: {image.size}, {len(file_bytes)} bytes")
    
    print("\nAll files processed in memory - no disk writes!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="In-Memory Training Example")
    parser.add_argument("--simple", action="store_true", 
                       help="Run simple example instead of full training")
    args = parser.parse_args()
    
    if args.simple:
        simple_example()
    else:
        train_with_in_memory_batches()

