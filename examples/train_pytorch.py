"""
Example: Training PyTorch model with DSX Hub using in-memory streaming.

This example demonstrates how to train a PyTorch model using data streamed
directly into memory from the server, without saving any files to disk.
Perfect for systems with limited storage or when you want zero disk usage.
"""

import io
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import os
import sys
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


class InMemoryImageDataset(Dataset):
    """PyTorch Dataset that loads images directly into memory (no disk writes)."""
    
    def __init__(self, hub_client, dataset_name, file_list, transform=None):
        """
        Initialize in-memory dataset.
        
        Args:
            hub_client: DatasetHubClient instance
            dataset_name: Name of the dataset
            file_list: List of file paths relative to dataset root
            transform: Optional image transform (torchvision.transforms)
        """
        self.hub_client = hub_client
        self.dataset_name = dataset_name
        self.file_list = file_list
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """Load image directly into memory from server."""
        subpath = self.file_list[idx]
        
        # Get file bytes directly from server (no disk write!)
        file_bytes = self.hub_client.get_file_bytes(self.dataset_name, subpath)
        
        # Load image from bytes in memory
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        
        # Apply transform if provided
        if self.transform:
            image = self.transform(image)
        
        # Dummy label (replace with actual label extraction from path or metadata)
        label = 0
        
        return image, label


def train_model():
    """Main training loop with in-memory data loading."""
    print("=" * 60)
    print("PyTorch Training with In-Memory Data Streaming")
    print("=" * 60)
    print("Data is streamed directly into memory - zero disk usage!")
    print()
    
    # Initialize hub client (cache_dir=None disables disk caching)
    # You can also set cache_dir="./cache" if you want manifest caching
    hub = DatasetHubClient(HUB_URL, API_KEY, cache_dir=None)
    
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
    
    # Limit dataset size for demo (remove this for full training)
    image_files = image_files[:100]
    print(f"Using {len(image_files)} images for training")
    
    # Define image transforms
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create in-memory dataset
    dataset = InMemoryImageDataset(
        hub_client=hub,
        dataset_name=DATASET_NAME,
        file_list=image_files,
        transform=transform
    )
    
    # Create dataloader
    # Note: num_workers > 0 can help with parallel loading, but each worker
    # will make separate network requests. Adjust based on your needs.
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 0 for single-threaded, or >0 for parallel loading
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    print(f"Batch size: {BATCH_SIZE}")
    print()
    
    # Simple model (example)
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 10)
    )
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("Starting training...")
    print("-" * 60)
    
    num_epochs = 2
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            # Move batch to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print progress
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = running_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} complete - Average Loss: {avg_loss:.4f}")
        print("-" * 60)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print("\nKey benefits:")
    print("✓ Zero disk usage - all data streamed into memory")
    print("✓ No cache management needed")
    print("✓ Perfect for systems with limited storage")
    print("✓ Works seamlessly with PyTorch DataLoader")


def example_simple_usage():
    """Simple example showing basic in-memory dataset usage."""
    print("\n" + "=" * 60)
    print("Simple In-Memory Dataset Example")
    print("=" * 60)
    
    # Initialize client (no disk caching)
    hub = DatasetHubClient(HUB_URL, API_KEY, cache_dir=None)
    
    # Get manifest
    manifest = hub.get_manifest(DATASET_NAME)
    image_files = [
        item["path"] for item in manifest[:10]  # First 10 images
        if item["path"].lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    
    # Create dataset
    dataset = InMemoryImageDataset(hub, DATASET_NAME, image_files)
    
    # Use with DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    print("\nLoading batches...")
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}: images shape = {images.shape}, labels shape = {labels.shape}")
        # All data is in memory - no disk reads!


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PyTorch Training with In-Memory Data")
    parser.add_argument("--simple", action="store_true",
                       help="Run simple example instead of full training")
    args = parser.parse_args()
    
    if args.simple:
        example_simple_usage()
    else:
        train_model()
