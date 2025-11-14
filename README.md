# DSX Hub

A lightweight framework for serving datasets from a server and accessing them via SDK for training on remote devices.

## Overview

Host datasets on a server PC (with storage leverage) and use the SDK to stream or download data for training models on Mac or other devices.

---

## Installation

### SDK Only (Client Library)

```bash
pip install -e .
```

**Dependencies:** `requests`, `pydantic`

### SDK + Server

```bash
pip install -e .[server]
```

**Additional dependencies:** `fastapi`, `uvicorn`, `python-multipart`, `python-dotenv`

### Using .env File

Create a `.env` file in the project root:

```bash
DSX_DATASET_PATH=/path/to/your/datasets
DSX_PORT=8000
DSX_API_KEY=your-secret-key
```

---

## Server Setup

### Configuration

Set the following environment variables:

- **`DSX_API_KEY`** (required) - API key for authentication
- **`DSX_DATASET_PATH`** (required) - Path to your datasets directory
- **`DSX_PORT`** (optional) - Server port, defaults to `8000`

### Running the Server

```bash
export DSX_API_KEY="your-secret-key-here"
export DSX_DATASET_PATH="/path/to/datasets"
python server/server.py
```

The server will start on `http://0.0.0.0:8000`. Visit `http://localhost:8000/docs` for API documentation.

### Server Endpoints

| Endpoint | Method | Auth Required | Description |
|----------|--------|---------------|-------------|
| `/datasets` | GET | Yes | List all available datasets |
| `/manifest/<dataset>` | GET | Yes | Get manifest for a dataset |
| `/file/<dataset>/<path>` | GET | Yes | Download a file |
| `/health` | GET | No | Health check endpoint |
| `/docs` | GET | No | Interactive API documentation |

All authenticated endpoints require header: `Authorization: Bearer <api_key>`

---

## SDK Usage

### Initialize the Client

```python
from dsx import DatasetHubClient

hub = DatasetHubClient(
    base_url="http://localhost:8000",  # Server URL
    api_key="my-secret-key",            # API key
    cache_dir="./cache"                 # Local cache directory (optional, set to None to disable)
)
```

### List Available Datasets

```python
datasets = hub.list_datasets()
print(datasets)
# Output: ['imagenet', 'cifar10', 'mnist']
```

### Get Dataset Manifest

The manifest contains metadata about all files in a dataset. It's cached locally if `cache_dir` is set.

```python
manifest = hub.get_manifest("imagenet")
print(f"Dataset has {len(manifest)} files")

# Each item in manifest contains:
# {
#     "path": "train/img_001.jpg",
#     "size": 123456,
#     "hash": "abc123..."
# }
```

### Download Files

#### Option 1: Download to Disk Cache

```python
local_path = hub.get_file("imagenet", "train/img_001.jpg")
print(f"File saved to: {local_path}")
# Output: PosixPath('./cache/imagenet/train/img_001.jpg')
```

Files are cached locally. If the file already exists in cache, it returns immediately without downloading.

#### Option 2: Stream into Memory

```python
# Get file bytes directly (no disk write)
file_bytes = hub.get_file_bytes("imagenet", "train/img_001.jpg")

# Use with PIL
from PIL import Image
import io
image = Image.open(io.BytesIO(file_bytes))
```

### Download Multiple Files

```python
# Download specific files
file_list = [
    "train/img_001.jpg",
    "train/img_002.jpg",
    "train/img_003.jpg"
]
downloaded_paths = hub.download_dataset("imagenet", file_list=file_list)

# Or download entire dataset
downloaded_paths = hub.download_dataset("imagenet")
print(f"Downloaded {len(downloaded_paths)} files")
```

---

## API Reference

### `DatasetHubClient`

#### Constructor

```python
DatasetHubClient(base_url, api_key, cache_dir="./cache")
```

#### Methods

##### `list_datasets() -> List[str]`

List all available datasets on the server.

**Returns:** List of dataset names

##### `get_manifest(dataset: str) -> List[Dict]`

Get manifest for a dataset. Manifest is cached locally if `cache_dir` is set.

**Parameters:**
- `dataset` (str): Name of the dataset

**Returns:** List of dictionaries, each containing:
- `path` (str): File path relative to dataset root
- `size` (int): File size in bytes
- `hash` (str): MD5 hash of first 1024 bytes

##### `get_file(dataset: str, subpath: str) -> Path`

Download a file from a dataset. File is cached locally.

**Parameters:**
- `dataset` (str): Name of the dataset
- `subpath` (str): Path to file relative to dataset root

**Returns:** `Path` object pointing to local cached file

**Raises:** `ValueError` if `cache_dir` is `None`

##### `get_file_bytes(dataset: str, subpath: str) -> bytes`

Get file content as bytes directly from server (no disk write).

**Parameters:**
- `dataset` (str): Name of the dataset
- `subpath` (str): Path to file relative to dataset root

**Returns:** `bytes` - File content as bytes

##### `download_dataset(dataset: str, file_list: Optional[List[str]] = None) -> List[Path]`

Download multiple files from a dataset.

**Parameters:**
- `dataset` (str): Name of the dataset
- `file_list` (List[str], optional): List of file paths to download. If `None`, downloads all files from manifest.

**Returns:** List of `Path` objects pointing to downloaded files

##### `clear_cache(dataset: Optional[str] = None) -> int`

Clear cache for a dataset or all datasets.

**Parameters:**
- `dataset` (str, optional): Dataset name to clear cache for. If `None`, clears all cache.

**Returns:** Number of bytes freed

##### `get_cache_size(dataset: Optional[str] = None) -> int`

Get size of cache in bytes.

**Parameters:**
- `dataset` (str, optional): Dataset name. If `None`, returns total cache size.

**Returns:** Size in bytes

---

## Examples

### Example 1: Filtering Files by Extension

```python
from dsx import DatasetHubClient

hub = DatasetHubClient("http://localhost:8000", "my-key")

# Get manifest
manifest = hub.get_manifest("imagenet")

# Filter for image files only
image_files = [
    item["path"] for item in manifest 
    if item["path"].lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"Found {len(image_files)} image files")

# Download only images
hub.download_dataset("imagenet", file_list=image_files)
```

### Example 2: In-Memory File Loading

```python
from dsx import DatasetHubClient
import io
from PIL import Image

hub = DatasetHubClient("http://localhost:8000", "my-key", cache_dir=None)
manifest = hub.get_manifest("imagenet")

# Get file bytes directly (no disk write)
for item in manifest[:10]:  # First 10 files
    file_bytes = hub.get_file_bytes("imagenet", item["path"])
    
    # Process in memory
    image = Image.open(io.BytesIO(file_bytes))
    print(f"Loaded {item['path']}: {image.size}, {len(file_bytes)} bytes")
```

### Example 3: Cache Management

```python
from dsx import DatasetHubClient

hub = DatasetHubClient("http://localhost:8000", "my-key", cache_dir="./cache")

# Check cache size
size_mb = hub.get_cache_size() / (1024**2)
print(f"Cache using {size_mb:.2f} MB")

# Clear specific dataset cache
bytes_freed = hub.clear_cache("imagenet")
print(f"Freed {bytes_freed / (1024**2):.2f} MB")

# Clear all cache
hub.clear_cache()
```

---

## Configuration

### Server Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DSX_API_KEY` | Yes | - | API key for authentication |
| `DSX_DATASET_PATH` | Yes | - | Path to datasets directory |
| `DSX_PORT` | No | `8000` | Server port |

### Client Configuration

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `base_url` | Yes | - | Server URL (e.g., `http://localhost:8000`) |
| `api_key` | Yes | - | API key matching server's `DSX_API_KEY` |
| `cache_dir` | No | `"./cache"` | Local cache directory. Set to `None` to disable caching |

### Cache Directory Structure

```
cache/
├── imagenet_manifest.json          # Cached manifest
├── imagenet/
│   ├── train/
│   │   ├── img_001.jpg
│   │   └── img_002.jpg
│   └── val/
│       └── img_001.jpg
└── cifar10_manifest.json
```

---

## Troubleshooting

**Connection Error**
- Make sure the server is running
- Check that `base_url` is correct

**Unauthorized Error**
- Verify `api_key` matches server's `DSX_API_KEY`
- Check that `Authorization` header is being sent

**Dataset Not Found**
- Verify dataset name exists on server
- Check `DSX_DATASET_PATH` on server

**File Not Found**
- Verify file path is correct
- Check that file exists in dataset directory

---

## License

MIT License
