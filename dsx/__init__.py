"""
DSX Hub SDK - Client library for accessing DSX Hub datasets.
"""

from .client import DatasetHubClient
from .models import ManifestEntry

__version__ = "0.1.0"
__all__ = ["DatasetHubClient", "ManifestEntry"]

