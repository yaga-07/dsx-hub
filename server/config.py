"""Configuration management for DSX Hub server."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


def load_env_file(env_path: Optional[Path] = None) -> None:
    """
    Load environment variables from .env file.
    
    Looks for .env file in the following order:
    1. Provided env_path if given
    2. .env in the project root (parent of server directory)
    3. .env in the current working directory
    
    Args:
        env_path: Optional path to .env file. If None, searches for .env automatically.
    """
    if env_path and env_path.exists():
        load_dotenv(env_path, override=True)
        return
    
    # Try project root (parent of server directory)
    server_dir = Path(__file__).parent
    project_root = server_dir.parent
    env_file = project_root / ".env"
    
    if env_file.exists():
        load_dotenv(env_file, override=True)
        return
    
    # Try current working directory
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        load_dotenv(cwd_env, override=True)
        return


def get_env(key: str, default: Optional[str] = None) -> str:
    """
    Get environment variable.
    
    The .env file is automatically loaded when this module is imported.
    This function simply retrieves the environment variable value.
    
    Args:
        key: Environment variable name.
        default: Default value if environment variable is not set.
    
    Returns:
        Environment variable value or default.
    """
    return os.getenv(key, default)


# Load .env file when module is imported
# This ensures environment variables are available for all imports
load_env_file()

