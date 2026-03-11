"""
Common MLflow utilities for artifact management.

This module provides shared utilities for saving and managing MLflow artifacts
across different pipeline components.
"""

import json
from pathlib import Path
from typing import Any, Union


def save_artifact(artifacts_dir: Union[Path, str], filename: str, data: Any) -> Path:
    """Save data as an artifact file and return its path."""
    # Convert to Path if string
    if isinstance(artifacts_dir, str):
        artifacts_dir = Path(artifacts_dir)

    # Ensure directory exists
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    file_path = artifacts_dir / filename

    # Handle pandas DataFrame
    def save_artifact(
        artifacts_dir: Union[Path, str], filename: str, data: Any
    ) -> Path:
        """Save data as an artifact file and return its path."""

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    file_path = artifacts_dir / filename

    if _is_dataframe(data):
        _save_dataframe(file_path, data)
    elif isinstance(data, bytes):
        _save_binary(file_path, data)
    elif isinstance(data, str):
        _save_text(file_path, data)
    elif isinstance(data, (dict, list, int, float, bool, type(None))):
        _save_json(file_path, data)
    else:
        _save_json_fallback(file_path, data)

    return file_path


def _is_dataframe(data: Any) -> bool:
    """Check if data is a pandas DataFrame without importing pandas globally."""
    try:
        import pandas as pd

        return isinstance(data, pd.DataFrame)
    except ImportError:
        return False


def _save_dataframe(file_path: Path, data: Any) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        data.to_json(f, orient="records", indent=2, force_ascii=False)


def _save_binary(file_path: Path, data: bytes) -> None:
    with open(file_path, "wb") as f:
        f.write(data)


def _save_text(file_path: Path, data: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(data)


def _save_json(
    file_path: Path, data: Union[dict, list, int, float, bool, None]
) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _save_json_fallback(file_path: Path, data: Any) -> Path:
    """Attempt to serialize arbitrary objects as JSON with string fallback."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"Unsupported data type: {type(data).__name__}. "
            f"Supported types: dict, list, DataFrame, str, bytes, or JSON-serializable objects."
        ) from e

    # Handle different data types
    if isinstance(data, bytes):
        # Binary data
        with open(file_path, "wb") as f:
            f.write(data)
    elif isinstance(data, str):
        # Plain text
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(data)
    elif isinstance(data, (dict, list, int, float, bool, type(None))):
        # JSON-serializable objects
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        # Try to serialize as JSON (might work for custom objects with __dict__)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"Unsupported data type: {type(data).__name__}. "
                f"Supported types: dict, list, DataFrame, str, bytes, or JSON-serializable objects."
            ) from e

    return file_path
