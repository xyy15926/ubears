#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: storage.py
#   Author: xyy15926
#   Created: 2026-04-22 10:01:20
#   Updated: 2026-05-02 15:07:08
#   Description:
# ---------------------------------------------------------

# %%
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


# %%
class StorageBackend(Protocol):
    """Storage backend protocol."""

    def get(self, key: str) -> bytes | None:
        """Get data by key."""
        ...

    def set(self, key: str, data: bytes) -> None:
        """Set data by key."""
        ...

    def exists(self, key: str) -> bool:
        """Check if data exists."""
        ...

    def delete(self, key: str) -> None:
        """Delete data by key."""
        ...

    def list_keys(self, prefix: str = "") -> list[str]:
        """List keys with given prefix."""
        ...


# %%
class LocalFileStorage:
    """Storage with local filesystem.

    1. Both meta and data are stored with file. But it may be better to
      store meta with some lite DB.
    """

    def __init__(
        self,
        base_dir: str = "./cache",
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def _get_lock(self, key: str) -> threading.Lock:
        """Lock for different keys.

        Todo:
        Thread lock is used here to handle racing among multithread, while
        file lock may be better to handle racing among multiprocess.
        """
        with self._global_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def _data_path(self, key: str) -> Path:
        """Get data path."""
        return self.base_dir / f"{key}.bin"

    def get(self, key: str) -> bytes | None:
        """Read data bytes and data."""
        data_path = self._data_path(key)
        if not data_path.exists():
            return None
        with open(data_path, "rb") as f:
            logger.info(f"Load bytes from {data_path}.")
            return f.read()

    def set(self, key: str, bytes_: bytes):
        """Write data."""
        # Lock prevent multithreads access the file at the same time.
        with self._get_lock(key):
            tmp_data = tempfile.NamedTemporaryFile(  # noqa: SIM115
                dir=self.base_dir, delete=False
            )
            tmp_data.write(bytes_)
            tmp_data.flush()
            os.fsync(tmp_data.fileno())
            tmp_data.close()

            # Replace the temporary file's name atomically by OS so that
            # errors in writing to the tempfile won't ruin the original
            # cache.
            data_path = self._data_path(key)
            os.replace(tmp_data.name, data_path)
            logger.info(f"Save cache at {data_path}.")

    def exists(self, key: str) -> bool:
        """Check if data exists."""
        return self._data_path(key).exists()

    def delete(self, key: str) -> None:
        """Delete data."""
        data_path = self._data_path(key)
        if data_path.exists():
            data_path.unlink()
        logger.info(f"Delete cache related to key: {key}.")

    def list_keys(self, prefix: str = "") -> list[str]:
        """List data with key startswith specified prefix."""
        keys = []
        for f in self.base_dir.glob("*.bin"):
            key = f.stem
            if key.startswith(prefix):
                keys.append(key)
        return keys
