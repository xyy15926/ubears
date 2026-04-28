#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: storage.py
#   Author: xyy15926
#   Created: 2026-04-22 10:01:20
#   Updated: 2026-04-23 09:26:26
#   Description:
# ---------------------------------------------------------

# %%
import logging
from typing import(
    Dict, List, Tuple, Callable,
    Any, Self, Type, Protocol, Optional,
)
from pathlib import Path
import os
import json
import tempfile
import threading
import dataclasses
from datetime import datetime

logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")

DEFAULT_EXPIRE = datetime(2099, 12, 12).isoformat()


# %%
class StorageBackend(Protocol):
    """Storage backend protocol."""
    def get(self, key: str) -> Optional[bytes]: ...
    def set(self, key: str, data: bytes, metadata: dict) -> None: ...
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...
    def list_keys(self, prefix: str = "") -> list[str]: ...


# %%
@dataclasses.dataclass
class MetaData:
    key: str
    size: int
    type_: str = "pickle"
    hit_count: int = 0
    created_at: datetime = dataclasses.field(
        default_factory = lambda: datetime.now().isoformat()
    )
    expires_at: datetime = dataclasses.field(
        default_factory = lambda: DEFAULT_EXPIRE
    )
    last_accessed: datetime = dataclasses.field(
        default_factory = lambda: datetime.now().isoformat()
    )

    def to_json(self) -> str:
        return json.dumps({
            "key": self.key,
            "type_": self.type_,
            "size": self.size,
            "hit_count": self.hit_count,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
        })


# %%
class LocalFileStorage:
    """Storage with local filesystem.

    1. Both meta and data are stored with file. But it may be better to
      store meta with some lite DB.
    """
    def __init__(
        self,
        base_dir: str = "./checkpoints",
        max_inline_size: int = 64 * 1024,
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir = self.base_dir / "meta"
        self.data_dir = self.base_dir / "data"
        self.meta_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)

        self.max_inline_size = max_inline_size
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
    
    def _get_lock(self, key: str) -> threading.Lock:
        """Lock for different keys.

        TODO:
        Thread lock is used here to handle racing among multithread, while
        file lock may be better to handle racing among multiprocess.
        """
        with self._global_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]
    
    def _meta_path(self, key: str) -> Path:
        """Meta file path."""
        return self.meta_dir / f"{key}.json"
    
    def _data_path(self, key: str) -> Path:
        """Data file path."""
        return self.data_dir / f"{key}.bin"

    def get(self, key: str) -> Optional[Tuple[Dict, bytes]]:
        """Read data bytes and meta data."""
        meta_path = self._meta_path(key)
        if not meta_path.exists():
            return None
        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Check the expire.
        if meta.get("expires_at"):
            expires = datetime.fromisoformat(meta["expires_at"])
            if datetime.now() > expires:
                self.delete(key)
                return None

        # Read data.
        if meta.get("inline"):
            logger.info(f"Load inline cache from {meta_path}.")
            return bytes.fromhex(meta["data"])
        else:
            data_path = self._data_path(key)
            if not data_path.exists():
                return None
            with open(data_path, "rb") as f:
                logger.info(f"Load cache from {data_path}.")
                return f.read()

    def set(self, key: str, data: bytes, meta: dict = None) -> None:
        """Write data.

        TODO:
        Maybe date-order-mark could be used to keep multiple version of
        data.
        """
        # Prepare the metadata.
        record = MetaData(
            key = key,
            size = len(data),
        )
        meta_dict = dataclasses.asdict(record)
        if meta is not None:
            meta_dict.update(meta)
        is_inline = len(data) <= self.max_inline_size
        meta_dict["inline"] = is_inline
        if is_inline:
            meta_dict["data"] = data.hex()

        # Lock prevent multithreads access the file at the same time.
        with self._get_lock(key):
            # Write meta to temporary file.
            tmp_meta = tempfile.NamedTemporaryFile(
                mode="w", dir=self.meta_dir, delete=False
            )
            json.dump(meta_dict, tmp_meta)
            tmp_meta.flush()
            os.fsync(tmp_meta.fileno())
            tmp_meta.close()

            # Write data to temporary file.
            tmp_data = None
            if not is_inline:
                tmp_data = tempfile.NamedTemporaryFile(
                    dir=self.data_dir, delete=False
                )
                tmp_data.write(data)
                tmp_data.flush()
                os.fsync(tmp_data.fileno())
                tmp_data.close()

            # Replace the temperary file's name atomically by OS so that
            # errors in writing to the tempfile won't ruin the original
            # cache.
            meta_path = self._meta_path(key)
            os.replace(tmp_meta.name, meta_path)
            if not is_inline:
                data_path = self._data_path(key)
                os.replace(tmp_data.name, data_path)
                logger.info(f"Save cache at {data_path}.")
            else:
                logger.info(f"Save inline cache at {meta_path}.")

    def exists(self, key: str) -> bool:
        """Check if data exists."""
        return self._meta_path(key).exists()

    def delete(self, key: str) -> None:
        """Deleta data."""
        meta_path = self._meta_path(key)
        data_path = self._data_path(key)
        if meta_path.exists():
            meta_path.unlink()
        if data_path.exists():
            data_path.unlink()
        logger.info(f"Delete cache related to key: {key}.")

    def list_keys(self, prefix: str = "") -> list[str]:
        """List data with key startswith specified prefix."""
        keys = []
        for f in self.meta_dir.glob("*.json"):
            key = f.stem
            if key.startswith(prefix):
                keys.append(key)
        return keys
