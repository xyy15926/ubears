#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: cache.py
#   Author: xyy15926
#   Created: 2026-05-01 14:42:56
#   Updated: 2026-05-20 15:58:10
#   Description:
# ---------------------------------------------------------

# %%
import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol, Self, TypeAlias

# from IPython.core.debugger import set_trace

if __name__ == "__main__":
    from importlib import reload

    from flagbear.slp import finer, serializer, storage
    reload(finer)
    reload(storage)
    reload(serializer)
from flagbear.slp.serializer import deserialize, serialize
from flagbear.slp.storage import LocalFileStorage, StorageBackend

logger = logging.getLogger(__name__)

CacheMetaDict: TypeAlias = "CacheMeta | dict | None"


# %%
@dataclass
class CachePolicy:
    """Cache policy to determine the behavior the cache.

    Attrs:
    -------------------------
    ttl: Time to live.
    type_: Serialization type.
      Only Valid for persistent cache only.
    inline: If to store cache in memory.
      Only Valid for persistent cache only.
    """
    ttl: timedelta | None = None
    type_: str | None = None
    inline: bool | None = None


@dataclass
class CacheMeta:
    """Metadata of the cache.

    Attrs:
    -------------------------
    key: Cache key.
    created_at: Time to create cache.
    expires_at: Time the cache will expire.
    last_accessed: Time the cache was accessed last time.
    hits: Count of the cache hit.
    size: The length of bytes of the serialization of the cache.
      Only Valid for persistent cache only.
    type_: Serialization type.
      Only Valid for persistent cache only.
    inline: If to store cache in memory.
      Only Valid for persistent cache only.
    value: The exact value of the cache.
      Only set when `inline == True`
    """
    key: str | None = None
    created_at: datetime = field(default_factory = datetime.now)
    expires_at: datetime | None = None
    last_accessed: datetime | None = None
    hits: int = 0
    size: int = 0
    type_: str | None = None
    inline: bool | None = None
    value: Any | None = None

    def to_json(self) -> bytes:
        """Dump into json string."""
        metadict = {
            "key": self.key,
            "created_at": (self.created_at.isoformat()
                           if self.created_at is not None else None),
            "expires_at": (self.expires_at.isoformat()
                           if self.expires_at is not None else None),
            "last_accessed": (self.last_accessed.isoformat()
                           if self.last_accessed is not None else None),
            "hits": self.hits,
            "size": self.size,
            "type_": self.type_,
            "inline": self.inline,
        }
        return json.dumps(metadict, ensure_ascii=False).encode("utf8")

    @classmethod
    def from_json(cls, bytes_: bytes) -> Self:
        """Load from json string."""
        metadict = json.loads(bytes_.decode("utf8"))
        for d in ["created_at", "expires_at", "last_accessed"]:
            if metadict[d] is not None:
                metadict[d] = datetime.fromisoformat(metadict[d])
        return cls(**metadict)

    @classmethod
    def from_meta(
        cls,
        meta: CacheMetaDict,
        key: str | None = None,
    ) -> Self:
        """Create CacheMeta from dict, None, CacheMeta, CachePolicy passed."""
        if meta is None:
            return cls(key)
        if isinstance(meta, dict):
            return cls(**meta)
        if isinstance(meta, cls):
            if meta.key is None and key is not None:
                meta.key = key
            return meta
        if isinstance(meta, CachePolicy):
            policy = meta
            meta = cls(key)
            if policy.ttl is not None:
                meta.expires_at = meta.created_at + policy.ttl
            meta.type_ = policy.type_
            meta.inline = policy.inline
            return meta
        raise ValueError(f"Invalid metadata {meta}.")


# %%
class Cache(Protocol):
    """Cache protocol."""
    def get(self, key: str) -> Any:
        """Get data by key."""
        ...
    def set(self, key: str, data: Any, meta: CacheMetaDict) -> None:
        """Set data by key."""
        ...
    def exists(self, key: str) -> bool:
        """Check if cache exists."""
        ...
    def delete(self, key: str) -> None:
        """Delete cache by key."""
        ...
    def list_keys(self, prefix: str = "") -> list[str]:
        """List keys with given prefix."""
        ...


# %%
def meta_serialize(
    data: Any,
    meta: CacheMetaDict = None,
) -> tuple[bytes, CacheMeta]:
    """Serialize data into bytes with metadata."""
    meta = CacheMeta.from_meta(meta)

    # Use specified `type_` if passed.
    bytes_, type_ = serialize(data, meta.type_)
    meta.type_ = type_

    # Concat meta-length, meta and data.
    meta_bytes = meta.to_json()
    meta_size = len(meta_bytes).to_bytes(4, "big")
    bytes_ = meta_size + meta_bytes + bytes_

    return bytes_, meta


def meta_deserialize(bytes_: bytes) -> tuple[Any, CacheMeta]:
    """Deserialize bytes into data and metadata."""
    # Get metadata from data bytes.
    meta_size = int.from_bytes(bytes_[:4], "big")
    meta = CacheMeta.from_json(bytes_[4: 4 + meta_size])
    if len(bytes_) > 4 + meta_size:
        data = deserialize(
            bytes_[4 + meta_size:],
            meta.type_,
        )
        return data, meta
    return None, meta


# %%
class MemoryCache:
    """Cache in memory with dict.

    Attrs:
    --------------------------
    storage: Inner dict to store cache.
    ttl: Time to live by default.
    on_hit: Hook function to call when cache hits.
    on_miss: Hook function to call when cache misses.
    _stats: Global stats of all the cache.
    _lock: Lock for updating the cache for multi-thread situation.
    """
    def __init__(
        self,
        # TODO: CachePolicy???
        ttl: timedelta | None = None,
        on_hit: Callable[[str], None] | None = None,
        on_miss: Callable[[str], None] | None = None,
    ):
        """Init memory cache.

        Params:
        ---------------------------
        ttl: Time to live by default.
        on_hit: Hook function to call when cache hits.
        on_miss: Hook function to call when cache misses.
        """
        self.storage = dict()
        self.ttl = timedelta(days = 1e4) if ttl is None else ttl
        self.on_hit = on_hit
        self.on_miss = on_miss
        self._stats = {"hits": 0, "misses": 0, "errors": 0}
        self._lock = threading.Lock()

    def get(self, key: str, default: Any = None) -> Any:
        """Get cache with key."""
        meta = self.storage.get(key, None)

        # Cache missed.
        if meta is None:
            if self.on_miss:
                self.on_miss(key)
            with self._lock:
                self._stats["misses"] += 1
            return default
        if meta.expires_at is not None and meta.expires_at < datetime.now():
            if self.on_miss:
                self.on_miss(key)
            with self._lock:
                self._stats["misses"] += 1
            self.delete(key)
            return default

        # Cache hits.
        if self.on_hit:
            self.on_hit(key)
        with self._lock:
            self._stats["hits"] += 1
            meta.hits += 1
            meta.last_accessed = datetime.now()
        return meta.value

    def set(
        self,
        key: str,
        data: Any,
        meta: dict | CacheMeta | None = None,
    ):
        """Set cache with key, data and metadata."""
        # Prepare metadata.
        meta = CacheMeta.from_meta(meta, key)
        if meta.expires_at is None:
            meta.expires_at = meta.created_at + self.ttl
        meta.value = data
        with self._lock:
            self.storage[key] = meta

    def exists(self, key: str) -> bool:
        """Check if cache, expired cache included, exists."""
        return key in self.storage

    def delete(self, key: str):
        """Delete cache."""
        with self._lock:
            self.storage.pop(key, None)

    def list_keys(self, prefix: str = "") -> list[str]:
        """Delete keys starting with prefix."""
        return [key for key in self.storage.keys()
                if key.startswith(prefix)]


# %%
class PersistentCache:
    """Persistent cache with some storage backend.

    A dict will be used to store metadata in memory to for acceleration.

    Attrs:
    --------------------------
    storage: Persistent storage backend.
    meta_storage: Dict to store metadata.
    ttl: Time to live by default.
    on_hit: Hook function to call when cache hits.
    on_miss: Hook function to call when cache misses.
    _stats: Global stats of all the cache.
    _lock: Lock for updating the cache for multi-thread situation.
    max_mem_size: The maximum length of bytes of the serialization result of
      the data to store inline, in memory, with metadata by default.
    """
    def __init__(
        self,
        storage: StorageBackend | None = None,
        ttl: timedelta | None = None,
        on_hit: Callable[[str], None] | None = None,
        on_miss: Callable[[str], None] | None = None,
        max_mem_size: int = 64 * 1024,
    ):
        """Init persistent cache.

        Params:
        ---------------------------
        storage: Persistent storage backend.
        ttl: Time to live by default.
        on_hit: Hook function to call when cache hits.
        on_miss: Hook function to call when cache misses.
        max_mem_size: The maximum length of bytes of the serialization result of
          the data to store inline, in memory, with metadata by default.
        """
        self.storage = storage or LocalFileStorage("./cache")
        self.meta_storage = dict()
        self.ttl = timedelta(days = 1e4) if ttl is None else ttl
        self.on_hit = on_hit
        self.on_miss = on_miss
        self._stats = {"hits": 0, "misses": 0, "errors": 0}
        self._lock = threading.Lock()
        self.max_mem_size = max_mem_size
        self._init_from_storage()

    def get(self, key: str, default: Any = None) -> Any:
        """Get cache with key."""
        meta = self.meta_storage.get(key, None)
        data = None

        # No cache exists.
        if meta is None:
            if self.on_miss:
                self.on_miss(key)
            with self._lock:
                self._stats["misses"] += 1
            return default

        # Recover metadata from persistent storage.
        if meta.key is None:
            data, meta = self._from_storage(key)
            with self._lock:
                self.meta_storage[key] = meta

        # Cache expired.
        if meta.expires_at is not None and meta.expires_at < datetime.now():
            if self.on_miss:
                self.on_miss(key)
            with self._lock:
                self._stats["misses"] += 1
                self.meta_storage.pop(key)
            self.storage.delete(key)
            return default

        # Cache hits.
        if self.on_hit:
            self.on_hit(key)
        with self._lock:
            meta.hits += 1
            meta.last_accessed = datetime.now()
            self._stats["hits"] += 1

        if meta.inline:
            return meta.value
        # Load bytes from persistent storage if not loaded earlier.
        elif data is None:
            data, _meta = self._from_storage(key)

        return data

    def _from_storage(self, key: str) -> tuple[Any, dict]:
        """Recover metadata from persistent storage."""
        bytes_ = self.storage.get(key)
        if bytes_ is None:
            raise RuntimeError(f"Fail to get data bytes of key {key} from "
                               f"persistent storage.")
        data, meta = meta_deserialize(bytes_)
        # Store data in metadata(memory) too.
        if meta.inline is None:
            meta.inline = len(bytes_) <= self.max_mem_size
        if meta.inline:
            meta.value = data

        return data, meta

    def set(
        self,
        key: str,
        data: Any,
        meta: dict | CacheMeta | None = None,
    ):
        """Set cache with key, data and metadata."""
        # Prepare metadata.
        meta = CacheMeta.from_meta(meta, key)
        if meta.expires_at is None:
            meta.expires_at = meta.created_at + self.ttl
        bytes_, meta = meta_serialize(data, meta)
        meta.size = len(bytes_)

        # Store data in memory too.
        # Note:
        # `meta.inline` is set after `meta_serialize`, that the `meta.inline`
        # in `bytes_` won't be effected.
        if meta.inline is None:
            meta.inline = len(bytes_) <= self.max_mem_size
        if meta.inline:
            meta.value = data

        with self._lock:
            self.meta_storage[key] = meta
        self.storage.set(key, bytes_)

    def exists(self, key: str) -> bool:
        """Check if cache, expired cache included, exists."""
        return key in self.meta_storage

    def delete(self, key: str):
        """Delete cache."""
        with self._lock:
            self.meta_storage.pop(key, None)
        self.storage.delete(key)

    def list_keys(self, prefix: str = "") -> list[str]:
        """Delete keys starting with prefix."""
        return list(self.meta_storage.keys())

    def _init_from_storage(self):
        """Recover metadata from persistent storage."""
        for key in self.storage.list_keys():
            with self._lock:
                # Keep `key` unset so to indicate the meta in memory should be
                # loaded from persistent storage when necessary.
                self.meta_storage[key] = CacheMeta()
