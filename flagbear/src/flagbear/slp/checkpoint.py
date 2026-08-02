#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: checkpointer.py
#   Author: xyy15926
#   Created: 2026-04-21 19:53:32
#   Updated: 2026-05-04 21:52:31
#   Description:
# ---------------------------------------------------------

# %%
import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from typing import Any

if __name__ == "__main__":
    from importlib import reload

    from flagbear.slp import cache, finer, serializer, storage
    reload(finer)
    reload(serializer)
    reload(storage)
    reload(cache)
from flagbear.slp.cache import Cache, CacheMeta, CachePolicy, PersistentCache
from flagbear.slp.finer import date_order_mark, get_tmp_path
from flagbear.slp.storage import LocalFileStorage

logger = logging.getLogger(__name__)

DEFAULT_EXPIRE = datetime(2099, 12, 12).isoformat()


# %%
def json_args(
    args: list[Any],
    kwargs: dict[str, Any],
    skip_args: list = None,
) -> str:
    """Dumps arguments into JSON string.

    Only the int, float and str argument will be used to construct the name
    that will be used as the key or identifier.

    Params:
    -----------------------------
    args: List of positional arguments.
    kwargs: Dict of keywords arguments.
    skip_args: Index of positional arguments or key or keywords arguments
      that should be skipped before hashing.

    Return:
    -----------------------------
    String as the key or identifier.
    """
    if skip_args:
        args = tuple(a for i, a in enumerate(args) if i not in skip_args)
        kwargs = {k: v for k, v in kwargs.items() if k not in skip_args}
    try:
        arg_str = json.dumps((args, kwargs), sort_keys=True, default=str)
    except (TypeError, ValueError):
        arg_str = str((args, kwargs))
    arg_str = hashlib.md5(arg_str.encode("utf8")).hexdigest()[:16]  # noqa: S324
    return arg_str


# %%
@dataclass
class CheckpointPolicy:
    """Checkpoint policy to determine the behavior the checkpoint.

    Attrs:
    --------------------------
    mode: How to generate checkpoint.
      cache: Try to fetch cache for function called with the same arguments,
        and the older version checkpoint will be overwritten if function
        with the same arguments really processed.
      record: Call the function every time and save the result in checkpoint,
        even for the same arguments.
      manual: Generate checkpoints for a function with the same arguments
        only when function really processed, namely no cache exists or `force`
        flag is passed, and older version checkpoint won't be deleted.
    keyskip_args: Arguments to skip when generating the key.
    keyonly_args: Arguments for generating key only, which will be dropped
      before passed to the function.
    arg_func: Function to generate unique key with arguments.
    """
    mode: str | None = "cache"
    keyskip_args: list[int | str] | None = None
    keyonly_args: list[str] | None = None
    arg_func: Callable = json_args

    def gen_key(
        self,
        func: Callable | str,
        args: list[Any],
        kwargs: dict[str, Any],
    ):
        """Generate cache key from function and arguments."""
        key = (f"{func.__module__}.{func.__qualname__}"
               if callable(func)
               else func)
        arg_str = self.arg_func(args, kwargs, self.keyskip_args)
        return f"{key}_{arg_str}"


# %%
class CheckpointManager:
    """Checkpointer manager.

    CheckpointManager only generate the cache key for a single call to a
    function with some arguments, and the cache things are done by the
    inner cache.

    Attrs:
    ------------------------------
    checkpoint_policy: Checkpoint policy by default.
    ttl: Time for checkpoint to be valid.
    cache: Inner cache.
    """
    def __init__(
        self,
        checkpoint_policy: CheckpointPolicy | None = None,
        ttl: timedelta | None = None,
        cache: Cache | None = None,
    ):
        self.checkpoint_policy = checkpoint_policy or CheckpointPolicy("cache")
        if cache is not None:
            self.cache = cache
        else:
            storage = LocalFileStorage(get_tmp_path("checkpoint"))
            self.cache = PersistentCache(storage, ttl = ttl)

    def checkpoint(
        self,
        func: Callable | None = None,
        *,
        name: str | None = None,
        checkpoint_policy: CheckpointPolicy | None = None,
        cache_policy: CachePolicy | None = None,
    ) -> Callable:
        """Checkpoint decorator to cache functions' result.

        Params:
        -------------------------------
        func: Function to be decorated.
        checkpoint_policy: Checkpoint policy to determine how to generate
          checkpoint.
        cache_policy: Cache policy to determine the behavior of inner cache.
        """
        def decorator(ffunc: Callable) -> Callable:
            nonlocal checkpoint_policy, cache_policy, self, name
            checkpoint_policy = checkpoint_policy or self.checkpoint_policy
            cache_policy = cache_policy or CachePolicy()
            @wraps(ffunc)
            def wrapper(*args, **kwargs) -> Any:
                nonlocal ffunc, checkpoint_policy, cache_policy, name
                # Pop the keyword arugment `force` out.
                is_forced = kwargs.pop("force", None)
                mode = checkpoint_policy.mode
                if mode == "record":
                    is_forced = True
                name = name or ffunc

                # Try use cache.
                if not is_forced:
                    result = self.get(name, args, kwargs, checkpoint_policy)
                    if result is not None:
                        return result

                # Roll back to call `func` to get the result.
                result = self.set(name, args, kwargs, checkpoint_policy, cache_policy)

                return result
            return wrapper

        # `@checkpoint`: decorate the `func` directly with the default setting
        # `@checkpoint()`: return the inner decorator with customed setting
        return decorator(func) if func else decorator

    def get(
        self,
        func: Callable | str,
        args: Any,
        kwargs: Any,
        checkpoint_policy: CheckpointPolicy | None = None,
    ) -> Any:
        """Get inner cache.

        Params:
        -----------------------------
        func: Callable.
        args: List of positional arguments.
        kwargs: Dict of keywords arguments.
        checkpoint_policy: Checkpoint policy to determine how to generate
          checkpoint.
        cache_policy: Cache policy to determine the behavior of inner cache.

        Return:
        -----------------------------
        Cached result.
        """
        checkpoint_policy = checkpoint_policy or CheckpointPolicy()
        key = checkpoint_policy.gen_key(func, args, kwargs)
        mode = checkpoint_policy.mode
        if mode == "cache":
            return self.cache.get(key)
        elif mode in ("record", "manual"):
            existed_keys = self.cache.list_keys(key)
            if len(existed_keys) == 0:
                return None
            else:
                key = date_order_mark(key, existed_keys, None, 0)
                return self.cache.get(key)

    def set(
        self,
        func: Callable | str,
        args: list[Any],
        kwargs: dict[str, Any],
        checkpoint_policy: CheckpointPolicy | None = None,
        cache_policy: CachePolicy | None = None,
    ) -> Any:
        """Process the function and set inner cache.

        Params:
        -----------------------------
        func: Callable.
        args: List of positional arguments.
        kwargs: Dict of keywords arguments.
        checkpoint_policy: Checkpoint policy to determine how to generate
          checkpoint.
        cache_policy: Cache policy to determine the behavior of inner cache.

        Return:
        -----------------------------
        Function result.
        """
        checkpoint_policy = checkpoint_policy or CheckpointPolicy()
        key = checkpoint_policy.gen_key(func, args, kwargs)
        cache_policy = cache_policy or CachePolicy()
        cache_meta = CacheMeta.from_meta(cache_policy, key)
        mode = checkpoint_policy.mode
        result = func(*args, **kwargs)
        if mode == "cache":
            self.cache.set(key, result, cache_meta)
        elif mode in ("record", "manual"):
            existed_keys = self.cache.list_keys(key)
            key = date_order_mark(key, existed_keys, "today", 1)
            self.cache.set(key, result, cache_meta)

        return result

    def clear_func_cache(self, func: Callable | str):
        """Clear function cache."""
        prefix = (f"{func.__module__}.{func.__qualname__}"
                  if callable(func)
                  else func)
        for key in self.cache.list_keys(prefix):
            self.cache.delete(key)


# %%
_global_checkpointe_manager = CheckpointManager()
checkpoint = _global_checkpointe_manager.checkpoint
