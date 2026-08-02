#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: serializer.py
#   Author: xyy15926
#   Created: 2026-04-22 15:21:57
#   Updated: 2026-07-03 11:28:22
#   Description:
# ---------------------------------------------------------

# %%
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np  # noqa: TC004
    import pandas as pd  # noqa: TC004
import io
import json
import pickle
import threading
import zlib

# from IPython.core.debugger import set_trace

if __name__ == "__main__":
    from importlib import reload

    from flagbear.slp import finer, ser_exception, storage
    reload(finer)
    reload(storage)
    reload(ser_exception)

from flagbear.slp.ser_exception import (
    destr_exception,
    str_exception,
)

logger = logging.getLogger(__name__)

SerializerFn = Callable[[Any], bytes]
DeserializerFn = Callable[[bytes], Any]
CheckerFn = Callable[[Any], bool]

_strategy_map: dict[str, tuple[CheckerFn, SerializerFn, DeserializerFn]] = {}
_priority_order: list[str] = []
_registration_lock = threading.Lock()

JSON_MAX = 32 * 1024
NUMPY_COMPRESSS_MIN = 8 * 1024 * 1024
DF_CSV_MAX = 32 * 1024
DF_FEATHER_MAX = 8 * 1024 * 1024


# %%
def checker(type_: str, priority: int = 50):
    """Decorator to regist checker function."""
    def decorator(fn: CheckerFn) -> CheckerFn:
        with _registration_lock:
            if type_ not in _strategy_map:
                _strategy_map[type_] = (fn, None, None)
                _priority_order.append((priority, type_))
                _priority_order.sort(reverse=True)
            else:
                _, ser, deser = _strategy_map[type_]
                _strategy_map[type_] = (fn, ser, deser)
        return fn
    return decorator


def serializer(type_: str):
    """Decorator to regist serializer function."""
    def decorator(fn: SerializerFn) -> SerializerFn:
        with _registration_lock:
            if type_ not in _strategy_map:
                _strategy_map[type_] = (None, fn, None)
            else:
                checker, _, deser = _strategy_map[type_]
                _strategy_map[type_] = (checker, fn, deser)
        return fn
    return decorator


def deserializer(type_: str):
    """Decorator to regist deserializer function."""
    def decorator(fn: DeserializerFn) -> DeserializerFn:
        with _registration_lock:
            if type_ not in _strategy_map:
                _strategy_map[type_] = (None, None, fn)
            else:
                checker, ser, _ = _strategy_map[type_]
                _strategy_map[type_] = (checker, ser, fn)
        return fn
    return decorator


# %%
def serialize(
    obj: Any,
    type_: str | None = None,
) -> tuple[bytes, str]:
    """Select proper serializer to serialize.

    Params:
    ------------------------------
    type_: Str of the data type indicaing how to serialize the object.
      Additional messages after the data type with `:` as the seperator will
      be passed to `serializer` function.

    Return:
    ------------------------------
    bytes: Serialization result of the object.
    type_: The data type without additional messages.
    """
    if type_ is not None:
        splited = type_.split(":")
        if len(splited) == 1:
            type_, addon = splited[0], None
        elif len(splited) == 2:
            type_, addon = splited
        else:
            type_, *addon = splited
        if type_ not in _strategy_map:
            raise RuntimeError("No serializer found.")
        checker_fn, ser_fn, deser_fn = _strategy_map.get(type_)
    else:
        addon = None
        for _priority, type_ in _priority_order:
            checker_fn, ser_fn, deser_fn = _strategy_map[type_]
            if checker_fn and checker_fn(obj):
                break
        else:
            raise RuntimeError("No serializer found.")
    # Try to serialize.
    try:
        bytes_ = ser_fn(obj, addon)
    except ValueError as e:
        logger.warning(f"Failed to serialize object by {type_}: {e}.")
    return bytes_, type_


def deserialize(
    bytes_: bytes,
    type_: str,
) -> Any:
    """Use deserializer specified by `type_` to deserialize.

    Params:
    ------------------------------
    type_: Str of the data type indicaing how to serialize the object.
      Additional messages after the data type with `:` as the seperator will
      be passed to `serializer` function.

    Return:
    ------------------------------
    Object restored from bytes.
    """
    splited = type_.split(":")
    if len(splited) == 1:
        type_, addon = splited[0], None
    elif len(splited) == 2:
        type_, addon = splited
    else:
        type_, *addon = splited
    _, _, deserializer_fn = _strategy_map[type_]
    return deserializer_fn(bytes_, addon)


# %%
@checker("json", priority = 99)
def is_json(obj: Any) -> bool:
    if (isinstance(obj, (dict, list, str, int, float, bool, type(None)))
        and obj.__sizeof__() <= JSON_MAX):
        try:
            json.dumps(obj)
            return True
        except (TypeError, ValueError):
            return False
    return False


@serializer("json")
def json_serialize(
    obj: Any | None,
    addon: str | list[str] | None = None,
) -> bytes:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf8")


@deserializer("json")
def json_deserialize(
    bytes_: bytes,
    addon: str | list[str] | None = None,
) -> Any:
    if len(bytes_) == 0:
        return None

    return json.loads(bytes_.decode("utf8"))


# %%
@checker("pickle", priority = 10)
def is_anything(obj: Any) -> bool:
    return True


@serializer("pickle")
def pickle_serialize(
    obj: Any | None,
    addon: str | list[str] | None = None,
) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


@deserializer("pickle")
def pickle_deserialize(
    bytes_: bytes,
    addon: str | list[str] | None = None,
) -> Any:
    if len(bytes_) == 0:
        return None

    return pickle.loads(bytes_)  # noqa: S301


# %%
@checker("numpy", priority = 95)
def is_numpy(obj: Any) -> bool:
    try:
        # Checker will be called first, so check if Numpy is installed here.
        # And it's clear that the object is not np.ndarray if import fails.
        import numpy as np
        return isinstance(obj, np.ndarray)
    except ImportError:
        return False


@serializer("numpy")
def numpy_serialize(
    obj: np.ndarray | None,
    addon: str | list[str] | None = None,
) -> bytes:
    if obj is None:
        return b""

    import numpy as np

    buffer = io.BytesIO()
    np.save(buffer, obj, allow_pickle=False)
    bytes_ = buffer.getvalue()
    if len(bytes_) > NUMPY_COMPRESSS_MIN:
        bytes_ = zlib.compress(bytes_)
    return bytes_


@deserializer("numpy")
def numpy_deserialize(
    bytes_: bytes,
    addon: str | list[str] | None = None,
) -> np.ndarray:
    if len(bytes_) == 0:
        return None

    import numpy as np

    if len(bytes_) > NUMPY_COMPRESSS_MIN:
        bytes_ = zlib.decompress(bytes_)
    return np.load(io.BytesIO(bytes_), allow_pickle=False)


# %%
@checker("pddf_csv", priority = 1)
def is_pddf_csv(obj: Any) -> bool:
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        return False
    return (isinstance(obj, pd.DataFrame)
        and obj.memory_usage().sum() <= DF_CSV_MAX
        and not np.any(obj.dtypes == np.dtype("O"))
    )


@serializer("pddf_csv")
def pddf_csv_serialize(
    obj: pd.DataFrame | None,
    addon: str | list[str] | None = None,
) -> bytes:
    if obj is None:
        return b""

    buffer = io.StringIO()
    obj.to_csv(buffer, index=True, encoding="utf-8")
    return buffer.getvalue().encode("utf-8")


@deserializer("pddf_csv")
def pddf_csv_deserialize(
    bytes_: bytes,
    addon: str | list[str] | None = None,
) -> pd.DataFrame:
    if len(bytes_) == 0:
        return None

    import pandas as pd

    return pd.read_csv(
        io.StringIO(bytes_.decode("utf-8")),
        index_col=0
    )


# %%
@checker("pddf_feather", priority = 85)
def is_pddf_feather(obj: Any) -> bool:
    try:
        import numpy as np
        import pandas as pd
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return (isinstance(obj, pd.DataFrame)
        and obj.memory_usage().sum() <= DF_FEATHER_MAX
        and not np.any(obj.dtypes == np.dtype("O"))
    )


@serializer("pddf_feather")
def pddf_feather_serialize(
    obj: pd.DataFrame | None,
    addon: str | list[str] | None = None,
) -> bytes:
    if obj is None:
        return b""

    buffer = io.BytesIO()
    obj.to_feather(buffer, compression="zstd")
    return buffer.getvalue()


@deserializer("pddf_feather")
def pddf_feather_deserialize(
    bytes_: bytes,
    addon: str | list[str] | None = None,
) -> pd.DataFrame:
    if len(bytes_) == 0:
        return None

    import pandas as pd

    return pd.read_feather(io.BytesIO(bytes_))


# %%
@checker("pddf_parquet", priority = 80)
def is_pddf_parquet(obj: Any) -> bool:
    try:
        import numpy as np
        import pandas as pd
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return (isinstance(obj, pd.DataFrame)
        and not np.any(obj.dtypes == np.dtype("O"))
    )


@serializer("pddf_parquet")
def pddf_parquet_serialize(
    obj: pd.DataFrame | None,
    addon: str | list[str] | None = None,
) -> bytes:
    if obj is None:
        return b""

    buffer = io.BytesIO()
    obj.to_parquet(buffer, engine="pyarrow", compression="zstd", index=True)
    return buffer.getvalue()


@deserializer("pddf_parquet")
def pddf_parquet_deserialize(
    bytes_: bytes,
    addon: str | list[str] | None = None,
) -> pd.DataFrame:
    if len(bytes_) == 0:
        return None

    import pandas as pd

    return pd.read_parquet(io.BytesIO(bytes_))


# %%
@checker("exception", priority = 98)
def is_exception(obj: Any):
    if isinstance(obj, BaseException):
        return True
    return False


@serializer("exception")
def exception_serialize(
    obj: BaseException | None,
    addon: str | list[str] | None = None,
) -> bytes:
    if obj is None:
        return b""

    return str_exception(obj).encode("utf8")


@deserializer("exception")
def exception_deserialize(
    bytes_: bytes,
    addon: str | list[str] | None = None,
) -> BaseException:
    if len(bytes_) == 0:
        return None

    return destr_exception(bytes_.decode("utf8"))
