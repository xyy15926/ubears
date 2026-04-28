#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: serializer.py
#   Author: xyy15926
#   Created: 2026-04-22 15:21:57
#   Updated: 2026-04-27 22:39:01
#   Description:
# ---------------------------------------------------------

# %%
import logging
from typing import Any, Callable, Dict, Optional, Tuple, List
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
import json
import pickle
import zlib
import io
import threading
# from IPython.core.debugger import set_trace

if __name__ == "__main__":
    from importlib import reload
    from flagbear.slp import finer, storage
    reload(finer)
    reload(storage)

logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")

SerializerFn = Callable[[Any], bytes]
DeserializerFn = Callable[[bytes], Any]
CheckerFn = Callable[[Any], bool]

_strategy_map: Dict[str, Tuple[CheckerFn, SerializerFn, DeserializerFn]] = {}
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
def serialize(obj: Any, type_: str = None) -> Tuple[bytes, str]:
    """Select proper serializer to serialize."""
    if type_ is not None:
        if type_ not in _strategy_map:
            raise RuntimeError("No serializer found.")
        checker_fn, ser_fn, deser_fn = _strategy_map.get(type_)
    else:
        for _priority, type_ in _priority_order:
            checker_fn, ser_fn, deser_fn = _strategy_map[type_]
            if checker_fn and checker_fn(obj):
                break
        else:
            raise RuntimeError("No serializer found.")
    # Try to serialize.
    try:
        bytes_ = ser_fn(obj)
    except ValueError as e:
        logger.warning(f"Failed to serialize object by {type_}: {e}.")
    return bytes_, type_


def deserialize(bytes_: bytes, type_: str) -> Any:
    """Use deserializer specified by `type_` to deserialize."""
    _, _, deserializer_fn = _strategy_map[type_]
    return deserializer_fn(bytes_)


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
def json_serialize(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf8")


@deserializer("json")
def json_deserialize(bytes_: bytes) -> Any:
    return json.loads(bytes_.decode("utf8"))


# %%
@checker("pickle", priority = 10)
def is_anything(obj: Any) -> bool:
    return True


@serializer("pickle")
def pickle_serialize(obj: Any) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


@deserializer("pickle")
def pickle_deserialize(bytes_: bytes) -> Any:
    return pickle.loads(bytes_)


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
def numpy_serialize(obj: np.ndarray) -> bytes:
    import numpy as np

    buffer = io.BytesIO()
    np.save(buffer, obj, allow_pickle=False)
    bytes_ = buffer.getvalue()
    if len(bytes_) > NUMPY_COMPRESSS_MIN:
        bytes_ = zlib.compress(bytes_)
    return bytes_


@deserializer("numpy")
def numpy_deserialize(bytes_: bytes) -> np.ndarray:
    import numpy as np

    if len(bytes_) > NUMPY_COMPRESSS_MIN:
        bytes_ = zlib.decompress(bytes_)
    return np.load(io.BytesIO(bytes_), allow_pickle=False)


# %%
@checker("pddf:csv", priority = 1)
def is_pddf_csv(obj: Any) -> bool:
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        return False
    return (isinstance(obj, pd.DataFrame)
        and obj.memory_usage().sum() <= DF_CSV_MAX
        and not np.any(obj.dtypes == np.dtype("O"))
    )


@serializer("pddf:csv")
def pddf_csv_serialize(obj: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    obj.to_csv(buffer, index=True, encoding="utf-8")
    return buffer.getvalue().encode("utf-8")


@deserializer("pddf:csv")
def pddf_csv_deserialize(bytes_: bytes) -> pd.DataFrame:
    import pandas as pd

    return pd.read_csv(
        io.StringIO(bytes_.decode("utf-8")),
        index_col=0
    )


# %%
@checker("pddf:feather", priority = 85)
def is_pddf_feather(obj: Any) -> bool:
    try:
        import numpy as np
        import pandas as pd
        import pyarrow
    except ImportError:
        return False
    return (isinstance(obj, pd.DataFrame)
        and obj.memory_usage().sum() <= DF_FEATHER_MAX
        and not np.any(obj.dtypes == np.dtype("O"))
    )


@serializer("pddf:feather")
def pddf_feather_serialize(obj: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    obj.to_feather(buffer, compression="zstd")
    return buffer.getvalue()


@deserializer("pddf:feather")
def pddf_feather_deserialize(bytes_: bytes) -> pd.DataFrame:
    import pandas as pd

    return pd.read_feather(io.BytesIO(bytes_))


# %%
@checker("pddf:parquet", priority = 80)
def is_pddf_parquet(obj: Any) -> bool:
    try:
        import numpy as np
        import pandas as pd
        import pyarrow
    except ImportError:
        return False
    return (isinstance(obj, pd.DataFrame)
        and not np.any(obj.dtypes == np.dtype("O"))
    )


@serializer("pddf:parquet")
def pddf_parquet_serialize(obj: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    obj.to_parquet(buffer, engine="pyarrow", compression="zstd", index=True)
    return buffer.getvalue()


@deserializer("pddf:parquet")
def pddf_parquet_deserialize(bytes_: bytes) -> pd.DataFrame:
    import pandas as pd

    return pd.read_parquet(io.BytesIO(bytes_))
