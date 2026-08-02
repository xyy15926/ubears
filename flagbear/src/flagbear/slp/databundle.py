#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: databundle.py
#   Author: xyy15926
#   Created: 2026-04-10 11:52:19
#   Updated: 2026-04-21 22:40:04
#   Description:
# ---------------------------------------------------------

# %%
import inspect
import json
import logging
import pickle
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any

if __name__ == "__main__":
    from importlib import reload

    from flagbear.slp import finer
    reload(finer)
from flagbear.slp.finer import use_file

logger = logging.getLogger(__name__)

DATABUNDLE_DIR = "databundle"


# %%
@dataclass
class DataBundle(ABC):
    """Dataclass with data, metadata and lineage.

    Class Attrs:
    -------------------------------
    reg_name: The name use to identify the exact derived DataBundle in
      DataBundleFactory.

    Attrs:
    -------------------------------
    data: Data.
    metadata: Dict to describe the data.
    lineage: Dict with key as the stage name to record the infomation
      of the data-procedure stages.
    """
    data: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    lineage: dict[str, dict] = field(default_factory=dict)

    def add_metadata(self, key: str, value: Any):
        """Add metatdata."""
        self.metadata[key] = value
        return self

    def trace(self, key: str, info: dict):
        """Update lineage."""
        self.lineage.setdefault(key, {}).update(info)

    @abstractmethod
    def dumps_data(self) -> str | bytes:
        """Dump data into bytes."""
        pass

    @staticmethod
    @abstractmethod
    def loads_data(bytes_: str | bytes, metadata: dict = None) -> Any:
        """Load data from bytes."""
        pass

    def save_bundle(
        self,
        fname: str | Path,
        forced: bool = False,
    ) -> Path:
        """Save bundle in zip-file.

        All the data, metadata and lineage will be write into a zip-file,
        along wiht `reg_name.txt` to record the `reg_name` so to determine
        the exact bundle type.
        """
        fname = use_file(fname, "today", 1, DATABUNDLE_DIR)
        # Make dir if not exists.
        if fname.exists():
            if forced:
                logger.warning(f"{fname} already exists.")
            if not forced:
                logger.error(f"{fname} already exists.")
                raise ValueError(f"{fname} already exists.")
        # Try to save bundle.
        try:
            with zipfile.ZipFile(fname, "w") as zipf:
                # Save data with data-specified method.
                zipf.writestr("data.bin", self.dumps_data())
                # Save metadata, lineage with json.
                zipf.writestr("metadata.json", json.dumps(self.metadata))
                zipf.writestr("lineage.json", json.dumps(self.lineage))
                zipf.writestr("reg_name.txt", self.reg_name.encode("utf8"))
            logger.info(f"Save data bundle at {fname}.")
            return fname
        except Exception as e:
            logger.error(f"Fail to save data bundle at {fname}: {e}.")
            raise

    @classmethod
    def load_bundle(cls, fname: str | Path):
        """Load bundle from zip-file."""
        # The latest version will be used.
        fname = use_file(fname, None, 0, DATABUNDLE_DIR)
        # Check if file exists.
        if not fname.exists():
            logger.error(f"{fname} does not exist.")
            raise ValueError(f"{fname} does not exist.")
        # Try to load bundle.
        try:
            with zipfile.ZipFile(fname, "r") as zipf:
                metadata = json.loads(zipf.open("metadata.json", "r").read())
                lineage = json.loads(zipf.open("lineage.json", "r").read())
                data = cls.loads_data(zipf.open("data.bin", "r").read(), metadata)
            logger.info(f"Load data bundle at {fname}.")
            bundle = cls(data, metadata, lineage)
            return bundle
        except Exception as e:
            logger.error(f"Fail to load data bundle at {fname}: {e}.")
            raise


# %%
class DataBundleFactory:
    """Factory class to create and manage derived Databundle.

    Class Attrs:
    ------------------------
    _registry: Registry of derived DataBundle.
    """
    _registry: dict[str, type[DataBundle]] = {}

    @classmethod
    def register(cls, reg_name: str = None):
        """Register class derived from DataBundle."""
        def decorator(bundle_class: type[DataBundle]) -> type[DataBundle]:
            if not issubclass(bundle_class, DataBundle):
                raise TypeError(f"{bundle_class.__name__} is not derived from "
                                f"DataBundle.")
            reg_name_ = reg_name or bundle_class.__name__
            cls._registry[reg_name_] = bundle_class
            bundle_class.reg_name = reg_name_
            return bundle_class
        return decorator

    @classmethod
    def from_slfunc(
        cls,
        name: str,
        dumps_data: Callable,
        loads_data: Callable,
    ) -> type[DataBundle]:
        """Create dynanmic DataBundle class with dump and load function.

        Params:
        ----------------------------
        name: Dynamic DataBundle class name.
        dumps_data: Function, without `self` as the first parameter, to dump
          data into bytes. Namely, the function should not rely on `self` but
          only the exact arguments.
        loads_data: Function to load data from bytes.
        """
        def dumps_data_method(self):
            return dumps_data(self.data)

        def loads_data_method(bytes_, metadata = None):
            return loads_data(bytes_, metadata)

        new_class = cls.from_callables(name, dumps_data_method, loads_data_method)
        # Update the attribute `__module__`.
        new_class.__module__ = dumps_data.__module__
        return new_class

    @classmethod
    def from_callables(
        cls,
        name: str,
        dumps_data_method: Callable,
        loads_data_method: Callable,
        **methods: Callable,
    ) -> type[DataBundle]:
        """Create dynamic DataBundle class with a list of callables.

        Params:
        ----------------------------
        name: Dynamic DataBundle class name.
        dumps_data_method: Method, with `self` as the first parameter, to dumps
          data into bytes.
        loads_data: Staticmethod to load data from bytes.
        methods: Methods to be added to the dynamic class.
          1. `cls` or `self` should be the first parameter for normal methods
            or classmethods, just like they are defined in a normal class
            namespace.
          2. As the parameters will be used to determine the method type.
        """
        namespace = {
            "__doc__": f"Dynamic DataBundle class registed with {name}.",
            "__module__": dumps_data_method.__module__,
            "dumps_data": dumps_data_method,
            "loads_data": staticmethod(loads_data_method),
            "reg_name": name,
        }
        for method_name, method in methods.items():
            namespace[method_name] = cls._wrap_method(method)
        new_class = type(name, (DataBundle, ), namespace)
        cls._registry[name] = new_class
        return new_class

    @classmethod
    def _wrap_method(cls, func: Callable) -> Callable:
        """Wrap method by checking their params."""
        sig = inspect.signature(func)
        # Determine the functions' type with their parameters.
        params = list(sig.parameters.keys())
        if len(params) == 0:
            return staticmethod(func)
        if params[0] == "cls":
            return classmethod(func)
        if params[0] == "self":
            return func
        return staticmethod(func)

    @classmethod
    def create_instance(cls, reg_name: str, *args, **kwargs) -> DataBundle:
        """Create an instance of registed class with registed name."""
        if reg_name not in cls._registry:
            raise KeyError(f"{reg_name} not Found.")
        return cls._registry[reg_name](*args, **kwargs)

    @classmethod
    def load_instance(cls, reg_name: str, fname: str | Path) -> DataBundle:
        """Load instacne from given filename."""
        if reg_name not in cls._registry:
            raise KeyError(f"{reg_name} not Found.")
        return cls._registry[reg_name].load_bundle(fname)

    @classmethod
    def try_load_instance(cls, fname: str | Path) -> DataBundle | None:
        """Try to load instance from given filename.

        The exact data-bundle type will be read from the zip-file first.
        """
        # The latest version will be used.
        fname = use_file(fname, None, 0, DATABUNDLE_DIR)
        # Check if file exists.
        if not fname.exists():
            logger.error(f"{fname} does not exist.")
            raise ValueError(f"{fname} does not exist.")
        # Try to read the data-bunle type.
        try:
            with zipfile.ZipFile(fname, "r") as zipf:
                reg_name = zipf.open("reg_name.txt", "r").read().decode("utf8")
        except Exception as e:
            logger.error(f"Fail to read the bundle type from {fname}: {e}.")
            raise
        return cls.load_instance(reg_name, fname)

    @classmethod
    def registed_class(cls) -> list:
        """Return list of registered class names."""
        return list[cls._registry.keys()]


# %%
@DataBundleFactory.register()
class PickableBundle(DataBundle):
    """DataBundle using pickle for serialization."""
    def dumps_data(self) -> bytes:
        """Serialize data with pickle."""
        return pickle.dumps(self.data)

    @staticmethod
    def loads_data(bytes_, metadata: dict = None) -> Any:
        """Deserialize pickle bytes."""
        return pickle.loads(bytes_)  # noqa: S301


# %%
def pickle_dumps(data) -> bytes:
    """Serialize data with pickle."""
    return pickle.dumps(data)


def pickle_loads(bytes_, metadata: dict = None):
    """Deserialize pickle bytes."""
    return pickle.loads(bytes_)


def concat_params(
    reg_name: str,
    args: Any,
    kwargs: Any,
) -> str:
    """Concate registry name with arguments.

    Only the int, float and str argument will be used to construct the name
    that will be used to registed in DataBundleFactory.

    Params:
    -----------------------------
    reg_name: Part of the registry name.
    args: Tuple of positional arguments.
    kwargs: Dict of keywords arguments.

    Return:
    -----------------------------
    Concated name registed in DataBundleFactory
    """
    args_str = "_".join([
        str(ele) for ele in args
        if isinstance(ele, (int, float, str))
    ])
    kwargs_str = "_".join([
        f"{k}_{v}" for k, v in kwargs.items()
        if isinstance(v, (int, float, str))
    ])
    concated = "_".join([reg_name, args_str, kwargs_str])

    return concated
    # try:
    #     arg_str = json.dumps((args, kwargs), sort_keys=True, default=str)
    # except (TypeError, ValueError):
    #     arg_str = str((args, kwargs))
    # return hashlib.md5(arg_str.encode("utf8")).hexdigest()[:16]


def check_params(
    args: list[Any],
    kwargs: dict[str, Any],
    metadata: dict[str, Any],
) -> bool:
    """Check if the arguments passed and loaded are the same.

    Params:
    -----------------------------
    args: Positional arguments.
    kwargs: Keywords arguments.
    metadata: Metadata loaded from the bundle cache.
    """
    same_flag = True
    if len(metadata["_func_params_args"]) != len(args):
        same_flag = False
        logger.warning("Positional arguments are different.")
    if len(metadata["_func_params_kwargs"]) != len(kwargs):
        same_flag = False
        logger.warning("Keyword arguments are different.")
    for v1, v2 in zip(metadata["_func_params_args"], args, strict=False):
        if v1 != v2:
            logger.error(f"Argument {v1} is different from {v2}.")
            same_flag = False
    for k, v in metadata["_func_params_kwargs"].items():
        if v != kwargs.get(k):
            logger.error(f"Argument {k}:{v} is different "
                         f"from {k}:{kwargs.get(k)}.")
            same_flag = False

    return same_flag


def bundle_cache(
    reg_name: str = None,
    fname: str | Path = None,
    dumps_data: Callable = pickle_dumps,
    loads_data: Callable = pickle_loads,
    *,
    dest: str = DATABUNDLE_DIR,
    strict: bool = False,
    rollback: bool = True,
):
    """Data bundle cache decorator.

    Attention:
    -----------------------------
    As the arguments will effect the result of function, so the arguments
    will be used to determine the cache file.
    1. But only the int, float and str argument will be used to construct
      the cache file name, so make sure not to change the behavior to use
      the wrapped `func`.
    2. Also the arguments will be saved at the metadata of the data bundle,
      so make sure the arguments:
    2.1 Could be dumped by the `json.dumps`
    2.2 And compared with `=`.
    2.3 And won't be modified during the process of the `func`.

    Params:
    -----------------------------
    reg_name: Bundle type name in registry.
    fname: Exact filename to load cache from.
      This should be set to `None` in most case, unless the exact file should
      be used.
    dumps_data: Callable to dump data into bytes.
    loads_data: Callable to load data from bytes.
    dest: The root directory to save and load data bundle.
    strict: If the params passed to the `func` must be exact the same as the
      params loaded from the bundle.
    rollback: If to roll back to call the `func` to fetch the remote or
      expensive data when data cache exists but not valid.
    """
    def inner(func):
        @wraps(func)
        def decorator(*args, **kwargs):
            forced = kwargs.pop("forced", False)
            # The name of derived DataBundle will be like `XXXX.zip` which is
            # invalid when defined with `class` normally instead of `type`.
            reg_name_ = concat_params(
                reg_name or func.__name__,
                args,
                kwargs,
            ) + ".zip"

            # Regist new DataBundle type.
            if reg_name_ not in DataBundleFactory._registry:
                DataBundleFactory.from_slfunc(
                    reg_name_,
                    dumps_data,
                    loads_data
                )
                logger.info(f"Regist new DataBundle {reg_name_}.")

            # Try load data bundle from cache.
            fname_all = use_file(fname or reg_name_, None, 0, dest)
            if (not forced) and fname_all.is_file():
                try:
                    bundle = DataBundleFactory.load_instance(
                        reg_name_,
                        fname_all
                    )
                    is_same = check_params(args, kwargs, bundle.metadata)
                    if (not is_same) and strict:
                        raise ValueError("Arguments loaded are different "
                                         "from the ones passed.")
                    return bundle
                except Exception as e:
                    logger.warning(f"Recover bundle from cache {fname_all} "
                                   f"failed: {e}.")

            # Roll back to call the `func`.
            if forced or rollback or (not fname_all.is_file()):
                # Fetch remote or expensive data.
                data = func(*args, **kwargs)
                bundle = DataBundleFactory.create_instance(reg_name_, data)
                # bundle.trace("cache_stage", {"source": reg_name_})
                # As the `args` will be Tuple by default.
                bundle.add_metadata("_func_params_args", list(args))
                bundle.add_metadata("_func_params_kwargs", kwargs)

                # Save bundle.
                fname_all = use_file(fname or reg_name_, "today", 1, dest)
                bundle.save_bundle(fname_all)
                logger.info(f"Save cache at {fname_all}.")

                return bundle
            return None
        return decorator
    return inner
