#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: pipeline.py
#   Author: xyy15926
#   Created: 2026-04-09 19:16:01
#   Updated: 2026-04-27 22:30:27
#   Description:
# ---------------------------------------------------------

# %%
import datetime
import logging
import time
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Self

if __name__ == "__main__":
    from importlib import reload

    from flagbear.slp import databundle, finer

    reload(finer)
    reload(databundle)
from flagbear.slp.databundle import DataBundle, DataBundleFactory
from flagbear.slp.finer import use_dir

logger = logging.getLogger(__name__)


# %%
class Pipe(ABC):
    """One stage to process DataBundle.

    Class Attrs:
    ---------------------------
    reg_name: Registry name in PipeFactory.

    Attrs:
    ---------------------------
    config: Process config.
    exec_count: Execution counts.
    """

    def __init__(self, name: str | None = None, config: dict | None = None):
        """Init Pipe."""
        self.config = config or {}
        self.exec_count = 0

    @abstractmethod
    def process(self, bundle: DataBundle) -> DataBundle:
        """Process DataBundle."""
        pass

    def __call__(
        self, bundle: DataBundle, stage_key: str | None = None
    ) -> DataBundle:
        """Call `process` and do some additional records."""
        start = time.time()
        try:
            result = self.process(bundle)
            self.exec_count += 1
            bundle.trace(
                stage_key or self.__class__.__name__,
                {
                    "pipe_name": self.__class__.__name__,
                    "time_usage(ms)": time.time() - start,
                    "finished": datetime.datetime.now().isoformat(),
                    "exec_count": self.exec_count,
                },
            )
        except Exception:
            logger.exception(f"Stage [{self.name}] failed.")
            raise
        else:
            return result


# %%
class PipeFactory:
    """Factory to create and manage pipes.

    Class Attrs:
    --------------------------
    _registry: Registry of derived pipes.
    """

    _registry: ClassVar[dict[str, type[Pipe]]] = {}

    @classmethod
    def register(cls, reg_name: str | None = None):
        """Register derived pipes."""

        def decorator(pipe_class: type[Pipe]) -> type[Pipe]:
            if not issubclass(pipe_class, Pipe):
                raise TypeError(
                    f"{pipe_class.__name__} is not derived from Pipe."
                )
            reg_name_ = reg_name or pipe_class.__name__
            cls._registry[reg_name_] = pipe_class
            pipe_class.reg_name = reg_name_
            return pipe_class

        return decorator

    @classmethod
    def from_func(cls, name: str, func: Callable) -> type[Pipe]:
        """Create a pipe class with process functions."""

        def process_method(self, bundle: DataBundle) -> Any:
            return func(bundle)

        new_class = type(
            name,
            (Pipe,),
            {
                "process": process_method,
                "__doc__": "Dynamic Pipe derived from {func.__name__}",
                "__module__": func.__module__,
                "reg_name": name,
            },
        )
        cls._registry[name] = new_class
        return new_class

    @classmethod
    def create_instance(cls, reg_name: str, *args, **kwargs) -> type[Pipe]:
        """Create a pipe instantance with register name."""
        if reg_name not in cls._registry:
            raise KeyError(f"{reg_name} not Found.")
        return cls._registry[reg_name](*args, **kwargs)

    @classmethod
    def registed_pipe(cls) -> list:
        """Return list of registered pipe names."""
        return list[cls._registry.keys()]


# %%
class Pipeline:
    """Pipeline consist of a list of Pipe instances.

    Attrs:
    ---------------------------
    name: Name to identify the pipeline, which will be used to determine the
      checkpoint directory and mark the DataBundle's lineage processed.
    stages: Dict of the pipe instances.
    checkpoint_dir: Directory to save the checkpoints.
    """

    def __init__(
        self,
        name: str,
        checkpoint_dir: str | Path | None = None,
    ):
        """Init empty pipeline.

        Params:
        ------------------------------
        name: Name to identify the pipeline, which will be used to determine the
          checkpoint directory and mark the DataBundle's lineage processed.
        stages: Dict of the pipe instances.
        checkpoint_dir: Directory to save the checkpoints.
          1. `use_dir(name)` will be used by default.
          2. So an explicit `checkpoint_dir` must be passed, or a new
            `checkpoint_dir` with data_order_mark will be made and used.
        """
        self.name = name
        self.stages: dict[str, type[Pipe]] = {}
        self.stage_counter = Counter()
        self.checkpoint_dir = use_dir(
            checkpoint_dir or name, "today", 1, "tmp"
        )

    def add_pipe(self, pipe: Pipe, stage_key: str | None = None) -> Self:
        """Add a pipe.

        Params:
        ------------------------------
        stage_key: String to identify the pipe in the pipeline.
          1. Pipe class name will be used by default.
          2. Error will raised if duplicated `stage_key` is passed.
        """
        if stage_key is not None:
            if stage_key in self.stages:
                logger.error(f"Stage {stage_key} exists.")
                raise ValueError(f"Stage {stage_key} exists.")
        else:
            reg_name = pipe.reg_name
            stage_key = f"{reg_name}_{self.stage_counter[reg_name]}"
        self.stages[stage_key] = pipe
        self.stage_counter.update([pipe.reg_name])
        return self

    def load_checkpoint(self, stage_key: str) -> DataBundle:
        """Load from checkpoint."""
        fname = self.checkpoint_dir / (stage_key + ".zip")
        bundle = DataBundleFactory.try_load_instance(fname)
        if bundle is None:
            logger.error(f"Fail to load checkpoint from {fname}.")
            raise ValueError(f"Fail to load checkpoint from {fname}.")
        return bundle

    def save_checkpoint(self, stage_key: str, bundle: DataBundle):
        """Save into checkpoint."""
        fname = self.checkpoint_dir / (stage_key + ".zip")
        # Don't override.
        bundle.save_bundle(fname, forced=False)

    def process(
        self,
        bundle: DataBundle = None,
        start_from: str | None = None,
        save_checkpoints: bool = False,
    ) -> DataBundle:
        """Process DataBundle.

        Params:
        ------------------------------
        bundle: DataBundle.
        start_from: The name of the stage from which the process starts.
          Stages before `start_from`, including `start_from` will be skipped.
          1. If `bundle` is not provided, checkpoint will be loaded
            automatically.
          2. Else `bundle` provied will be used directly.
        save_checkpoints: If to save checkpoints after each stage.

        Return:
        ------------------------------
        DataBundle
        """
        for stage_key, stage in self.stages.items():
            if start_from is not None:
                if stage_key != start_from:
                    continue
                else:
                    # Only load checkpoint when no bundle provided.
                    if bundle is None:
                        bundle = self.load_checkpoint(start_from)
                        logger.info(f"Resume from checkpoint: {start_from}.")
                    start_from = None
            else:
                logger.info(f"Executing stage: {stage_key}.")
                bundle = stage(bundle, f"{self.name}_{stage_key}")
                if save_checkpoints:
                    self.save_checkpoint(stage_key, bundle)
        return bundle
