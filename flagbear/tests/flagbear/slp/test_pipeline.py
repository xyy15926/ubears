#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_pipeline.py
#   Author: xyy15926
#   Created: 2026-04-12 14:06:07
#   Updated: 2026-04-21 10:03:04
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations

import pytest

if __name__ == "__main__":
    from importlib import reload

    from flagbear.slp import databundle, finer, pipeline

    reload(finer)
    reload(databundle)
    reload(pipeline)

import pickle
import shutil

import numpy as np

from flagbear.slp.databundle import DataBundle, DataBundleFactory
from flagbear.slp.finer import get_tmp_path
from flagbear.slp.pipeline import (
    Pipe,
    PipeFactory,
    Pipeline,
)

TMP_DIR = get_tmp_path() / "pytest_tmpdir"


@pytest.fixture(scope="module", autouse=False)
def tmpfile_fixture(request):
    yield

    # Remove the tmp file during the pytest.
    # Clear only once with `scope=module`.
    pytest_tmp = get_tmp_path() / TMP_DIR
    shutil.rmtree(pytest_tmp, ignore_errors=True)
    if not any(get_tmp_path().iterdir()):
        get_tmp_path().rmdir()


@DataBundleFactory.register()
class NDABundle(DataBundle):
    def dumps_data(self):
        return self.data.dumps()

    @staticmethod
    def loads_data(bytes_, metadata=None):
        return pickle.loads(bytes_)


# %%
def test_PipeFactory_reigster():
    nda = np.random.randint(1, 100, (3, 4), dtype=np.int32)
    nda_ori = nda.copy()
    bundle = NDABundle(nda)

    pipe_name = "Add1Pipe"

    @PipeFactory.register()
    class Add1Pipe(Pipe):
        def process(self, bundle: NDABundle):
            bundle.data += 1
            return bundle

    pipe = Add1Pipe()

    stage_name = "Add1Stage"
    pipe(bundle, stage_name)
    assert np.all(bundle.data == nda_ori + 1)
    assert pipe.config == {}
    assert bundle.lineage[stage_name]["pipe_name"] == pipe_name


# %%
def test_PipelineStage_from_func():
    nda = np.random.randint(1, 100, (3, 4), dtype=np.int32)
    nda_ori = nda.copy()
    bundle = NDABundle(nda)

    pipe_name = "Add1Pipe"

    def add1(bundle: DataBundle):
        bundle.data += 1
        return bundle

    Add1Pipe = PipeFactory.from_func(pipe_name, add1)
    pipe = Add1Pipe()

    pipe(pipe(bundle, "S1"), "S2")
    assert np.all(bundle.data == nda_ori + 2)
    assert pipe.config == {}
    assert bundle.lineage["S1"]["pipe_name"] == pipe_name
    assert bundle.lineage["S2"]["pipe_name"] == pipe_name
    assert bundle.lineage["S2"]["exec_count"] == 2


# %%
def test_PipeFactory_create_instance():
    nda = np.random.randint(1, 100, (3, 4), dtype=np.int32)
    nda_ori = nda.copy()
    bundle = NDABundle(nda)

    @PipeFactory.register()
    class Add1Pipe(Pipe):
        def process(self, bundle: NDABundle):
            bundle.data += 1
            return bundle

    pipe_name = "Add1Pipe"
    pipe = Add1Pipe()

    pipe = PipeFactory.create_instance(pipe_name)
    pipe(pipe(bundle, "S1"), "S2")
    assert np.all(bundle.data == nda_ori + 2)
    assert pipe.config == {}
    assert bundle.lineage["S1"]["pipe_name"] == pipe_name
    assert bundle.lineage["S2"]["pipe_name"] == pipe_name
    assert bundle.lineage["S2"]["exec_count"] == 2


# %%
def test_Pipeline(tmpfile_fixture):
    nda = np.ones((3, 4), dtype=np.int32)
    nda_ori = nda.copy()
    bundle = NDABundle(nda)

    @PipeFactory.register()
    class Add1Pipe(Pipe):
        def process(self, bundle: NDABundle):
            bundle.data += 1
            return bundle

    @PipeFactory.register()
    class Add2Pipe(Pipe):
        def process(self, bundle: NDABundle):
            bundle.data += 2
            return bundle

    # Pipeline process.
    pipeline = Pipeline("add12", TMP_DIR)
    pipeline.add_pipe(Add1Pipe()).add_pipe(Add2Pipe())
    pipeline.process(bundle)
    assert np.all(bundle.data == nda_ori + 3)
    bundle.lineage["add12_Add1Pipe_0"]["exec_count"] = 1
    bundle.lineage["add12_Add2Pipe_0"]["exec_count"] = 1

    # Pipeline save and load.
    pipeline = Pipeline("add12", TMP_DIR)
    pipeline.add_pipe(Add1Pipe(), "Add1Pipe2").add_pipe(
        Add2Pipe(), "Add2Pipe2"
    )
    pipeline.process(bundle, save_checkpoints=True)
    # The bundle has been proceeded by 2 pipelines.
    assert np.all(bundle.data == nda_ori + 6)
    bundle.lineage["add12_Add1Pipe_0"]["exec_count"] = 1
    bundle.lineage["add12_Add2Pipe_0"]["exec_count"] = 1
    bundle.lineage["add12_Add1Pipe2"]["exec_count"] = 1
    bundle.lineage["add12_Add2Pipe2"]["exec_count"] = 1
    assert len(bundle.lineage) == 4

    # Load bundle from checkpoint manually.
    pipeline = Pipeline("add12", TMP_DIR)
    loaded = pipeline.load_checkpoint("Add1Pipe2")
    assert np.all(loaded.data == nda_ori + 4)
    assert len(loaded.lineage) == 3
    bundle.lineage["add12_Add1Pipe_0"]["exec_count"] = 1
    bundle.lineage["add12_Add2Pipe_0"]["exec_count"] = 1
    bundle.lineage["add12_Add1Pipe2"]["exec_count"] = 1

    # Continue process.
    pipeline = Pipeline("add12", TMP_DIR)
    pipeline.add_pipe(Add1Pipe(), "Add1Pipe2").add_pipe(
        Add2Pipe(), "Add2Pipe2"
    )
    # Load bundle automatically.
    loaded = pipeline.process(None, start_from="Add1Pipe2")
    assert np.all(loaded.data == nda_ori + 6)
    loaded.lineage["add12_Add1Pipe2"]["exec_count"] = 1
    loaded.lineage["add12_Add2Pipe2"]["exec_count"] = 1
    loaded.lineage["add12_Add1Pipe_0"]["exec_count"] = 1
    loaded.lineage["add12_Add2Pipe_0"]["exec_count"] = 1
    assert len(loaded.lineage) == 4

    # Skip some pipes.
    pipeline = Pipeline("otherpipeline", TMP_DIR)
    pipeline.add_pipe(Add1Pipe(), "Add1Pipe2").add_pipe(
        Add2Pipe(), "Add2Pipe2"
    )
    # Skip the first pipe.
    loaded = pipeline.process(loaded, start_from="Add1Pipe2")
    assert np.all(loaded.data == nda_ori + 8)
