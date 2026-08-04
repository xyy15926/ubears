# AGENTS.md

## Monorepo Context

nutsbear lives in the `ubears` monorepo. One sibling package is an **editable local dep**:
- `../flagbear` — string parsing, type coercion, data extraction utilities

If you need to modify flagbear code, edit `../flagbear` directly. It is not vendored here.

## Commands

```bash
pixi run test              # pytest (all tests)
pixi run pytest <path>     # single test file, e.g. tests/nutsbear/test_trainer.py
pixi run pypi-build        # build sdist + wheel
pixi run pypi-check        # twine check dist/*
pixi run pypi-upload       # upload to pypi
```

pytest is configured in `pytest.ini` with `-v -s --cache-clear`. Logs go to `pytest.log`.

## Build & Packaging

- Build backend: **hatchling** (`pyproject.toml`)
- Version is `__version__` in `src/nutsbear/__init__.py`
- Source layout: `src/nutsbear/` (PEP 621 src-layout)

## Source Structure

```
src/nutsbear/
├── mods/      — neural network modules (attention, diffusion, posemb, fixture)
├── demos/     — demo models (autoencoder, deepfm, transformer, unet)
├── trainer.py — training loop with TensorBoard logging
```

Tests mirror this under `tests/nutsbear/` and `tests/torch/`.

## Key Dependencies

- PyTorch ecosystem: `torch`, `torchvision`, `tensorboard`
- Optional: `torch-directml` (DirectML backend, via `[directml]` extra)
- `flagbear` for data parsing primitives

## Code Conventions

- Shebang + file header block on every `.py` file
- `from __future__ import annotations` at top of every module
- `# %%` cell markers between logical sections (IPython-style)
- Google-style docstrings with `Params:`, `Returns:`, `Attrs:` sections
- Debug reload block under `if __name__ == "__main__":` (see `trainer.py`)

## Linting

Ruff is the linter and formatter (`pyproject.toml [tool.ruff]`):
- Rules: E, W, F, B, C4, UP, SIM, PERF, RUF, S, I, N, D, TRY, LOG, C90, TCH, T20
- Line length: 79
- Format: double quotes, space indent
- Naming exceptions: N802/N803/N806/N812/N818 ignored (Hungarian notation, math symbols)
- Test files: relaxed rules (no docstring/annotation requirements, assert allowed)
