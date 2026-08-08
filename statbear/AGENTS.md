# AGENTS.md

## Monorepo Context

statbear lives in the `ubears` monorepo. No sibling package dependencies (unlike nutsbear/dirtbear).

## Commands

```bash
pixi run test              # pytest (all tests)
pixi run pytest <path>     # single test file, e.g. tests/statbear/discret/test_exdiscret.py
pixi run pytest <path>::<name> # single test function
pixi run pypi-build        # build sdist + wheel
pixi run pypi-check        # twine check dist/*
pixi run pypi-uptest       # upload to testpypi
pixi run pypi-upload       # upload to pypi
```

pytest is configured in `pytest.ini` with `-v -s --cache-clear`. Logs go to `pytest.log`.

## Build & Packaging

- Build backend: **hatchling** (`pyproject.toml`)
- Version is `__version__` in `src/statbear/__init__.py`
- Source layout: `src/statbear/` (PEP 621 src-layout)

## Source Structure

```
src/statbear/
├── discret/   — discretization methods
├── panel/     — panel data metrics
└── talib/     — TA-Lib wrappers
```

Tests mirror this under `tests/statbear/`.

## Key Dependencies

- `numpy`, `scipy`, `scikit-learn` — core scientific stack
- `TA-lib` — technical analysis (requires C library)
- `pandas` — only for production tests (marker: `pandas`)

## Code Conventions

- Shebang + file header block on every `.py` file
- `from __future__ import annotations` at top of every module
- `# %%` cell markers between logical sections (IPython-style)
- Google-style docstrings with `Params:`, `Returns:`, `Attrs:` sections

## Linting

Ruff extends `../ruff.toml` (root config):
- Rules: E, W, F, B, C4, UP, SIM, PERF, RUF, S, I, N, D, TRY, LOG, C90, TCH, T20
- Line length: 79
- Test files: relaxed rules (no docstring/annotation requirements, assert allowed)

## Gotchas

- TA-Lib requires C library installation; may fail on bare systems
- Test markers: `pkgs`, `pandas`, `numpy`, `bears`
- Tests use session/function timing fixtures from `conftest.py`
