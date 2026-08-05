# AGENTS.md

## Monorepo Context

dirtbear lives in the `ubears` monorepo. Two sibling packages are **editable local deps**:
- `../flagbear` — string parsing, type coercion, data extraction utilities
- `../statbear` — statistical helpers

If you need to modify flagbear/statbear code, you must edit those repos directly. They are not vendored here.

## Commands

```bash
pixi run test              # pytest (all tests)
pixi run pytest <path>     # single test file, e.g. tests/dirtbear/dflater/test_ex2df.py
pixi run pypi-build        # build sdist + wheel
pixi run pypi-check        # twine check dist/*
pixi run pypi-uptest       # upload to testpypi
pixi run pypi-upload       # upload to pypi
```

pytest is configured in `pytest.ini` with `-v -s --cache-clear`. Logs go to `pytest.log`.

## Build & Packaging

- Build backend: **hatchling** (`pyproject.toml`)
- Version is `__version__` in `src/dirtbear/__init__.py`
- Source layout: `src/dirtbear/` (PEP 621 src-layout)
- Package data includes `.csv` and `.json` under `src/`

## Source Structure

```
src/dirtbear/
├── dflater/   — expression parsing & DataFrame construction (ex2df, ex4df, exoptim, callables)
├── quant/     — backtesting, data loading (backtest, dataloader)
├── stats/     — AHP, sklearn wrappers (ahp, skltree)
├── visual/    — charting (gridchart, kline)
├── spanner/   — DSL parsing, DataFrame logging (pdsl, dflog)
├── locale/    — Chinese calendar, pronouns, geo-encoding (calender, pronoun, geoenc)
├── data/      — bundled CSVs/JSONs (govern_region, userdict)
└── docer/     — document processing
```

Tests mirror this under `tests/dirtbear/<subpackage>/`.

## Code Conventions

- Google-style docstrings with `Params:`, `Returns:`, `Attrs:` sections separated by `----------`
- `# %%` cell markers between logical sections (IPython-style)
- `from __future__ import annotations` at top of every module
- Shebang + file header block on every `.py` file
- `logging.basicConfig(level=logging.INFO)` block at module level when `__name__ == "__main__"`
- Uses `numpy`/`pandas` heavily; `flagbear` for string/data parsing primitives

## Linting

Ruff is the linter and formatter (`pyproject.toml [tool.ruff]`):
- Rules: E, F, B (errors, pyflakes, bugbear)
- Line length: 88
- Format: double quotes, space indent
- `__init__.py` ignores E402 (module-level imports not at top)
