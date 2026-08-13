# AGENTS.md

## Monorepo Overview

Flat monorepo with 4 active Python packages. Each package is independently managed with its own pixi environment, pyproject.toml, and test suite. No root-level build or task orchestration. `suitbear/` is inactive (no source files, no config).

### Packages

| Package | Description | Python | Key Deps |
|---|---|---|---|
| `flagbear` | Utility toolkit (string parsing, data extraction) | >=3.10 | None |
| `statbear` | Statistical helpers (discretization, panel metrics, TA-Lib) | >=3.9 | numpy, scipy, scikit-learn, TA-lib |
| `nutsbear` | Neural network modules, demos, trainer | ==3.12 (pinned) | torch, torchvision, tensorboard, flagbear |
| `dirtbear` | Data processing (expression parsing, backtesting, visualization) | >=3.10 | numpy, pandas, flagbear, statbear |

### Dependency Graph

```
flagbear  (leaf)
   ↑
   ├── nutsbear
   └── dirtbear ← also depends on statbear

statbear  (leaf)
   ↑
   └── dirtbear
```

### Cross-Package Edits

Edit sibling packages directly (e.g., `../flagbear`). They are not vendored.

## Commands

All commands run inside each package's directory using `pixi run`:

```bash
pixi run test                  # all tests
pixi run pytest <path>         # single file, e.g. tests/<pkg>/test_foo.py
pixi run pytest <path>::<name> # single test function
pixi run pytest -m pandas      # by marker (pkgs, pandas, numpy, bears)
pixi run pypi-build            # build sdist + wheel
pixi run pypi-check            # twine check dist/*
pixi run pypi-uptest           # upload to testpypi
pixi run pypi-upload           # upload to pypi
```

## Key Conventions

- **Build**: hatchling (pyproject.toml), version from `src/<pkg>/__init__.py`
- **Linting**: Ruff (line-length 79, Google docstrings, `# %%` cell markers)
- **Testing**: pytest (`-v -s --cache-clear`), markers: `pkgs`, `pandas`, `numpy`, `bears`
- **Env**: Pixi >=0.55, uses Tsinghua mirrors (network issues possible)
- **Source layout**: `src/<pkg>/` (PEP 621 src-layout)

## Gotchas

- Each package has its own `.pixi/` environment; no shared env.
- Python version varies: flagbear >=3.10, nutsbear ==3.12, statbear >=3.9, dirtbear >=3.10 (pixi.toml says >=3.9).
- Cross-package editable installs configured in each pixi.toml via `[feature.this.pypi-dependencies]`.
- isort knows all four packages as first-party: `flagbear`, `statbear`, `nutsbear`, `dirtbear`.

## Subpackage Details

See each package's own `AGENTS.md` for specific guidance:
- `flagbear/AGENTS.md`
- `nutsbear/AGENTS.md`
- `dirtbear/AGENTS.md`
- `statbear/AGENTS.md`
