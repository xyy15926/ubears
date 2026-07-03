# AGENTS.md

## Project Overview

flagbear: 工具集项目，包含多个子包。源码位于 `src/flagbear/`，测试位于 `tests/flagbear/`。

## Subpackages

| 子包 | 功能 |
|------|------|
| `answers` | 位运算、标量、随机数工具 |
| `const` | 模式匹配常量 |
| `llp` | 词法分析/语法分析器 |
| `sched` | 任务调度器 |
| `slp` | 序列化、缓存、存储工具 |
| `str2` | 字符串处理工具 |
| `tree` | 树/图结构 |

## Commands

```bash
# 运行所有测试
pixi run test

# 运行单个测试文件
pixi run pytest tests/flagbear/slp/test_finer.py

# 运行单个测试
pixi run pytest tests/flagbear/slp/test_finer.py::test_date_order_mark

# 运行指定标记的测试
pixi run pytest -m pandas
pixi run pytest -m numpy

# 构建
pixi run pypi-build

# 检查构建产物
pixi run pypi-check
```

## Key Conventions

- **构建**: hatchling (pyproject.toml)
- **Linting**: Ruff, rules: E, F, B (bugbear)
- **Testing**: pytest, 文件名 `test_*.py`, 类名 `Test*`, 函数名 `test_*`
- **Python**: >= 3.10 (target 3.13)
- **环境**: Pixi >= 0.55, 使用清华镜像源
- **测试默认参数**: `-v -s --cache-clear`
- **测试日志**: `pytest.log` (level: info), 控制台 (level: error)

## File Organization

```
src/flagbear/<subpkg>/
  __init__.py
  <module>.py

tests/flagbear/<subpkg>/
  test_<module>.py
```

## Gotchas

- 测试目录结构必须与 `src/flagbear/` 保持一致
- pytest.ini 中 markers: pkgs, pandas, numpy, bears
- 测试 fixtures 在 `tests/conftest.py`，scope: session/function
- 项目使用中文镜像源，网络问题可能影响依赖安装
