---
name: ruff-cleanup
description: Python 项目 ruff lint 错误的系统化清理工作流
---

## 功能

使用分阶段方法解决 Python 项目的 ruff lint 错误：自动修复 → 手动修复 → 内容验证。确保零 ruff 错误，同时保留代码行为和文档完整性。

## 使用场景

需要清理整个 Python 代码库的 ruff lint 错误时使用。适用于已在 `pyproject.toml` 中配置 ruff 且有测试套件的项目。

## 工作流

### 阶段 1：安全自动修复

```bash
ruff check --fix src/
```

修复导入排序、未使用的导入/变量等低风险问题。运行测试，然后提交。

### 阶段 2：不安全自动修复

```bash
ruff check --fix --unsafe-fixes src/
```

修复行为变更问题，如 `zip(..., strict=False)`。仔细运行测试，然后提交。

### 阶段 3：低难度手动修复

- E501：重新格式化过长注释
- D205/D414/D419：修复 docstring 格式
- RUF002：替换歧义 unicode 字符
- S301：对故意使用的 pickle 添加 noqa
- TRY004：使用具体异常类型
- PERF401：重写为列表推导式

### 阶段 4：中难度手动修复

- D103/D102：添加缺失的 docstring
- N806：对故意使用的大写变量添加 noqa
- SIM101：合并 isinstance 调用
- SIM115：使用上下文管理器

### 阶段 5：高难度手动修复

- C901/N802/N803/RUF012：对复杂/故意模式添加 noqa 抑制

### 内容验证

- 检查 git diff 中的 docstring 编辑，确保没有丢失内容
- E501 修复可能会截断文档——恢复丢失的内容
- 当 `# noqa` 不起作用时（如 docstring 内部），使用 `pyproject.toml` 中的 `per-file-ignores`

## 关键经验

- `# noqa` 注释在 docstring 内部不起作用——改用 `per-file-ignores`
- 领域特定的大写变量（SLIPPAGE、LOT）应抑制而非重命名
- 每个阶段提交一次，便于回滚
- 每个阶段后运行测试

## 常用命令

```bash
ruff check src/                        # 检查错误
ruff check --fix src/                  # 安全自动修复
ruff check --fix --unsafe-fixes src/   # 不安全自动修复
pixi run test                          # 运行测试
git diff --stat                        # 提交前确认范围
```
