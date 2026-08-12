#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_cli.py
#   Author: xyy15926
#   Created: 2026-08-12 14:00:00
#   Updated: 2026-08-12 14:00:00
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations

if __name__ == "__main__":
    from importlib import reload

    from dirtbear.docer import mdnorm

    reload(mdnorm)

from dirtbear.docer.mdnorm.cli import (
    _parse_args,
    _process_file,
)

# %%
# --- _parse_args ---


def test_parse_args_normalize():
    argv = ["prog", "file.md", "--normalize"]
    pattern, align, normalize, comments = _parse_args(argv)
    assert pattern == "file.md"
    assert normalize is True
    assert comments is False
    assert align is None


def test_parse_args_comments():
    argv = ["prog", "*.md", "--comments"]
    pattern, _align, _normalize, comments = _parse_args(argv)
    assert pattern == "*.md"
    assert comments is True


def test_parse_args_align_left():
    argv = ["prog", "file.md", "--align", "left"]
    _pattern, align, _normalize, _comments = _parse_args(argv)
    assert align == "left"


def test_parse_args_align_center():
    argv = ["prog", "file.md", "--align", "center"]
    _pattern, align, _normalize, _comments = _parse_args(argv)
    assert align == "center"


def test_parse_args_align_right():
    argv = ["prog", "file.md", "--align", "right"]
    _pattern, align, _normalize, _comments = _parse_args(argv)
    assert align == "right"


def test_parse_args_all_flags():
    argv = ["prog", "*.md", "--normalize", "--comments", "--align", "left"]
    pattern, align, normalize, comments = _parse_args(argv)
    assert pattern == "*.md"
    assert normalize is True
    assert comments is True
    assert align == "left"


def test_parse_args_invalid_align():
    import pytest

    argv = ["prog", "file.md", "--align", "invalid"]
    with pytest.raises(SystemExit):
        _parse_args(argv)


def test_parse_args_no_flags():
    import pytest

    argv = ["prog", "file.md"]
    with pytest.raises(SystemExit):
        _parse_args(argv)


def test_parse_args_no_args():
    import pytest

    argv = ["prog"]
    with pytest.raises(SystemExit):
        _parse_args(argv)


# %%
# --- _process_file ---


def test_process_file_normalize(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("# heading\ntext")
    _process_file(str(md_file), normalize=True, comments=False, align=None)
    content = md_file.read_text()
    assert "\n\n" in content


def test_process_file_no_change(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("plain text")
    _process_file(str(md_file), normalize=False, comments=False, align=None)
    content = md_file.read_text()
    assert content == "plain text"


def test_process_file_with_comments(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("```python\nx = 1  # comment\n```")
    _process_file(str(md_file), normalize=False, comments=True, align=None)
    content = md_file.read_text()
    assert "# comment" in content


def test_process_file_with_align(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("| a | b |\n|---|---|\n| 1 | 2 |")
    _process_file(str(md_file), normalize=False, comments=False, align="left")
    content = md_file.read_text()
    assert "|" in content
