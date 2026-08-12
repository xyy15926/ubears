#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_code_comments.py
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

from dirtbear.docer.mdnorm.code_comments import (
    _align_block_comments,
    _align_code_comments,
    _parse_code_line,
)


# %%
def test_align_code_comments_single():
    content = "```python\nx = 1  # comment\n```"
    result = _align_code_comments(content)
    assert "# comment" in result


def test_align_code_comments_multiple():
    content = "```python\nx = 1  # short\nlong_variable_name = 2  # long comment\n```"
    result = _align_code_comments(content)
    lines = result.strip().split("\n")
    assert len(lines) >= 3


def test_align_code_comments_pure_comment():
    content = "```python\n# pure comment\n```"
    result = _align_code_comments(content)
    assert "# pure comment" in result


def test_align_code_comments_skip_lex():
    content = "```lex\nabc | def\n```"
    result = _align_code_comments(content)
    assert "abc | def" in result


def test_align_code_comments_no_code_block():
    content = "plain text"
    result = _align_code_comments(content)
    assert result == "plain text"


def test_align_code_comments_multiple_blocks():
    content = (
        "```python\n"
        "x = 1  # short\n"
        "long_var = 2  # long comment\n"
        "```\n"
        "text\n"
        "```python\n"
        "a = 3  # another\n"
        "```"
    )
    result = _align_code_comments(content)
    lines = result.split("\n")
    assert "x = 1" in lines[1]
    assert "a = 3" in lines[6]


def test_align_code_comments_mixed_styles():
    content = "```\nx = 1  # hash comment\ny = 2  // slash comment\n```"
    result = _align_code_comments(content)
    assert "# hash comment" in result
    assert "// slash comment" in result


def test_align_code_comments_no_comments():
    content = "```\nx = 1\ny = 2\n```"
    result = _align_code_comments(content)
    assert "x = 1" in result
    assert "y = 2" in result


def test_align_code_comments_only_pure_comments():
    content = "```\n# comment1\n# comment2\n```"
    result = _align_code_comments(content)
    assert "# comment1" in result
    assert "# comment2" in result


def test_align_code_comments_mixed_pure_and_code():
    content = "```\nx = 1  # code comment\n# pure comment\ny = 2  # another\n```"
    result = _align_code_comments(content)
    lines = result.split("\n")
    code_lines = [line for line in lines if "=" in line and "#" in line]
    assert len(code_lines) == 2


def test_align_code_comments_preserves_non_code():
    content = "plain text\n```\nx = 1  # c\n```\nmore text"
    result = _align_code_comments(content)
    assert "plain text" in result
    assert "more text" in result


# %%
# --- _parse_code_line ---


def test_parse_code_line_with_comment():
    result = _parse_code_line("x = 1  # comment")
    assert result == ("", "x = 1", "# comment", False)


def test_parse_code_line_with_slash_comment():
    result = _parse_code_line("y = 2  // comment")
    assert result == ("", "y = 2", "// comment", False)


def test_parse_code_line_pure_comment():
    result = _parse_code_line("# pure comment")
    assert result == ("", None, "# pure comment", True)


def test_parse_code_line_indented():
    result = _parse_code_line("    x = 1  # comment")
    assert result == ("    ", "x = 1", "# comment", False)


def test_parse_code_line_no_comment():
    result = _parse_code_line("x = 1")
    assert result is None


def test_parse_code_line_fence():
    result = _parse_code_line("```python")
    assert result is None


def test_parse_code_line_empty():
    result = _parse_code_line("")
    assert result is None


def test_parse_code_line_mixed_comment_styles():
    r1 = _parse_code_line("a  # hash")
    r2 = _parse_code_line("b  // slash")
    assert r1[2] == "# hash"
    assert r2[2] == "// slash"


# %%
# --- _align_block_comments ---


def test_align_block_comments_with_code_comments():
    code_lines = [
        ("", "x = 1", "# short", False),
        ("", "long_var = 2", "# long comment", False),
    ]
    result = _align_block_comments(code_lines)
    assert len(result) == 2
    assert "# short" in result[0]
    assert "# long comment" in result[1]


def test_align_block_comments_only_pure():
    code_lines = [
        ("", None, "# comment1", True),
        ("", None, "# comment2", True),
    ]
    result = _align_block_comments(code_lines)
    assert len(result) == 2
    assert "# comment1" in result[0]
    assert "# comment2" in result[1]


def test_align_block_comments_mixed():
    code_lines = [
        ("", "x = 1", "# code comment", False),
        ("", None, "# pure comment", True),
    ]
    result = _align_block_comments(code_lines)
    assert len(result) == 2


def test_align_block_comments_raw_strings():
    code_lines = ["raw line 1", "raw line 2"]
    result = _align_block_comments(code_lines)
    assert result == ["raw line 1", "raw line 2"]


def test_align_block_comments_mixed_raw_and_parsed():
    code_lines = [
        ("", "x = 1", "# comment", False),
        "raw line",
    ]
    result = _align_block_comments(code_lines)
    assert len(result) == 2
