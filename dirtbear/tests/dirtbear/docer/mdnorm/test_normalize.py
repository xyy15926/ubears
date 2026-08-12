#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_normalize.py
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

from dirtbear.docer.mdnorm.normalize import (
    _align_heading,
    _align_list_item,
    _handle_skip_zone,
    _normalize_md,
)


# %%
def test_normalize_md_heading_spacing():
    content = "text\n# heading\ntext"
    result = _normalize_md(content)
    lines = result.split("\n")
    assert lines[1] == ""
    assert lines[3] == ""


def test_normalize_md_consecutive_blank_lines():
    content = "line1\n\n\n\nline2"
    result = _normalize_md(content)
    assert "\n\n\n" not in result


def test_normalize_md_frontmatter():
    content = "---\ntitle: test\n---\n# heading"
    result = _normalize_md(content)
    lines = result.split("\n")
    assert lines[0] == "---"
    assert lines[1] == "title: test"
    assert lines[2] == "---"


def test_normalize_md_code_block():
    content = "text\n```\ncode\twith\ttabs\n```\ntext"
    result = _normalize_md(content)
    assert "    " in result.split("\n")[2]


def test_normalize_md_list_alignment():
    content = "- item1\n  - nested"
    result = _normalize_md(content)
    assert "- " in result


# %%
# --- _align_heading ---


def test_align_heading_h1():
    result = _align_heading("# text")
    assert result == "#   text"


def test_align_heading_h2():
    result = _align_heading("## text")
    assert result == "##  text"


def test_align_heading_h3():
    result = _align_heading("### text")
    assert result == "### text"


def test_align_heading_h6():
    result = _align_heading("###### text")
    assert result == "######  text"


def test_align_heading_no_match():
    result = _align_heading("not a heading")
    assert result == "not a heading"


def test_align_heading_already_aligned():
    result = _align_heading("### text")
    assert result == "### text"


# %%
# --- _align_list_item ---


def test_align_list_item_unordered_dash():
    result = _align_list_item("- item")
    assert result.startswith("-")
    assert "item" in result


def test_align_list_item_unordered_star():
    result = _align_list_item("* item")
    assert result.startswith("*")
    assert "item" in result


def test_align_list_item_unordered_plus():
    result = _align_list_item("+ item")
    assert result.startswith("+")
    assert "item" in result


def test_align_list_item_ordered():
    result = _align_list_item("1. first")
    assert result.startswith("1.")
    assert "first" in result


def test_align_list_item_ordered_double_digit():
    result = _align_list_item("10. tenth")
    assert result.startswith("10.")
    assert "tenth" in result


def test_align_list_item_nested():
    result = _align_list_item("  - nested")
    assert "  " in result
    assert "nested" in result


def test_align_list_item_no_match():
    result = _align_list_item("not a list")
    assert result == "not a list"


def test_align_list_item_preserves_text():
    result = _align_list_item("- hello world")
    assert "hello world" in result


# %%
# --- _normalize_md additional cases ---


def test_normalize_md_ordered_list():
    content = "1. first\n2. second\n3. third"
    result = _normalize_md(content)
    lines = result.split("\n")
    for line in lines:
        assert line.startswith("1.") or line.startswith("2.") or line.startswith("3.")


def test_normalize_md_nested_unordered_list():
    content = "- outer\n  - inner\n    - deep"
    result = _normalize_md(content)
    lines = result.split("\n")
    assert lines[0].startswith("-")
    assert lines[1].startswith(" ")


def test_normalize_md_latex_block():
    content = "text\n$$\nx + y\n$$\ntext"
    result = _normalize_md(content)
    assert "$$" in result


def test_normalize_md_consecutive_headings():
    content = "# h1\n## h2\n### h3"
    result = _normalize_md(content)
    lines = result.split("\n")
    blank_count = sum(1 for line in lines if line.strip() == "")
    assert blank_count >= 2


def test_normalize_md_heading_at_start():
    content = "# heading\ntext"
    result = _normalize_md(content)
    lines = result.split("\n")
    assert lines[0].startswith("#")
    assert lines[1] == ""
    assert lines[2] == "text"


def test_normalize_md_heading_at_end():
    content = "text\n# heading"
    result = _normalize_md(content)
    lines = result.split("\n")
    assert lines[-3] == "text"
    assert lines[-2] == ""
    assert lines[-1].startswith("#")


def test_normalize_md_code_block_tab_replace():
    content = "```\nline1\twith\ttabs\n```"
    result = _normalize_md(content)
    lines = result.split("\n")
    assert "    " in lines[1]
    assert "\t" not in lines[1]


def test_normalize_md_trailing_whitespace():
    content = "text   \n# heading  \ntext   "
    result = _normalize_md(content)
    for line in result.split("\n"):
        assert line == line.rstrip()


def test_normalize_md_frontmatter_preserved():
    content = "---\ntitle: test\nauthor: me\n---\n# heading"
    result = _normalize_md(content)
    lines = result.split("\n")
    assert lines[0] == "---"
    assert lines[1] == "title: test"
    assert lines[2] == "author: me"
    assert lines[3] == "---"


# %%
# --- _handle_skip_zone ---


def test_handle_skip_zone_frontmatter_start():
    result = []
    state = {
        "in_frontmatter": False,
        "in_code_block": False,
        "in_latex_block": False,
        "prev_blank": False,
        "prev_was_heading": False,
    }
    handled = _handle_skip_zone("---", result, state)
    assert handled is True
    assert state["in_frontmatter"] is True
    assert result == ["---"]


def test_handle_skip_zone_frontmatter_end():
    result = ["---", "title: test"]
    state = {
        "in_frontmatter": True,
        "in_code_block": False,
        "in_latex_block": False,
        "prev_blank": False,
        "prev_was_heading": False,
    }
    handled = _handle_skip_zone("---", result, state)
    assert handled is True
    assert state["in_frontmatter"] is False
    assert len(result) == 3


def test_handle_skip_zone_code_fence():
    result = []
    state = {
        "in_frontmatter": False,
        "in_code_block": False,
        "in_latex_block": False,
        "prev_blank": False,
        "prev_was_heading": False,
    }
    handled = _handle_skip_zone("```python", result, state)
    assert handled is True
    assert state["in_code_block"] is True


def test_handle_skip_zone_inside_code_block():
    result = []
    state = {
        "in_frontmatter": False,
        "in_code_block": True,
        "in_latex_block": False,
        "prev_blank": False,
        "prev_was_heading": False,
    }
    handled = _handle_skip_zone("x = 1\twith\ttabs", result, state)
    assert handled is True
    assert "    " in result[0]


def test_handle_skip_zone_latex_block():
    result = []
    state = {
        "in_frontmatter": False,
        "in_code_block": False,
        "in_latex_block": True,
        "prev_blank": False,
        "prev_was_heading": False,
    }
    handled = _handle_skip_zone("  x + y", result, state)
    assert handled is True
    assert state["in_latex_block"] is True


def test_handle_skip_zone_latex_end():
    result = []
    state = {
        "in_frontmatter": False,
        "in_code_block": False,
        "in_latex_block": True,
        "prev_blank": False,
        "prev_was_heading": False,
    }
    handled = _handle_skip_zone("  = z$$", result, state)
    assert handled is True
    assert state["in_latex_block"] is False


def test_handle_skip_zone_normal_line():
    result = []
    state = {
        "in_frontmatter": False,
        "in_code_block": False,
        "in_latex_block": False,
        "prev_blank": False,
        "prev_was_heading": False,
    }
    handled = _handle_skip_zone("normal text", result, state)
    assert handled is False
    assert result == []


def test_handle_skip_zone_latex_start():
    result = []
    state = {
        "in_frontmatter": False,
        "in_code_block": False,
        "in_latex_block": False,
        "prev_blank": False,
        "prev_was_heading": False,
    }
    handled = _handle_skip_zone("$$", result, state)
    assert handled is False
    assert state["in_latex_block"] is True
