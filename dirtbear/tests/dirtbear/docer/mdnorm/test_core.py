#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_core.py
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

from dirtbear.docer.mdnorm.core import process_md_string


# %%
def test_process_md_string_normalize():
    content = "# heading\ntext"
    result = process_md_string(content, normalize=True)
    assert "\n\n" in result


def test_process_md_string_comments():
    content = "```python\nx = 1  # comment\n```"
    result = process_md_string(content, comments=True)
    assert "# comment" in result


def test_process_md_string_align():
    content = "| a | b |\n|---|---|\n| 1 | 2 |"
    result = process_md_string(content, align="left")
    assert "|" in result


def test_process_md_string_all():
    content = "# heading\n\n| a | b |\n|---|---|\n| 1 | 2 |"
    result = process_md_string(content, normalize=True, align="left")
    assert "|" in result


def test_process_md_string_no_options():
    content = "# heading"
    result = process_md_string(content)
    assert result == content


def test_process_md_string_normalize_and_align():
    content = "# heading\n\n| a | b |\n|---|---|\n| 1 | 2 |"
    result = process_md_string(content, normalize=True, align="left")
    assert "|" in result
    assert "\n\n" in result


def test_process_md_string_comments_and_align():
    content = "```python\nx = 1  # comment\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |"
    result = process_md_string(content, comments=True, align="left")
    assert "# comment" in result
    assert "|" in result


def test_process_md_string_all_three():
    content = (
        "# heading\n"
        "```python\n"
        "x = 1  # comment\n"
        "```\n"
        "| a | b |\n"
        "|---|---|\n"
        "| 1 | 2 |"
    )
    result = process_md_string(
        content, normalize=True, comments=True, align="left"
    )
    assert "#" in result
    assert "|" in result
