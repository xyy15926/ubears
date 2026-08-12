#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: normalize.py
#   Author: mimo-v2.5
#   Created: 2026-08-12 13:59:19
#   Updated: 2026-08-12 15:07:27
#   Description:
#     Markdown normalization utilities.
# ---------------------------------------------------------
"""Markdown normalization utilities.

Provides functions for normalizing Markdown files including:
- Tab to space conversion
- Heading alignment
- List item alignment
- Blank line normalization
"""

# %%
from __future__ import annotations

import re


# %%
def _align_heading(line: str) -> str:
    """Align heading hash marks to 4-char boundary.

    Params:
    ------------------------
    line: Line starting with # markers.

    Returns:
    ------------------------
    str: Heading with aligned spacing.
    """
    match = re.match(r"^(#{1,6})\s+(.*)", line)
    if not match:
        return line
    hashes = match.group(1)
    text = match.group(2)
    current_len = len(hashes) + 1
    target_len = ((current_len + 3) // 4) * 4
    return hashes + " " * (target_len - len(hashes)) + text


def _align_list_item(line: str) -> str:
    """Align list item content to 4-char boundary.

    Supports unordered (-, *, +) and ordered (1. 2. etc.) markers.

    Params:
    ------------------------
    line: Line starting with a list marker.

    Returns:
    ------------------------
    str: List item with aligned spacing.
    """
    match = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)", line)
    if not match:
        return line
    m_indent = match.group(1)
    marker = match.group(2)
    text = match.group(3)
    indent_width = len(m_indent)
    marker_len = len(marker)
    new_indent_width = (
        0 if indent_width == 0 else ((indent_width + 3) // 4) * 4
    )
    target = ((new_indent_width + marker_len + 1 + 3) // 4) * 4
    gap = max(target - new_indent_width - marker_len, 1)
    return " " * new_indent_width + marker + " " * gap + text


# %%
def _handle_skip_zone(
    line: str,
    result: list[str],
    state: dict[str, bool],
) -> bool:
    """Handle lines in skip zones (frontmatter, code block, LaTeX block).

    Params:
    ------------------------
    line: Current line.
    result: Output list to append to.
    state: Mutable state dict with keys: in_frontmatter, in_code_block,
      in_latex_block, prev_blank, prev_was_heading.

    Returns:
    ------------------------
    bool: True if the line was handled (caller should continue),
      False if the line should be processed normally.
    """
    if not result and line.strip() == "---":
        state["in_frontmatter"] = True
        result.append(line)
        return True

    if state["in_frontmatter"]:
        result.append(line)
        if line.strip() == "---":
            state["in_frontmatter"] = False
        return True

    if line.strip().startswith("```"):
        state["in_code_block"] = not state["in_code_block"]
        result.append(line)
        state["prev_blank"] = False
        state["prev_was_heading"] = False
        return True

    if state["in_code_block"]:
        result.append(line.replace("\t", "    "))
        state["prev_blank"] = False
        state["prev_was_heading"] = False
        return True

    if state["in_latex_block"]:
        result.append(line.replace("\t", "    "))
        if "$$" in line:
            state["in_latex_block"] = False
        state["prev_blank"] = False
        state["prev_was_heading"] = False
        return True

    if line.strip().count("$$") % 2 == 1:
        state["in_latex_block"] = True

    return False


# %%
def _normalize_md(content: str) -> str:
    """Normalize Markdown content.

    This function:
    1. Skips frontmatter (--- block at file start)
    2. Replaces all tab characters with 4 spaces
    3. Normalizes heading spacing to align to multiples of 4
    4. Aligns list item content to multiples of 4 (unordered lists - * +,
       ordered lists 1. 2. etc.)
    5. Ensures blank lines before and after headings
    6. Removes trailing whitespace from each line
    7. Removes consecutive blank lines outside code blocks

    Params:
    ------------------------
    content: Markdown string content.

    Returns:
    ------------------------
    str: Normalized content string.
    """
    lines = content.split("\n")
    result = []
    state = {
        "in_code_block": False,
        "in_frontmatter": False,
        "in_latex_block": False,
        "prev_blank": False,
        "prev_was_heading": False,
    }

    for line in lines:
        if _handle_skip_zone(line, result, state):
            continue

        line = line.replace("\t", "    ")
        is_heading = bool(re.match(r"^(#{1,6})\s+", line))

        if (is_heading and result and result[-1].strip()) or (
            not is_heading and line.strip() and state["prev_was_heading"]
        ):
            result.append("")
            state["prev_blank"] = True

        line = _align_heading(line) if is_heading else _align_list_item(line)
        line = line.rstrip()

        if not line:
            if state["prev_blank"]:
                continue
            state["prev_blank"] = True
            state["prev_was_heading"] = False
        else:
            state["prev_blank"] = False

        result.append(line)
        state["prev_was_heading"] = is_heading

    return "\n".join(result)


# %%
def normalize_md_file(filepath: str) -> str:
    """Normalize a Markdown file: replace tabs with 4 spaces, align headings and lists to positions 4, 8, 12, 16, etc.

    This function:
    1. Skips frontmatter (--- block at file start)
    2. Replaces all tab characters with 4 spaces
    3. Normalizes heading spacing to align to multiples of 4
    4. Aligns list item content to multiples of 4 (unordered lists - * +,
       ordered lists 1. 2. etc.)
    5. Ensures blank lines before and after headings
    6. Removes trailing whitespace from each line
    7. Removes consecutive blank lines outside code blocks

    Params:
    ------------------------
    filepath: Path to the Markdown file.

    Returns:
    ------------------------
    str: Processed content string.
    """
    with open(filepath, encoding="utf-8") as f:
        return _normalize_md(f.read())
