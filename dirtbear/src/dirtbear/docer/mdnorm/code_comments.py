#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: code_comments.py
#   Author: mimo-v2.5
#   Created: 2026-08-12 13:59:19
#   Updated: 2026-08-12 15:07:27
#   Description:
#     Code comment alignment utilities.
# ---------------------------------------------------------
"""Code comment alignment utilities.

Provides functions for aligning trailing comments in Markdown code blocks.
"""

# %%
from __future__ import annotations

import re

from .str_width import str_width


# %%
def _parse_code_line(
    line: str,
) -> tuple[str, str | None, str, bool] | None:
    """Parse a code line into (indent, code, comment, pure_comment).

    Params:
    ------------------------
    line: A single line from a code block.

    Returns:
    ------------------------
    tuple or None: (indent, code, comment, pure_comment) if the line contains
      a comment, None otherwise. pure_comment is True if the line is only a
      comment (no code before it).
    """
    if line.strip().startswith("```"):
        return None
    match = re.match(r"^(\s*)(\S.*?)\s+((?:#|//)\s+.*)$", line)
    if match:
        return match.group(1), match.group(2), match.group(3), False
    match = re.match(r"^(\s*)((?:#|//)\s+.*)$", line)
    if match:
        return match.group(1), None, match.group(2), True
    return None


def _align_block_comments(code_lines: list) -> list[str]:
    """Align comments within a single code block.

    If there are code+comment lines, aligns all comments to the same column.
    Otherwise, outputs lines as-is.

    Params:
    ------------------------
    code_lines: List of parsed code lines (tuples) or raw strings.

    Returns:
    ------------------------
    list[str]: Formatted lines.
    """
    comment_items = [
        x for x in code_lines if isinstance(x, tuple) and not x[3]
    ]
    result = []
    if comment_items:
        max_code_len = max(
            str_width(indent + code) for indent, code, _, _ in comment_items
        )
        for item in code_lines:
            if isinstance(item, tuple):
                indent, code, comment, pure_comment = item
                if pure_comment:
                    result.append(indent + comment)
                else:
                    padding = " " * (
                        max_code_len - str_width(indent + code) + 1
                    )
                    result.append(indent + code + padding + comment)
            else:
                result.append(item)
    else:
        for item in code_lines:
            if isinstance(item, tuple):
                indent, code, comment, pure_comment = item
                if pure_comment:
                    result.append(indent + comment)
                else:
                    result.append(indent + code + " " + comment)
            else:
                result.append(item)
    return result


# %%
def _align_code_comments(content: str) -> str:
    """Align trailing comments after code lines in code blocks.

    This function:
    1. Identifies code blocks in the Markdown file
    2. Detects comments after code lines (starting with # or //)
    3. Aligns these comments to the same column position
    4. Does not align lines that contain only comments (pure comment lines)
    5. Skips lex code blocks

    Params:
    ------------------------
    content: Markdown string content.

    Returns:
    ------------------------
    str: Processed content string.
    """
    lines = content.split("\n")
    in_code_block = False
    skip_block = False

    aligned_result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        is_fence = line.strip().startswith("```")

        if is_fence:
            if not in_code_block:
                lang = line.strip()[3:].strip()
                skip_block = lang in ("lex",)
            else:
                skip_block = False
            in_code_block = not in_code_block
            aligned_result.append(line)
            i += 1
            continue

        if not in_code_block or skip_block:
            aligned_result.append(line)
            i += 1
            continue

        code_lines = []
        while i < len(lines) and not lines[i].strip().startswith("```"):
            parsed = _parse_code_line(lines[i])
            if parsed is not None:
                code_lines.append(parsed)
            else:
                code_lines.append(lines[i])
            i += 1

        aligned_result.extend(_align_block_comments(code_lines))

    return "\n".join(aligned_result)


# %%
def align_code_comments_file(filepath: str) -> str:
    """Align trailing comments after code lines in code blocks in a file.

    This function:
    1. Reads the Markdown file
    2. Identifies code blocks in the Markdown file
    3. Detects comments after code lines (starting with # or //)
    4. Aligns these comments to the same column position
    5. Does not align lines that contain only comments (pure comment lines)
    6. Skips lex code blocks

    Params:
    ------------------------
    filepath: Path to the Markdown file.

    Returns:
    ------------------------
    str: Processed content string.
    """
    with open(filepath, encoding="utf-8") as f:
        return _align_code_comments(f.read())
