#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: __init__.py
#   Author: mimo-v2.5
#   Created: 2026-08-12 13:59:19
#   Updated: 2026-08-12 15:07:27
#   Description:
#     Markdown normalization utilities package.
# ---------------------------------------------------------
"""Markdown normalization utilities package.

Provides functions for normalizing Markdown files including:
- Table alignment (left/center/right)
- Code comment alignment
- Heading and list alignment
- CJK character width calculation
"""

from __future__ import annotations

from .code_comments import align_code_comments_file
from .core import process_md_string
from .normalize import normalize_md_file
from .str_width import str_width
from .table import align_md_tables, align_table, parse_table, parse_table_line

__all__ = [
    "align_code_comments_file",
    "align_md_tables",
    "align_table",
    "normalize_md_file",
    "parse_table",
    "parse_table_line",
    "process_md_string",
    "str_width",
]
