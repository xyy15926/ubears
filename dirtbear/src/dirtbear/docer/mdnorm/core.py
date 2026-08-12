#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: core.py
#   Author: mimo-v2.5
#   Created: 2026-08-12 13:59:19
#   Updated: 2026-08-12 15:07:27
#   Description:
#     Core Markdown processing utilities.
# ---------------------------------------------------------
"""Core Markdown processing utilities.

Provides the main process_md_string function for Markdown normalization,
table alignment, and code comment alignment.
"""

# %%
from __future__ import annotations

from .code_comments import _align_code_comments
from .normalize import _normalize_md
from .table import align_md_tables


# %%
def process_md_string(
    content: str,
    normalize: bool = False,
    comments: bool = False,
    align: str | None = None,
) -> str:
    """Process a Markdown string directly, without a file path.

    Params:
    ------------------------
    content: Markdown text content.
    normalize: Normalize (tabs->spaces, heading/list alignment, etc.).
    comments: Align trailing comments in code blocks.
    align: Alignment mode (left/center/right), None means no alignment.

    Returns:
    ------------------------
    str: Processed string.
    """
    if normalize:
        content = _normalize_md(content)
    if comments:
        content = _align_code_comments(content)
    if align is not None:
        content = align_md_tables(content, align)
    return content
