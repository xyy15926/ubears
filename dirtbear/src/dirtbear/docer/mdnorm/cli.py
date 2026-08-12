#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: cli.py
#   Author: mimo-v2.5
#   Created: 2026-08-12 13:59:19
#   Updated: 2026-08-12 15:07:27
#   Description:
#     Command-line interface for mdnorm utilities.
# ---------------------------------------------------------
"""Command-line interface for mdnorm utilities.

Provides CLI entry point for Markdown normalization, table alignment,
and code comment alignment.
"""

# %%
from __future__ import annotations

import glob
import logging
import os
import sys

from .core import process_md_string

logger = logging.getLogger(__name__)


def _parse_args(
    argv: list[str],
) -> tuple[str, str | None, bool, bool]:
    """Parse command-line arguments.

    Params:
    ------------------------
    argv: Command-line arguments (typically sys.argv).

    Returns:
    ------------------------
    tuple: (pattern, align, normalize, comments)

    Raise:
    ------------------------
    SystemExit: If arguments are invalid.
    """
    if len(argv) < 2:
        logger.warning(
            "Usage: python align_table.py <markdown file or glob "
            "pattern> [--align left|center|right] [--normalize] "
            "[--comments]"
        )
        sys.exit(1)

    pattern = argv[1]
    align = None
    normalize = False
    comments = False

    for i, arg in enumerate(argv[2:], 2):
        if arg == "--align" and i + 1 < len(argv):
            align = argv[i + 1]
            if align not in ("left", "center", "right"):
                logger.warning("Alignment must be left, center, or right")
                sys.exit(1)
        elif arg == "--normalize":
            normalize = True
        elif arg == "--comments":
            comments = True

    if not normalize and not comments and align is None:
        logger.warning(
            "No operation flag provided. Use --normalize, "
            "--comments, or --align."
        )
        sys.exit(1)

    return pattern, align, normalize, comments


def _process_file(
    filepath: str,
    normalize: bool,
    comments: bool,
    align: str | None,
) -> None:
    """Process a single Markdown file.

    Params:
    ------------------------
    filepath: Path to the Markdown file.
    normalize: Whether to normalize content.
    comments: Whether to align code comments.
    align: Alignment mode, or None.
    """
    with open(filepath, encoding="utf-8") as f:
        content = f.read()
    processed = process_md_string(content, normalize, comments, align)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(processed)
    ops = []
    if normalize:
        ops.append("normalize")
    if comments:
        ops.append("comments")
    if align is not None:
        ops.append("align")
    logger.info(f"{'+'.join(ops)}: {filepath}")


# %%
def main() -> None:
    """Command-line interface for mdnorm utilities."""
    pattern, align, normalize, comments = _parse_args(sys.argv)

    files = glob.glob(pattern, recursive=True)
    if not files:
        logger.error(f"No files matched: {pattern}")
        sys.exit(1)

    for filepath in files:
        if not os.path.isfile(filepath):
            continue
        try:
            _process_file(filepath, normalize, comments, align)
        except Exception:
            logger.exception(f"Error processing {filepath}")


if __name__ == "__main__":
    main()
