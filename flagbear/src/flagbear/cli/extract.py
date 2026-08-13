#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: extract.py
#   Author: UBeaRLy
#   Created: 2026-08-13 09:50:00
#   Updated: 2026-08-13 09:50:00
#   Description:
#     Command-line interface for JSON field extraction.
# ---------------------------------------------------------
"""Command-line interface for JSON field extraction.

Provides CLI entry point for extracting fields from JSON data using
path expressions with support for conditions and aggregations.

Examples:
    # Extract from stdin
    echo '{"a":{"b":1}}' | flagbear-extract "a:b"

    # Extract from file
    flagbear-extract --file data.json --path "users:0:name"

    # With type conversion
    flagbear-extract --file data.json --path "count" --dtype INT

    # With aggregation
    flagbear-extract --file data.json --path "items:{len}"

    # Output as JSON
    echo '{"a":1}' | flagbear-extract "a" --json
"""

# %%
from __future__ import annotations

import json
import logging
import sys
from typing import Any

from flagbear.str2.fliper import extract_field

# %%
logger = logging.getLogger(__name__)


# %%
def _parse_flag(
    arg: str,
    argv: list[str],
    i: int,
    flags: dict[str, str | bool],
) -> tuple[int, bool]:
    """Parse a single flag from arguments.

    Params:
    ------------------------
    arg: Current argument.
    argv: All arguments.
    i: Current index.
    flags: Dict mapping flag names to values.

    Returns:
    ------------------------
    tuple: (new_index, was_handled)
    """
    if arg in ("--help", "-h"):
        _print_usage()
        sys.exit(0)
    if arg in ("--forced", "-F"):
        flags["forced"] = True
        return i + 1, True
    if arg in ("--json", "-j"):
        flags["as_json"] = True
        return i + 1, True
    if arg == "--default" and i + 1 < len(argv):
        flags["default"] = argv[i + 1]
        return i + 2, True
    return i, False


def _parse_args(
    argv: list[str],
) -> tuple[str | None, str, str | None, str | None, bool, bool]:
    """Parse command-line arguments.

    Params:
    ------------------------
    argv: Command-line arguments (typically sys.argv).

    Returns:
    ------------------------
    tuple: (path, file, dtype, default, forced, as_json)

    Raise:
    ------------------------
    SystemExit: If arguments are invalid.
    """
    if len(argv) < 2:
        _print_usage()
        sys.exit(1)

    path = None
    file = None
    dtype = None
    flags: dict[str, str | bool] = {
        "default": None,
        "forced": False,
        "as_json": False,
    }

    i = 1
    while i < len(argv):
        arg = argv[i]
        i, handled = _parse_flag(arg, argv, i, flags)
        if handled:
            continue
        if arg in ("--path", "-p") and i + 1 < len(argv):
            path = argv[i + 1]
            i += 2
        elif arg in ("--file", "-f") and i + 1 < len(argv):
            file = argv[i + 1]
            i += 2
        elif arg in ("--dtype", "-d") and i + 1 < len(argv):
            dtype = argv[i + 1].upper()
            i += 2
        else:
            if path is None and not arg.startswith("-"):
                path = arg
            else:
                logger.warning(f"Unknown argument: {arg}")
                _print_usage()
                sys.exit(1)
            i += 1

    if path is None:
        logger.warning("Path expression is required.")
        _print_usage()
        sys.exit(1)

    return (
        path,
        file,
        dtype,
        flags["default"],
        flags["forced"],
        flags["as_json"],
    )


def _print_usage() -> None:
    """Print usage information."""
    print(
        "Usage: flagbear-extract PATH [OPTIONS]\n"
        "\n"
        "Extract fields from JSON data using path expressions.\n"
        "\n"
        "Arguments:\n"
        '  PATH                 Path expression (e.g., "a:b:c", "items:{len}")\n'
        "\n"
        "Options:\n"
        "  -f, --file FILE      Read JSON from file (reads from stdin if\n"
        "                       not provided)\n"
        "  -d, --dtype TYPE     Target data type for conversion: INT, FLOAT,\n"
        "                       DATE, DATETIME, AUTO\n"
        "  --default VALUE      Default value on extraction failure\n"
        "  -F, --forced         Force using default value on any failure\n"
        "  -j, --json           Output result as JSON\n"
        "  -h, --help           Show this help message\n"
        "\n"
        "Path Expression Syntax:\n"
        "  a:b:c          Navigate nested dicts\n"
        "  items:{len}    Apply aggregation on values\n"
        "  items:[0]      Apply aggregation on items\n"
        "  a&&b=1:c       Conditional extraction\n"
        "\n"
        "Examples:\n"
        '  echo \'{"a":{"b":1}}\' | flagbear-extract "a:b"\n'
        '  flagbear-extract --file data.json --path "users:0:name"\n'
        '  flagbear-extract --file data.json --path "items:{len}" --json',
    )


def _format_output(result: Any, as_json: bool) -> str:
    """Format output value.

    Params:
    ------------------------
    result: Value to format.
    as_json: Whether to output as JSON.

    Returns:
    ------------------------
    Formatted string.
    """
    if as_json:
        return json.dumps(result, ensure_ascii=False, indent=2)
    elif isinstance(result, (dict, list)):
        return json.dumps(result, ensure_ascii=False)
    else:
        return str(result)


# %%
def main() -> None:
    """Command-line interface for JSON field extraction."""
    path, file, dtype, default, forced, as_json = _parse_args(sys.argv)

    # Read input.
    if file is not None:
        try:
            with open(file, encoding="utf-8") as f:
                data = f.read()
        except Exception:
            logger.exception(f"Error reading file: {file}")
            sys.exit(1)
    else:
        if sys.stdin.isatty():
            logger.warning(
                "No input provided. Use --help for usage information."
            )
            sys.exit(1)
        data = sys.stdin.read()

    # Extract field.
    try:
        result = extract_field(
            data,
            path,
            dtype=dtype,
            dfill=default,
            dforced=forced,
        )
        print(_format_output(result, as_json))
    except Exception:
        logger.exception(f"Error extracting field: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
