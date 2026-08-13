#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: cast.py
#   Author: UBeaRLy
#   Created: 2026-08-13 09:50:00
#   Updated: 2026-08-13 09:50:00
#   Description:
#     Command-line interface for string type casting.
# ---------------------------------------------------------
"""Command-line interface for string type casting.

Provides CLI entry point for converting strings to specified data types
or auto-detecting types using regex patterns.

Examples:
    # Auto-detect type
    echo "2024-01-01" | flagbear-cast
    flagbear-cast "3.14"

    # Specify type
    flagbear-cast "2024/01/01" --dtype DATE
    flagbear-cast "123" --dtype FLOAT

    # Batch process file
    flagbear-cast --file data.txt --dtype INT --default 0

    # Force default value on failure
    flagbear-cast "abc" --dtype INT --default -1 --forced
"""

# %%
from __future__ import annotations

import logging
import sys
from typing import Any

from flagbear.str2.dtyper import regex_caster, str_caster

# %%
logger = logging.getLogger(__name__)


# %%
def _parse_args(
    argv: list[str],
) -> tuple[str | None, str, str | None, bool, bool, str | None]:
    """Parse command-line arguments.

    Params:
    ------------------------
    argv: Command-line arguments (typically sys.argv).

    Returns:
    ------------------------
    tuple: (value, dtype, file, extended, forced, default)

    Raise:
    ------------------------
    SystemExit: If arguments are invalid.
    """
    if len(argv) < 2:
        _print_usage()
        sys.exit(1)

    value = None
    dtype = "AUTO"
    file = None
    extended = False
    forced = False
    default = None

    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg in ("--dtype", "-d") and i + 1 < len(argv):
            dtype = argv[i + 1].upper()
            i += 2
        elif arg in ("--file", "-f") and i + 1 < len(argv):
            file = argv[i + 1]
            i += 2
        elif arg in ("--extended", "-e"):
            extended = True
            i += 1
        elif arg in ("--forced", "-F"):
            forced = True
            i += 1
        elif arg == "--default" and i + 1 < len(argv):
            default = argv[i + 1]
            i += 2
        elif arg in ("--help", "-h"):
            _print_usage()
            sys.exit(0)
        else:
            if value is None and not arg.startswith("-"):
                value = arg
            else:
                logger.warning(f"Unknown argument: {arg}")
                _print_usage()
                sys.exit(1)
            i += 1

    return value, dtype, file, extended, forced, default


def _print_usage() -> None:
    """Print usage information."""
    print(
        "Usage: flagbear-cast [VALUE] [OPTIONS]\n"
        "\n"
        "Convert strings to specified data types.\n"
        "\n"
        "Arguments:\n"
        "  VALUE                String value to convert (reads from stdin if\n"
        "                       not provided)\n"
        "\n"
        "Options:\n"
        "  -d, --dtype TYPE     Target data type: INT, FLOAT, DATE, DATETIME,\n"
        "                       AUTO (default: AUTO)\n"
        "  -f, --file FILE      Read values from file (one per line)\n"
        "  --default VALUE      Default value on conversion failure\n"
        "  -F, --forced         Force using default value on any failure\n"
        "  -e, --extended       Use extended mode (numpy types)\n"
        "  -h, --help           Show this help message\n"
        "\n"
        "Examples:\n"
        '  flagbear-cast "3.14" --dtype FLOAT\n'
        '  echo "2024-01-01" | flagbear-cast\n'
        "  flagbear-cast --file data.txt --dtype INT --default 0",
    )


def _cast_value(
    value: str,
    dtype: str,
    extended: bool,
    forced: bool,
    default: Any,
) -> Any:
    """Cast a single value.

    Params:
    ------------------------
    value: String value to cast.
    dtype: Target data type.
    extended: Use extended mode.
    forced: Force default on failure.
    default: Default value.

    Returns:
    ------------------------
    Casted value.
    """
    try:
        if dtype == "AUTO":
            result = regex_caster(value, extended=extended)
            return result[0] if result is not None else value
        else:
            return str_caster(
                value,
                dtype=dtype,
                extended=extended,
                dfill=default,
                dforced=forced,
            )
    except Exception as e:
        logger.warning(f"Cast failed for '{value}': {e}")
        if forced:
            return default if default is not None else value
        raise


# %%
def main() -> None:
    """Command-line interface for string type casting."""
    value, dtype, file, extended, forced, default = _parse_args(sys.argv)

    # Process file input.
    if file is not None:
        try:
            with open(file, encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    try:
                        result = _cast_value(
                            line, dtype, extended, forced, default
                        )
                        print(result)
                    except Exception:
                        logger.exception(f"Error casting: {line}")
                        if forced:
                            print(default if default is not None else line)
        except Exception:
            logger.exception(f"Error reading file: {file}")
            sys.exit(1)
        return

    # Process single value from argument or stdin.
    if value is None:
        if sys.stdin.isatty():
            logger.warning(
                "No input provided. Use --help for usage information."
            )
            sys.exit(1)
        value = sys.stdin.read().rstrip("\n")

    try:
        result = _cast_value(value, dtype, extended, forced, default)
        print(result)
    except Exception:
        logger.exception(f"Error casting: {value}")
        sys.exit(1)


if __name__ == "__main__":
    main()
