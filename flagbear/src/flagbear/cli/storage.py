#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: storage.py
#   Author: UBeaRLy
#   Created: 2026-08-13 09:50:00
#   Updated: 2026-08-13 09:50:00
#   Description:
#     Command-line interface for local file storage.
# ---------------------------------------------------------
"""Command-line interface for local file storage.

Provides CLI entry point for managing local file-based key-value storage.

Examples:
    # List all keys
    flagbear-storage list --dir ./cache

    # List keys with prefix
    flagbear-storage list --dir ./cache --prefix "user:"

    # Get value
    flagbear-storage get --dir ./cache --key mykey

    # Set value from file
    flagbear-storage set --dir ./cache --key mykey --file data.bin

    # Set value from stdin
    echo "hello" | flagbear-storage set --dir ./cache --key mykey

    # Check if key exists
    flagbear-storage exists --dir ./cache --key mykey

    # Delete key
    flagbear-storage delete --dir ./cache --key mykey
"""

# %%
from __future__ import annotations

import logging
import sys

from flagbear.slp.storage import LocalFileStorage

# %%
logger = logging.getLogger(__name__)


# %%
def _parse_args(
    argv: list[str],
) -> tuple[str, str | None, str | None, str | None, str | None]:
    """Parse command-line arguments.

    Params:
    ------------------------
    argv: Command-line arguments (typically sys.argv).

    Returns:
    ------------------------
    tuple: (command, dir, key, file, prefix)

    Raise:
    ------------------------
    SystemExit: If arguments are invalid.
    """
    if len(argv) < 2:
        _print_usage()
        sys.exit(1)

    command = argv[1]
    if command not in ("list", "get", "set", "exists", "delete"):
        logger.warning(f"Unknown command: {command}")
        _print_usage()
        sys.exit(1)

    dir_ = None
    key = None
    file = None
    prefix = ""

    i = 2
    while i < len(argv):
        arg = argv[i]
        if arg in ("--dir", "-d") and i + 1 < len(argv):
            dir_ = argv[i + 1]
            i += 2
        elif arg in ("--key", "-k") and i + 1 < len(argv):
            key = argv[i + 1]
            i += 2
        elif arg in ("--file", "-f") and i + 1 < len(argv):
            file = argv[i + 1]
            i += 2
        elif arg in ("--prefix", "-p") and i + 1 < len(argv):
            prefix = argv[i + 1]
            i += 2
        elif arg in ("--help", "-h"):
            _print_usage()
            sys.exit(0)
        else:
            logger.warning(f"Unknown argument: {arg}")
            _print_usage()
            sys.exit(1)

    # Validate required arguments.
    if command in ("get", "set", "exists", "delete") and key is None:
        logger.warning(f"--key is required for '{command}' command.")
        sys.exit(1)

    return command, dir_, key, file, prefix


def _print_usage() -> None:
    """Print usage information."""
    print(
        "Usage: flagbear-storage COMMAND [OPTIONS]\n"
        "\n"
        "Manage local file-based key-value storage.\n"
        "\n"
        "Commands:\n"
        "  list                 List all keys\n"
        "  get                  Get value by key\n"
        "  set                  Set value by key\n"
        "  exists               Check if key exists\n"
        "  delete               Delete key\n"
        "\n"
        "Options:\n"
        "  -d, --dir DIR        Storage directory (default: ./cache)\n"
        "  -k, --key KEY        Key name\n"
        "  -f, --file FILE      File to read/write data\n"
        "  -p, --prefix PREFIX  Filter keys by prefix (for list command)\n"
        "  -h, --help           Show this help message\n"
        "\n"
        "Examples:\n"
        "  flagbear-storage list --dir ./cache\n"
        "  flagbear-storage get --dir ./cache --key mykey\n"
        "  flagbear-storage set --dir ./cache --key mykey --file data.bin\n"
        '  echo "hello" | flagbear-storage set --dir ./cache --key mykey',
    )


# %%
def _cmd_list(
    storage: LocalFileStorage,
    prefix: str,
) -> None:
    """List keys in storage.

    Params:
    ------------------------
    storage: Storage instance.
    prefix: Key prefix filter.
    """
    keys = storage.list_keys(prefix)
    for key in sorted(keys):
        print(key)


def _cmd_get(
    storage: LocalFileStorage,
    key: str,
) -> None:
    """Get value by key.

    Params:
    ------------------------
    storage: Storage instance.
    key: Key to retrieve.
    """
    data = storage.get(key)
    if data is None:
        logger.warning(f"Key not found: {key}")
        sys.exit(1)
    sys.stdout.buffer.write(data)


def _cmd_set(
    storage: LocalFileStorage,
    key: str,
    file: str | None,
) -> None:
    """Set value by key.

    Params:
    ------------------------
    storage: Storage instance.
    key: Key to set.
    file: File to read data from, or None for stdin.
    """
    if file is not None:
        try:
            with open(file, "rb") as f:
                data = f.read()
        except Exception:
            logger.exception(f"Error reading file: {file}")
            sys.exit(1)
    else:
        if sys.stdin.isatty():
            logger.warning("No input provided.")
            sys.exit(1)
        data = sys.stdin.buffer.read()

    storage.set(key, data)
    print(f"Set key '{key}' ({len(data)} bytes)")


def _cmd_exists(
    storage: LocalFileStorage,
    key: str,
) -> None:
    """Check if key exists.

    Params:
    ------------------------
    storage: Storage instance.
    key: Key to check.
    """
    if storage.exists(key):
        print("true")
    else:
        print("false")
        sys.exit(1)


def _cmd_delete(
    storage: LocalFileStorage,
    key: str,
) -> None:
    """Delete key.

    Params:
    ------------------------
    storage: Storage instance.
    key: Key to delete.
    """
    if not storage.exists(key):
        logger.warning(f"Key not found: {key}")
        sys.exit(1)
    storage.delete(key)
    print(f"Deleted key '{key}'")


# %%
def main() -> None:
    """Command-line interface for local file storage."""
    command, dir_, key, file, prefix = _parse_args(sys.argv)

    storage = LocalFileStorage(base_dir=dir_ or "./cache")

    commands = {
        "list": lambda: _cmd_list(storage, prefix),
        "get": lambda: _cmd_get(storage, key),
        "set": lambda: _cmd_set(storage, key, file),
        "exists": lambda: _cmd_exists(storage, key),
        "delete": lambda: _cmd_delete(storage, key),
    }

    try:
        commands[command]()
    except Exception:
        logger.exception(f"Error executing '{command}' command")
        sys.exit(1)


if __name__ == "__main__":
    main()
