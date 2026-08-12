#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: table.py
#   Author: mimo-v2.5
#   Created: 2026-08-12 13:59:19
#   Updated: 2026-08-12 15:07:27
#   Description:
#     Markdown table parsing and alignment utilities.
# ---------------------------------------------------------
"""Markdown table parsing and alignment utilities.

Provides functions for parsing Markdown tables and aligning cell contents.
Supports CJK character width calculation and LaTeX block handling.
"""

# %%
from __future__ import annotations

from .str_width import str_width


# %%
def _find_matching_ticks(line: str, start: int, tick_count: int) -> int | None:
    """Find the closing backticks of the same count starting from position."""
    j = start
    while j < len(line):
        if line[j] == "`":
            close_count = 1
            while j + close_count < len(line) and line[j + close_count] == "`":
                close_count += 1
            if close_count == tick_count:
                return j
            j += close_count
        else:
            j += 1
    return None


def _find_matching_dollars(
    line: str, start: int, dollar_count: int
) -> int | None:
    """Find the closing dollars of the same count starting from position."""
    j = start
    while j < len(line):
        if line[j] == "$":
            close_count = 1
            while j + close_count < len(line) and line[j + close_count] == "$":
                close_count += 1
            if close_count == dollar_count:
                return j
            j += close_count
        else:
            j += 1
    return None


def _handle_backtick(line: str, i: int, current: str) -> tuple[str, int]:
    """Handle backtick sequences, return updated (current, new_index)."""
    tick_count = 1
    while i + tick_count < len(line) and line[i + tick_count] == "`":
        tick_count += 1
    close_pos = _find_matching_ticks(line, i + tick_count, tick_count)
    if close_pos is not None:
        current += line[i : close_pos + tick_count]
        i = close_pos + tick_count
    else:
        current += "`" * tick_count
        i += tick_count
    return current, i


def _handle_dollar(line: str, i: int, current: str) -> tuple[str, int]:
    """Handle dollar sequences, return updated (current, new_index)."""
    dollar_count = 1
    while i + dollar_count < len(line) and line[i + dollar_count] == "$":
        dollar_count += 1
    close_pos = _find_matching_dollars(line, i + dollar_count, dollar_count)
    if close_pos is not None:
        current += line[i : close_pos + dollar_count]
        i = close_pos + dollar_count
    else:
        current += "$" * dollar_count
        i += dollar_count
    return current, i


# %%
def parse_table_line(line: str) -> list[str]:
    """Parse a table line and extract cell contents.

    Supports treating | characters as normal content (not cell delimiters)
    within the following syntax:
    - Single backtick `code`: | within matched backtick pairs is preserved
    - Double backtick ``code with backtick``: can contain single backtick
    - Inline formula $...$: | within matched $ pairs is preserved
    - Block formula $$...$$: | within matched $$ pairs is preserved
      (supports multi-line, pre-processed by parse_table)
    - Unmatched backticks or $ are treated as normal characters

    Params:
    ------------------------
    line: Table line string, format: | cell1 | cell2 | ... |

    Returns:
    ------------------------
    list[str]: Cell content list (with leading/trailing empty cells removed).
    """
    cells = []
    current = ""
    i = 0

    while i < len(line):
        char = line[i]
        if char == "`":
            current, i = _handle_backtick(line, i, current)
        elif char == "$":
            current, i = _handle_dollar(line, i, current)
        elif char == "|":
            cells.append(current.strip())
            current = ""
            i += 1
        else:
            current += char
            i += 1

    if current:
        cells.append(current.strip())

    if cells and cells[0] == "":
        cells.pop(0)
    if cells and cells[-1] == "":
        cells.pop()

    return cells


# %%
def parse_table(lines: list[str]) -> tuple[list[list[str]], str, dict]:
    """Parse Markdown table lines and extract cell contents.

    Multi-line $$ block formulas are merged into a single line and stored
    after the $$OPEN$$ marker. latex_rows records the original lines and
    first-row cell count of multi-line $$ blocks.

    Params:
    ------------------------
    lines: List of Markdown table lines.

    Returns:
    ------------------------
    tuple: (table, indent, latex_rows)
    """
    table = []
    indent = ""
    in_latex_block = False
    latex_buf: list[str] = []
    latex_rows: dict[int, tuple[int, list[str]]] = {}
    latex_row_idx = -1
    first_cell_count = 0
    for line in lines:
        stripped = line.strip()
        if in_latex_block:
            latex_buf.append(line)
            if "$$" in stripped:
                in_latex_block = False
                merged = " ".join(ln.strip() for ln in latex_buf)
                table[latex_row_idx][-1] = "$$OPEN$$" + merged
                latex_rows[latex_row_idx] = (first_cell_count, latex_buf)
            continue
        if not stripped.startswith("|"):
            continue
        if not indent:
            indent = line[: len(line) - len(stripped)]
        cells = parse_table_line(stripped)
        table.append(cells)
        if stripped.count("$$") % 2 == 1:
            in_latex_block = True
            latex_row_idx = len(table) - 1
            first_cell_count = len(cells)
            latex_buf = [line]
    return table, indent, latex_rows


# %%
def _calc_prefix_width(n: int, col_widths: list[int]) -> int:
    """Calculate total display width of first n columns (including | separators)."""
    w = 2  # "| " prefix
    for ci in range(n):
        w += col_widths[ci] + 3  # cell width + " | " separator
    return w


def _format_cell(text: str, width: int, align: str) -> str:
    """Format cell content to specified width (left/center/right)."""
    pad = width - str_width(text)
    if align == "right":
        return " " * pad + text
    elif align == "center":
        left_pad = pad // 2
        return " " * left_pad + text + " " * (pad - left_pad)
    return text + " " * pad


def _format_first_latex_line(
    first_line: str,
    before_n: int,
    col_widths: list[int],
    align: str,
    indent: str,
) -> str:
    """Format the first line of a multi-line $$ block."""
    first_cells = parse_table_line(first_line.strip())
    before_cells = first_cells[:before_n] + [""] * (
        before_n - len(first_cells)
    )
    dollar_pos = first_line.find("$$")
    after_dollar = first_line[dollar_pos:]

    if before_n > 0:
        aligned_cells = [
            _format_cell(before_cells[i], col_widths[i], align)
            for i in range(before_n)
        ]
        return indent + "| " + " | ".join(aligned_cells) + " | " + after_dollar
    return indent + "| " + after_dollar


def _process_after_cells(
    after_cells: list[str],
    first_n: int,
    num_cols: int,
    col_widths: list[int],
    align: str,
) -> list[str]:
    """Process after_cells: remove leading/trailing empties, pad to expected length."""
    if after_cells and after_cells[0] == "":
        after_cells = after_cells[1:]
    while after_cells and after_cells[-1] == "":
        after_cells.pop()
    expected = max(num_cols - first_n, len(after_cells))
    after_cells = after_cells[:expected] + [""] * max(
        0, expected - len(after_cells)
    )
    return [
        _format_cell(
            ct,
            col_widths[first_n + ci] if first_n + ci < num_cols else 0,
            align,
        )
        for ci, ct in enumerate(after_cells)
    ]


def _format_last_latex_line(
    last_line: str,
    first_n: int,
    col_widths: list[int],
    num_cols: int,
    align: str,
    indent: str,
) -> str:
    """Format the last line of a multi-line $$ block."""
    dollar_pos = last_line.find("$$")
    if dollar_pos < 0:
        return indent + last_line

    before = last_line[: dollar_pos + 2]
    after_part = last_line[dollar_pos + 2 :]
    if after_part.startswith("|"):
        after_part = after_part[1:]
    after_cells = [c.strip() for c in after_part.split("|")]

    prefix_w = _calc_prefix_width(first_n, col_widths) - 2
    pad = prefix_w - str_width(before)
    padded_before = before + (" " * pad if pad > 0 else "")

    if after_cells:
        after_parts = _process_after_cells(
            after_cells, first_n, num_cols, col_widths, align
        )
        return indent + padded_before + "| " + " | ".join(after_parts) + " |"

    empty_parts = [
        _format_cell(
            "",
            col_widths[first_n + ci] if first_n + ci < num_cols else 0,
            align,
        )
        for ci in range(num_cols - first_n)
    ]
    return indent + padded_before + "| " + " | ".join(empty_parts) + " |"


# %%
def _fmt_latex_row(
    orig_lines: list[str],
    first_n: int,
    col_widths: list[int],
    num_cols: int,
    align: str,
    indent: str,
) -> list[str]:
    """Format multi-line $$ block output.

    Multi-line $$ blocks occupy a single cell position in the table, with
    effective width 0 so they don't affect other column alignment. Block
    content is output verbatim between the opening and closing $$ markers.

    Processing strategy:
    - First line: cells before $$ are aligned normally, $$ and its content
      are preserved as-is
    - Middle lines: block content is output line by line (preserving
      indentation and formatting)
    - Last line: padding aligns to the start position of first_n columns,
      cells after $$ are aligned by column width

    Params:
    ------------------------
    orig_lines: Original line list (containing first line, middle block
      content, and last line).
    first_n: Number of parsed cells in the first line (including the cell
      containing $$).
    col_widths: Maximum width of each column (excluding $$ blocks).
    num_cols: Total number of columns in the table.
    align: Alignment mode (left/center/right).
    indent: Table indentation string.

    Returns:
    ------------------------
    list[str]: Formatted line list.
    """
    before_n = first_n - 1
    result = []

    # First line
    result.append(
        _format_first_latex_line(
            orig_lines[0], before_n, col_widths, align, indent
        )
    )

    # Middle lines
    result.extend(indent + mid for mid in orig_lines[1:-1])

    # Last line
    result.append(
        _format_last_latex_line(
            orig_lines[-1], first_n, col_widths, num_cols, align, indent
        )
    )

    return result


# %%
def _calculate_col_widths(table: list[list[str]]) -> list[int]:
    """Calculate max width of each column (skip separator row and $$ blocks)."""
    num_cols = max(len(row) for row in table)
    col_widths = [0] * num_cols
    for ridx, row in enumerate(table):
        if ridx == 1:
            continue
        for i, cell in enumerate(row):
            if cell.startswith("$$OPEN$$"):
                continue
            col_widths[i] = max(col_widths[i], str_width(cell))
    return col_widths


def _find_non_empty_columns(
    table: list[list[str]], col_widths: list[int]
) -> list[int]:
    """Find columns that have content (excluding separator row and $$ blocks)."""
    num_cols = len(col_widths)
    non_empty_cols = []
    for ci in range(num_cols):
        has_content = False
        for ridx, row in enumerate(table):
            if ridx == 1:
                continue
            cell = row[ci] if ci < len(row) else ""
            if cell.strip() and not cell.startswith("$$OPEN$$"):
                has_content = True
                break
        if has_content:
            non_empty_cols.append(ci)
    return non_empty_cols


def _remove_blank_columns(
    table: list[list[str]],
    col_widths: list[int],
    non_empty_cols: list[int],
) -> tuple[list[list[str]], list[int]]:
    """Remove blank columns and rebuild table/widths."""
    table = [
        [row[ci] if ci < len(row) else "" for ci in non_empty_cols]
        for row in table
    ]
    col_widths = [col_widths[ci] for ci in non_empty_cols]
    return table, col_widths


def _make_separator_row(col_widths: list[int], align: str, indent: str) -> str:
    """Generate separator row based on alignment."""
    sep = []
    for w in col_widths:
        if align == "right":
            sep.append("-" * (w + 1) + ":")
        elif align == "center":
            sep.append(":" + "-" * w + ":")
        else:
            sep.append("-" * (w + 2))
    return indent + "|" + "|".join(sep) + "|"


def _make_row(
    cells: list[str], col_widths: list[int], align: str, indent: str
) -> str:
    """Format a list of cells into a Markdown table row."""
    parts = [
        _format_cell(cells[i] if i < len(cells) else "", col_widths[i], align)
        for i in range(len(col_widths))
    ]
    return indent + "| " + " | ".join(parts) + " |"


# %%
def align_table(
    table: list[list[str]],
    align: str = "left",
    indent: str = "",
    latex_rows: dict | None = None,
) -> list[str]:
    """Align all cells in the table.

    Multi-line $$ blocks are treated as having effective width 0 and do not
    participate in column width calculation. Block content is handled by the
    _fmt_latex_row function for line-break output.

    Params:
    ------------------------
    table: Parsed 2D table list.
    align: Alignment mode (left/center/right).
    indent: Table indentation string.
    latex_rows: {row_index: (first_row_cell_count, original_line_list)}.

    Returns:
    ------------------------
    list[str]: Aligned table line list.
    """
    if not table:
        return []
    if latex_rows is None:
        latex_rows = {}

    col_widths = _calculate_col_widths(table)

    non_empty_cols = _find_non_empty_columns(table, col_widths)

    if len(non_empty_cols) < len(col_widths):
        table, col_widths = _remove_blank_columns(
            table, col_widths, non_empty_cols
        )
        latex_rows = {
            ridx: (
                sum(1 for c in non_empty_cols if c < fn),
                orig,
            )
            for ridx, (fn, orig) in latex_rows.items()
        }

    result = []
    for row_idx, row in enumerate(table):
        if row_idx == 1:
            result.append(_make_separator_row(col_widths, align, indent))
        elif row_idx in latex_rows:
            first_n, orig = latex_rows[row_idx]
            result.extend(
                _fmt_latex_row(
                    orig, first_n, col_widths, len(col_widths), align, indent
                )
            )
        else:
            result.append(_make_row(row, col_widths, align, indent))

    return result


# %%
def align_md_tables(content: str, align: str = "left") -> str:
    """Align tables in a Markdown string.

    This function:
    1. Identifies all tables (consecutive lines starting with |)
    2. Aligns each table
    3. Preserves non-table content unchanged
    4. Skips | symbols inside code blocks (not recognized as tables)

    Params:
    ------------------------
    content: Markdown string content.
    align: Alignment mode, options are "left", "center", or "right".

    Returns:
    ------------------------
    str: Processed content string.
    """
    lines = content.split("\n")
    result = []
    i = 0
    in_code_block = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 代码块状态跟踪
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            result.append(line)
            i += 1
            continue

        # Lines inside code blocks are preserved as-is, no table detection
        if in_code_block:
            result.append(line)
            i += 1
            continue

        if len(stripped) > 1 and stripped.startswith("|"):
            table_lines = []
            in_latex_block = False
            while i < len(lines):
                current = lines[i]
                current_stripped = current.strip()
                if in_latex_block:
                    table_lines.append(current)
                    if "$$" in current_stripped:
                        in_latex_block = False
                    i += 1
                elif current_stripped.startswith("|"):
                    table_lines.append(current)
                    if current_stripped.count("$$") % 2 == 1:
                        in_latex_block = True
                    i += 1
                else:
                    break

            table, indent, latex_rows = parse_table(table_lines)
            aligned = align_table(table, align, indent, latex_rows)
            result.extend(aligned)
        else:
            result.append(line)
            i += 1

    return "\n".join(result)
