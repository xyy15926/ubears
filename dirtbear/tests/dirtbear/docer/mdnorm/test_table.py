#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_table.py
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

from dirtbear.docer.mdnorm.str_width import str_width
from dirtbear.docer.mdnorm.table import (
    _calc_prefix_width,
    _calculate_col_widths,
    _find_matching_dollars,
    _find_matching_ticks,
    _find_non_empty_columns,
    _fmt_latex_row,
    _format_cell,
    _format_first_latex_line,
    _format_last_latex_line,
    _handle_backtick,
    _handle_dollar,
    _make_row,
    _make_separator_row,
    _process_after_cells,
    _remove_blank_columns,
    align_md_tables,
    align_table,
    parse_table,
    parse_table_line,
)


# %%
def test_parse_table_line_simple():
    result = parse_table_line("| a | b | c |")
    assert result == ["a", "b", "c"]


def test_parse_table_line_no_leading_pipe():
    result = parse_table_line("a | b | c")
    assert result == ["a", "b", "c"]


def test_parse_table_line_empty_cells():
    result = parse_table_line("| | | |")
    assert result == ["", ""]


def test_parse_table_line_with_backtick():
    result = parse_table_line("| `a|b` | c |")
    assert result == ["`a|b`", "c"]


def test_parse_table_line_with_double_backtick():
    result = parse_table_line("| ``a|b`` | c |")
    assert result == ["``a|b``", "c"]


def test_parse_table_line_with_dollar():
    result = parse_table_line("| $a|b$ | c |")
    assert result == ["$a|b$", "c"]


def test_parse_table_line_with_double_dollar():
    result = parse_table_line("| $$a|b$$ | c |")
    assert result == ["$$a|b$$", "c"]


# %%
def test_parse_table_simple():
    lines = [
        "| a | b |",
        "|---|---|",
        "| 1 | 2 |",
    ]
    table, indent, latex_rows = parse_table(lines)
    assert table == [["a", "b"], ["---", "---"], ["1", "2"]]
    assert indent == ""
    assert latex_rows == {}


def test_parse_table_with_indent():
    lines = [
        "  | a | b |",
        "  |---|---|",
        "  | 1 | 2 |",
    ]
    _table, indent, _latex_rows = parse_table(lines)
    assert indent == "  "


def test_parse_table_skips_non_table():
    lines = [
        "not a table",
        "| a | b |",
        "|---|---|",
    ]
    table, _indent, _latex_rows = parse_table(lines)
    assert len(table) == 2


# %%
def test_align_table_left():
    table = [
        ["Name", "Age"],
        ["---", "---"],
        ["Alice", "30"],
        ["Bob", "25"],
    ]
    result = align_table(table, align="left")
    assert "| Name  | Age |" in result[0]
    assert "| Alice | 30  |" in result[2]
    assert "| Bob   | 25  |" in result[3]


def test_align_table_center():
    table = [
        ["Name", "Age"],
        ["---", "---"],
        ["Alice", "30"],
    ]
    result = align_table(table, align="center")
    assert ":---:" in result[1] or ":---" in result[1]


def test_align_table_right():
    table = [
        ["Name", "Age"],
        ["---", "---"],
        ["Alice", "30"],
    ]
    result = align_table(table, align="right")
    assert "-:" in result[1]


def test_align_table_empty():
    result = align_table([])
    assert result == []


def test_align_table_with_indent():
    table = [
        ["a", "b"],
        ["---", "---"],
        ["1", "2"],
    ]
    result = align_table(table, indent="  ")
    assert result[0].startswith("  ")


# %%
# --- _find_matching_ticks / _find_matching_dollars ---


def test_find_matching_ticks_matched():
    assert _find_matching_ticks("`abc`", 1, 1) == 4
    assert _find_matching_ticks("``abc``", 2, 2) == 5


def test_find_matching_ticks_no_match():
    assert _find_matching_ticks("`abc", 1, 1) is None
    assert _find_matching_ticks("``abc`", 2, 2) is None


def test_find_matching_ticks_mismatched_count():
    assert _find_matching_ticks("`abc``", 1, 1) is None
    assert _find_matching_ticks("``abc`", 2, 2) is None


def test_find_matching_dollars_matched():
    assert _find_matching_dollars("$a$", 1, 1) == 2
    assert _find_matching_dollars("$$a$$", 2, 2) == 3


def test_find_matching_dollars_no_match():
    assert _find_matching_dollars("$a", 1, 1) is None
    assert _find_matching_dollars("$$a$", 2, 2) is None


# %%
# --- _handle_backtick / _handle_dollar ---


def test_handle_backtick_matched():
    current, i = _handle_backtick("`code`", 0, "")
    assert current == "`code`"
    assert i == 6


def test_handle_backtick_unmatched():
    current, i = _handle_backtick("`code", 0, "pre")
    assert current == "pre`"
    assert i == 1


def test_handle_backtick_double():
    current, i = _handle_backtick("``code``", 0, "")
    assert current == "``code``"
    assert i == 8


def test_handle_dollar_matched():
    current, i = _handle_dollar("$x$", 0, "")
    assert current == "$x$"
    assert i == 3


def test_handle_dollar_unmatched():
    current, i = _handle_dollar("$x", 0, "pre")
    assert current == "pre$"
    assert i == 1


def test_handle_dollar_double():
    current, i = _handle_dollar("$$x$$", 0, "")
    assert current == "$$x$$"
    assert i == 5


# %%
# --- parse_table_line edge cases ---


def test_parse_table_line_unmatched_backtick():
    result = parse_table_line("| `a|b | c |")
    assert result == ["`a", "b", "c"]


def test_parse_table_line_unmatched_dollar():
    result = parse_table_line("| $a|b | c |")
    assert result == ["$a", "b", "c"]


def test_parse_table_line_mixed_backtick_dollar():
    result = parse_table_line("| `a|$b` | c |")
    assert result == ["`a|$b`", "c"]


def test_parse_table_line_triple_backtick():
    result = parse_table_line("| ```a|b``` | c |")
    assert result == ["```a|b```", "c"]


def test_parse_table_line_empty():
    result = parse_table_line("")
    assert result == []


def test_parse_table_line_single_pipe():
    result = parse_table_line("|")
    assert result == []


def test_parse_table_line_no_pipe():
    result = parse_table_line("hello world")
    assert result == ["hello world"]


def test_parse_table_line_mixed_unmatched():
    result = parse_table_line("| `a | $b | c |")
    assert result == ["`a", "$b", "c"]


# %%
# --- parse_table edge cases ---


def test_parse_table_empty():
    table, indent, latex_rows = parse_table([])
    assert table == []
    assert indent == ""
    assert latex_rows == {}


def test_parse_table_separator_only():
    lines = ["|---|---|"]
    table, _indent, _latex_rows = parse_table(lines)
    assert len(table) == 1
    assert table[0] == ["---", "---"]


def test_parse_table_multiline_latex():
    lines = [
        "| a | b |",
        "|---|---|",
        "| 1 | $$",
        "  x + y",
        "  = z$$ |",
    ]
    _table, _indent, latex_rows = parse_table(lines)
    assert 2 in latex_rows
    first_n, orig = latex_rows[2]
    assert first_n == 2
    assert len(orig) == 3


def test_parse_table_latex_no_before_cells():
    lines = [
        "| $$",
        "  x + y$$ |",
    ]
    _table, _indent, latex_rows = parse_table(lines)
    assert 0 in latex_rows


# %%
# --- _format_cell ---


def test_format_cell_left():
    assert _format_cell("ab", 5, "left") == "ab   "


def test_format_cell_right():
    assert _format_cell("ab", 5, "right") == "   ab"


def test_format_cell_center():
    result = _format_cell("ab", 5, "center")
    assert str_width(result) == 5
    assert "ab" in result


def test_format_cell_cjk():
    result = _format_cell("你好", 8, "left")
    assert str_width(result) == 8


def test_format_cell_exact_width():
    assert _format_cell("abc", 3, "left") == "abc"


# %%
# --- _calc_prefix_width ---


def test_calc_prefix_width_zero():
    assert _calc_prefix_width(0, [5, 10]) == 2


def test_calc_prefix_width_one():
    assert _calc_prefix_width(1, [5, 10]) == 2 + 5 + 3


def test_calc_prefix_width_two():
    assert _calc_prefix_width(2, [5, 10]) == 2 + 5 + 3 + 10 + 3


# %%
# --- _make_separator_row ---


def test_make_separator_row_left():
    result = _make_separator_row([5, 3], "left", "")
    assert result == "|-------|-----|"


def test_make_separator_row_right():
    result = _make_separator_row([5, 3], "right", "")
    assert result == "|------:|----:|"


def test_make_separator_row_center():
    result = _make_separator_row([5, 3], "center", "")
    assert result == "|:-----:|:---:|"


def test_make_separator_row_with_indent():
    result = _make_separator_row([5], "left", "  ")
    assert result.startswith("  |")


# %%
# --- _make_row ---


def test_make_row_left():
    result = _make_row(["a", "bb"], [3, 3], "left", "")
    assert result == "| a   | bb  |"


def test_make_row_right():
    result = _make_row(["a", "bb"], [3, 3], "right", "")
    assert result == "|   a |  bb |"


def test_make_row_center():
    result = _make_row(["a", "bb"], [3, 3], "center", "")
    assert " a " in result
    assert " bb " in result


def test_make_row_fewer_cells():
    result = _make_row(["a"], [3, 3], "left", "")
    assert result == "| a   |     |"


# %%
# --- _calculate_col_widths ---


def test_calculate_col_widths_simple():
    table = [["abc", "de"], ["---", "---"], ["a", "bcd"]]
    widths = _calculate_col_widths(table)
    assert widths == [3, 3]


def test_calculate_col_widths_skips_separator():
    table = [["a", "bb"], ["---", "---"], ["abc", "b"]]
    widths = _calculate_col_widths(table)
    assert widths[0] == 3
    assert widths[1] == 2


def test_calculate_col_widths_skips_latex():
    table = [["a", "$$OPEN$$x"], ["---", "---"], ["abc", "b"]]
    widths = _calculate_col_widths(table)
    assert widths == [3, 1]


# %%
# --- _find_non_empty_columns ---


def test_find_non_empty_columns_all_filled():
    table = [["a", "b"], ["---", "---"], ["c", "d"]]
    widths = [1, 1]
    cols = _find_non_empty_columns(table, widths)
    assert cols == [0, 1]


def test_find_non_empty_columns_one_empty():
    table = [["a", ""], ["---", "---"], ["c", ""]]
    widths = [1, 0]
    cols = _find_non_empty_columns(table, widths)
    assert cols == [0]


def test_find_non_empty_columns_skips_latex():
    table = [["$$OPEN$$x", "b"], ["---", "---"], ["$$OPEN$$y", "d"]]
    widths = [1, 1]
    cols = _find_non_empty_columns(table, widths)
    assert cols == [1]


# %%
# --- _remove_blank_columns ---


def test_remove_blank_columns():
    table = [["a", "", "b"], ["---", "---", "---"], ["c", "", "d"]]
    widths = [1, 0, 1]
    non_empty = [0, 2]
    new_table, new_widths = _remove_blank_columns(table, widths, non_empty)
    assert new_table == [["a", "b"], ["---", "---"], ["c", "d"]]
    assert new_widths == [1, 1]


# %%
# --- align_table with blank columns and first_n adjustment ---


def test_align_table_blank_column_removed():
    table = [
        ["Name", "", "Age"],
        ["---", "---", "---"],
        ["Alice", "", "30"],
    ]
    result = align_table(table, align="left")
    assert len(result) == 3
    assert "Name" in result[0]
    assert "Age" in result[0]
    cols = result[0].split("|")
    assert len([c for c in cols if c.strip()]) == 2


def test_align_table_latex_with_blank_columns():
    table = [
        ["a", "", "b"],
        ["---", "---", "---"],
        ["1", "", "$$OPEN$$x+y$$"],
    ]
    latex_rows = {2: (2, ["| 1 |  | $$", "  x+y$$ |"])}
    result = align_table(table, align="left", latex_rows=latex_rows)
    assert len(result) >= 3


def test_align_table_cjk_alignment():
    table = [
        ["名称", "值"],
        ["---", "---"],
        ["你好", "42"],
    ]
    result = align_table(table, align="left")
    data_row = result[2]
    cells = data_row.split("|")
    assert str_width(cells[1].strip()) == str_width("名称")
    assert str_width(cells[2].strip()) == str_width("值")


def test_align_table_multiple_latex_blocks():
    table = [
        ["a", "b"],
        ["---", "---"],
        ["1", "$$OPEN$$x$$"],
        ["2", "$$OPEN$$y$$"],
    ]
    latex_rows = {
        2: (1, ["| 1 | $$", "  x$$ |"]),
        3: (1, ["| 2 | $$", "  y$$ |"]),
    }
    result = align_table(table, align="left", latex_rows=latex_rows)
    assert len(result) >= 4


# %%
# --- _fmt_latex_row ---


def test_fmt_latex_row_basic():
    orig_lines = ["| a | $$", "  x + y", "  = z$$ |"]
    col_widths = [3, 0]
    result = _fmt_latex_row(orig_lines, 2, col_widths, 2, "left", "")
    assert len(result) == 3
    assert "a" in result[0]
    assert "x + y" in result[1]
    assert "$$" in result[2]


def test_fmt_latex_row_no_before_cells():
    orig_lines = ["| $$", "  x$$ |"]
    col_widths = [0]
    result = _fmt_latex_row(orig_lines, 1, col_widths, 1, "left", "")
    assert len(result) == 2


def test_fmt_latex_row_with_indent():
    orig_lines = ["  | a | $$", "    x$$ |"]
    col_widths = [3, 0]
    result = _fmt_latex_row(orig_lines, 2, col_widths, 2, "left", "  ")
    assert all(r.startswith("  ") for r in result)


# %%
# --- _format_first_latex_line / _format_last_latex_line ---


def test_format_first_latex_line_with_before():
    result = _format_first_latex_line(
        "| a | $$", 1, [3], "left", ""
    )
    assert "a" in result
    assert "$$" in result


def test_format_first_latex_line_no_before():
    result = _format_first_latex_line(
        "| $$", 0, [], "left", ""
    )
    assert "$$" in result


def test_format_last_latex_line_with_after():
    result = _format_last_latex_line(
        "  = z$$ | b |", 2, [3, 3], 2, "left", ""
    )
    assert "$$" in result
    assert "b" in result


def test_format_last_latex_line_no_after():
    result = _format_last_latex_line(
        "  = z$$ |", 2, [3, 3], 2, "left", ""
    )
    assert "$$" in result


def test_format_last_latex_line_no_dollar():
    result = _format_last_latex_line(
        "  no dollar here", 2, [3], 2, "left", ""
    )
    assert "no dollar here" in result


# %%
# --- _process_after_cells ---


def test_process_after_cells_strips_empty():
    result = _process_after_cells(["", "a", "b", ""], 1, 3, [3, 3, 3], "left")
    assert len(result) == 2


def test_process_after_cells_pads_to_expected():
    result = _process_after_cells(["a"], 1, 3, [3, 3, 3], "left")
    assert len(result) == 2


# %%
# --- _format_cell with CJK in align_table ---


def test_align_table_cjk_center():
    table = [
        ["名称", "值"],
        ["---", "---"],
        ["你好", "42"],
    ]
    result = align_table(table, align="center")
    assert ":---" in result[1]


def test_align_table_cjk_right():
    table = [
        ["名称", "值"],
        ["---", "---"],
        ["你好", "42"],
    ]
    result = align_table(table, align="right")
    assert "-:" in result[1]


# %%
# --- align_md_tables ---


def test_align_md_tables_simple():
    content = "| a | b |\n|---|---|\n| 1 | 2 |"
    result = align_md_tables(content)
    assert "|" in result


def test_align_md_tables_in_code_block():
    content = "```\n| a | b |\n```\n| c | d |"
    result = align_md_tables(content)
    lines = result.split("\n")
    assert "| a | b |" in lines[1]


def test_align_md_tables_left():
    content = "| Name | Age |\n|---|---|\n| Alice | 30 |"
    result = align_md_tables(content, align="left")
    assert "|" in result


def test_align_md_tables_right():
    content = "| Name | Age |\n|---|---|\n| Alice | 30 |"
    result = align_md_tables(content, align="right")
    assert "|" in result


def test_align_md_tables_center():
    content = "| Name | Age |\n|---|---|\n| Alice | 30 |"
    result = align_md_tables(content, align="center")
    assert "|" in result


def test_align_md_tables_multiple_tables():
    content = (
        "| a | b |\n|---|---|\n| 1 | 2 |\n"
        "\n"
        "| c | d |\n|---|---|\n| 3 | 4 |"
    )
    result = align_md_tables(content, align="left")
    assert "| a" in result
    assert "| c" in result


def test_align_md_tables_surrounded_by_text():
    content = "before\n| a | b |\n|---|---|\n| 1 | 2 |\nafter"
    result = align_md_tables(content, align="left")
    assert "before" in result
    assert "after" in result
    assert "| a" in result


def test_align_md_tables_left_alignment():
    content = "| Name | Age |\n|---|---|\n| Alice | 30 |"
    result = align_md_tables(content, align="left")
    lines = result.split("\n")
    assert "Name" in lines[0]
    assert "Alice" in lines[2]


def test_align_md_tables_right_alignment():
    content = "| Name | Age |\n|---|---|\n| Alice | 30 |"
    result = align_md_tables(content, align="right")
    lines = result.split("\n")
    assert "-:" in lines[1]


def test_align_md_tables_center_alignment():
    content = "| Name | Age |\n|---|---|\n| Alice | 30 |"
    result = align_md_tables(content, align="center")
    lines = result.split("\n")
    assert ":---:" in lines[1]


def test_align_md_tables_in_multiple_code_blocks():
    content = (
        "```\n| a | b |\n```\n"
        "| c | d |\n|---|---|\n| 3 | 4 |\n"
        "```\n| e | f |\n```"
    )
    result = align_md_tables(content)
    assert "| a | b |" in result
    assert "| e | f |" in result


def test_align_md_tables_latex_block():
    content = (
        "| a | b |\n"
        "|---|---|\n"
        "| 1 | $$\n"
        "  x + y\n"
        "  = z$$ |"
    )
    result = align_md_tables(content, align="left")
    assert "|" in result
