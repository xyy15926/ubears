#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: kline.py
#   Author: xyy15926
#   Created: 2024-11-29 12:13:36
#   Updated: 2026-04-20 22:16:06
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pyecharts import options as opts
from pyecharts.charts import Bar, Kline, Line
from pyecharts.commons.utils import JsCode
from pyecharts.globals import ThemeType

from dirtbear.visual.gridchart import GridChart

if TYPE_CHECKING:
    import pandas as pd

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")

RED = "#EF232A"
GREEN = "#14B14A"
YELLOW = "#FFD700"
PURPLE = "#9370DB"


# %%
def draw_kline(
    prices: pd.DataFrame,
    trades: pd.DataFrame,
    mas: dict[str, list],
    signals: dict[str, list] | None = None,
) -> GridChart:
    """Draw KLine with volume bars, trend lines.

    Params:
    ------------------------
    prices: Price data.
      date: Time tag.
      open_: Open price.
      close: Close price.
      high: High price.
      low: Low price.
      volume: Volume.
    trades: Trade records.
      date: Time tag.
      price: Buy or sell price.
      lotn: Lots of buy or sell.
      cash: Remaining cash.
      value: Sum of cash and stock.
    mas: {ma_type: moving average of prices, }
    signals: {signal_type: some kinds of scores, }

    Return:
    ------------------------
    GridChart
    """
    xticks = prices["date"].tolist()
    prices_ = prices[["open_", "close", "high", "low"]].values.tolist()
    volume_mark = (prices["close"] > prices["open_"]).astype(int) * 2 - 1
    volume = {
        "volume": list(
            zip(
                range(len(prices)),
                prices["volume"].values.tolist(),
                volume_mark,
                strict=True,
            )
        )
    }
    bs_points = trades[["date", "price", "lotn"]].values.tolist()
    cash = trades["cash"].values.tolist()
    stock = (trades["value"] - trades["cash"]).values.tolist()

    grid_chart = compose_kline(
        xticks,
        prices_,
        volume,
        mas,
        signals,
        bs_points,
        cash,
        stock,
    )
    return grid_chart


# %%
def compose_kline(
    xticks: list,
    prices: list,
    volume: list,
    mas: dict[str, list],
    signals: dict[str, list],
    bs_points: list[Any, float, int],
    cash: list,
    stock: list,
) -> GridChart:
    """Compose different parts of KLine.

    Params:
    -------------------------------
    xticks: Xaxis ticks, date in most cases.
    prices: [[open, close, high, low], ]
    volume: [[xidx, volume, pos or neg mark], ]
    mas: {ma_type: moving average of prices, }
    signals: {signal_type: some kinds of scores, }
    bs_points: [[xtick, price, lots of buying or selling], ]
    cash: Cash value.
    stock: Stock value.

    Return:
    -------------------------------
    GridChart
    """
    # Init Grid Chart.
    grid_chart = GridChart(
        init_opts=opts.InitOpts(
            width="1200px",
            height="800px",
            animation_opts=opts.AnimationOpts(animation=False),
            theme=ThemeType.INFOGRAPHIC,
        )
    )

    # Prices KLine.
    kline = prices_kline(xticks, prices, bs_points)

    # Additional lines.
    ma_line = None
    if mas is not None:
        ma_line = prices_lines(xticks, mas, line=None)
    if signals is not None:
        kline.extend_axis(yaxis=opts.AxisOpts(type_="value", position="right"))
        ma_line = signal_lines(xticks, signals, line=ma_line)
    if ma_line is not None:
        kline = kline.overlap(ma_line)
    grid_chart.add_chart(kline)

    # Bar of volume.
    if volume is not None:
        vol_bar = volume_bar(xticks, {"Volume": volume}, 0, 0)
        grid_chart.add_chart(vol_bar)

    # Bar of cash and stock value.
    if cash is not None:
        values_bar = volume_bar(xticks, {"Stock": stock, "Cash": cash}, 0, 0)
        grid_chart.add_chart(values_bar)

    grid_chart.set_defualt_opts()
    grid_chart.set_plain_layout([70, 15, 15], 100)
    return grid_chart


# %%
def prices_markpoints(
    xticks: list,
    prices: list,
) -> list[opts.MarkPointItem]:
    """Generate markpoints on KLine."""
    points = [
        opts.MarkPointItem(
            name=None,
            coord=[x, y],
            value=f"{v}",
            symbol="arrow" if v > 0 else "diamond",
            symbol_size=8,
            itemstyle_opts=opts.ItemStyleOpts(
                color=YELLOW if v > 0 else PURPLE
            ),
        )
        for x, y, v in prices
        if v != 0
    ]
    label_opts = opts.LabelOpts(
        is_show=True,
        position="left",
        color="#FFFFFF",
        font_size=10,
        font_style="normal",
        font_weight="bold",
        background_color="rgba(0,0,0,0.5)",
        border_color="#FFFFFF",
        border_width=1,
    )

    return points, label_opts


# %%
def prices_kline(
    xticks: list,
    prices: list,
    bs_points: list,
    name: str = "Prices",
) -> Kline:
    """Draw KLine."""
    markpoints, mp_label_opts = prices_markpoints(xticks, bs_points)
    # Basic Kline.
    kline = (
        Kline()
        .add_xaxis(xticks)
        .add_yaxis(
            series_name=name,
            y_axis=prices,
            itemstyle_opts=opts.ItemStyleOpts(
                color=RED,  # Rising red
                color0=GREEN,  # Falling green
                border_color=RED,
                border_color0=GREEN,
            ),
            markpoint_opts=opts.MarkPointOpts(
                data=markpoints,
                label_opts=mp_label_opts,
            ),
        )
    )

    return kline


# %%
def prices_lines(
    xticks: list,
    prices: dict[str, list],
    xaxis_index: int = 0,
    yaxis_index: int = 0,
    line: Line = None,
) -> Line:
    """Draw lines on KLine."""
    line = line or Line().add_xaxis(xaxis_data=xticks)
    for name, price in prices.items():
        line.add_yaxis(
            series_name=name,
            y_axis=price,
            xaxis_index=xaxis_index,
            yaxis_index=yaxis_index,
            is_smooth=False,
            linestyle_opts=opts.LineStyleOpts(
                width=2,
                opacity=0.8,
                # color=next(COLOR_ITER)
            ),
            label_opts=opts.LabelOpts(is_show=False),
        )

    return line


# %%
def signal_lines(
    xticks: list,
    signals: dict[str, list],
    xaxis_index: int = 0,
    yaxis_index: int = 1,
    line: Line = None,
) -> Line:
    """Draw lines on KLine but rely on another yaxis by default."""
    return prices_lines(
        xticks,
        signals,
        xaxis_index,
        yaxis_index,
        line,
    )


# %%
def volume_bar(
    xticks: list,
    volume: list | dict,
    xaxis_index: int = 1,
    yaxis_index: int = 2,
    bar: Bar = None,
):
    """Draw bar with two color determined by the mark passed."""
    bar = bar or Bar().add_xaxis(xaxis_data=xticks)
    if isinstance(volume, list):
        volume = {"": volume}
    for ser_name, vol in volume.items():
        if isinstance(vol[0], (tuple, list)) and len(vol[0]) == 3:
            itemstyle_opts = opts.ItemStyleOpts(
                color=JsCode(f"""
                    function(params) {{
                        console.log(params.values);
                        if (params.value instanceof Array){{
                            return params.value[2] > 0 ? '{RED}': '{GREEN}';
                        }} else {{
                            return 'YELLOW';
                        }}
                    }}
                """),
            )
        else:
            itemstyle_opts = None

        bar.add_yaxis(
            series_name=ser_name,
            y_axis=vol,
            xaxis_index=xaxis_index,
            yaxis_index=yaxis_index,
            stack=f"stack{yaxis_index}",
            itemstyle_opts=itemstyle_opts,
            label_opts=opts.LabelOpts(is_show=False),
        )
    bar.set_global_opts(
        yaxis_opts=opts.AxisOpts(is_show=False),
        legend_opts=opts.LegendOpts(is_show=False),
    )

    return bar
