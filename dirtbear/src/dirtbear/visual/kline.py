#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: kline.py
#   Author: xyy15926
#   Created: 2024-11-29 12:13:36
#   Updated: 2026-04-16 14:56:43
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import Dict, List

import logging
import itertools
import pandas as pd
from pyecharts import options as opts
from pyecharts.globals import ThemeType
from pyecharts.charts import Kline, Line, Bar
from pyecharts.commons.utils import JsCode

from dirtbear.visual.gridchart import GridChart

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
class ColorIterator:
    def __init__(self, colors=None):
        if colors is None:
            self.colors = [
                "#5470c6", "#91cc75", "#fac858", "#ee6666",
                "#73c0de", "#3ba272", "#fc8452", "#9a60b4",
                "#ea7ccc",
            ]
        else:
            self.colors = colors
        self.cycle = itertools.cycle(self.colors)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.cycle)

    def getn(self, n=1):
        return [next(self) for _ in range(n)]

COLOR_ITER = ColorIterator()
RED = "#EF232A"
GREEN = "#14B14A"
YELLOW = "#FFD700"
PURPLE = "#9370DB"


# %%
def compose_kline(
    prices: pd.DataFrame,
    trades: pd.DataFrame,
    mas: Dict[str, list],
    signals: Dict[str, list] = None,
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
      value: Sum of cash and share.
    mas: Moving average of prices.
    signals: Signals line of some score.

    Return:
    ------------------------
    GridChart
    """
    xticks = prices["date"].tolist()
    prices_ = prices[["open_", "close", "high", "low"]].values.tolist()
    volume_mark = (prices["close"] > prices["open_"]).astype(int) * 2 - 1
    volumes = list(zip(
        range(len(prices)),
        prices["volume"].values.tolist(),
        volume_mark,
        strict=True,
    ))
    bs_points = trades[["date", "price", "lotn" ]].values.tolist()
    cash = trades["cash"].values.tolist()
    svalue = (trades["value"] - trades["cash"]).values.tolist()

    # Prices KLine.
    kline = prices_kline(xticks, prices_, bs_points)
    kline.extend_axis(
        yaxis=opts.AxisOpts(type_="value", position="right")
    )
    ma_line = prices_lines(xticks, mas, line=None)
    if signals is not None:
        ma_line = signal_lines(xticks, signals, line=ma_line)
    kline = kline.overlap(ma_line)

    # Volume Bar.
    vol_bar = volume_bar(xticks, volumes, 1, 2)

    # Value.
    values_bar = volume_bar(xticks, {"cash": cash, "share": svalue}, 2, 3)

    grid_chart = GridChart(init_opts = opts.InitOpts(
        width="1200px",
        height="800px",
        animation_opts=opts.AnimationOpts(animation=False),
        theme=ThemeType.INFOGRAPHIC,
    ))
    grid_chart.add_chart(
        kline,
        opts.GridOpts(
            pos_left="5%",
            pos_right="5%",
            pos_top="0%",
            height="60%",
        ),
        1, 2,
    ).add_chart(
        vol_bar,
        grid_opts=opts.GridOpts(
            pos_left="5%",
            pos_right="5%",
            pos_top="65%",
            height="15%",
        ),
    ).add_chart(
        values_bar,
        grid_opts=opts.GridOpts(
            pos_left="5%",
            pos_right="5%",
            pos_top="85%",
            height="5%",
        ),
    )
    grid_chart.set_defualt_opts()

    return grid_chart


# %%
def prices_markpoints(
    xticks: list,
    prices: list,
) -> List[opts.MarkPointItem]:
    points = [
        opts.MarkPointItem(
            name=None,
            coord=[x, y],
            value=f"{v}@{y}",
            symbol="arrow" if v > 0 else "diamond",
            symbol_size=8,
            itemstyle_opts=opts.ItemStyleOpts(
                color=YELLOW if v > 0 else PURPLE
            )
        )
        for x, y, v in prices if v != 0
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
    name: str = "prices",
) -> Kline:
    markpoints, mp_label_opts = prices_markpoints(xticks, bs_points)
    # Basic Kline.
    kline = (
        Kline()
        .add_xaxis(xticks)
        .add_yaxis(
            series_name=name,
            y_axis=prices,
            itemstyle_opts=opts.ItemStyleOpts(
                color=RED,                      # Rising red
                color0=GREEN,                   # Falling green
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
    prices: Dict[str, list],
    xaxis_index: int = 0,
    yaxis_index: int = 0,
    line: Line = None,
) -> Line:
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
                color=next(COLOR_ITER)
            ),
            label_opts=opts.LabelOpts(is_show=False),
        )

    return line


# %%
def signal_lines(
    xticks: list,
    signals: Dict[str, list],
    xaxis_index: int = 0,
    yaxis_index: int = 1,
    line: Line = None,
) -> Line:
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
    volumes: list | dict,
    xaxis_index: int = 1,
    yaxis_index: int = 2,
    bar: Bar = None,
):
    bar = bar or Bar().add_xaxis(xaxis_data=xticks)
    if isinstance(volumes, list):
        volumes = {"": volumes}
    for ser_name, vol in volumes.items():
        color = next(COLOR_ITER)
        bar.add_yaxis(
            series_name=ser_name,
            y_axis=vol,
            xaxis_index=xaxis_index,
            yaxis_index=yaxis_index,
            itemstyle_opts=opts.ItemStyleOpts(
                color=JsCode(f"""
                    function(params) {{
                        console.log(params.values);
                        if (params.value instanceof Array){{
                            return params.value[2] > 0 ? '{RED}': '{GREEN}';
                        }} else {{
                            return '{color}';
                        }}
                    }}
                """),
            ),
            stack=f"stack{yaxis_index}",
            label_opts=opts.LabelOpts(is_show=False),
        )
    bar.set_global_opts(
        yaxis_opts=opts.AxisOpts(is_show=False),
        legend_opts=opts.LegendOpts(is_show=False),
    )

    return bar
