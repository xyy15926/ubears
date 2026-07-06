#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: gridchart.py
#   Author: xyy15926
#   Created: 2026-04-15 22:56:58
#   Updated: 2026-04-18 18:58:54
#   Description:
# ---------------------------------------------------------

# %%
import logging
from typing import Self

from pyecharts import options as opts
from pyecharts.charts import Grid
from pyecharts.charts.chart import Chart, RectChart

# from IPython.core.debugger import set_trace

logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
class GridChart(Grid):
    """Enhanced Grid chart.

    1. GridChart could `set_global_options` by updating the `Grid.options`.
    2. GridChart will trace the number of xaxis and yaxis of contained chart,
      which will help to set some global options.

    Attrs:
    -----------------------------
    charts: Int.
      Contained charts.
    """

    def __init__(
        self,
        init_opts: opts.InitOpts | dict = None,
        render_opts: opts.RenderOpts | dict = None,
    ):
        """Init GridChart."""
        super().__init__(
            init_opts or opts.InitOpts(), render_opts or opts.InitOpts()
        )
        self.grid_xy = [(0, 0)]

    def cur_xidx(self) -> int:
        """Xaxis index for newly added chart."""
        if self.options is None:
            return 0
        return len(self.options.get("xAxis", []))

    def cur_yidx(self) -> int:
        """Yaxis index for newly added chart."""
        if self.options is None:
            return 0
        return len(self.options.get("yAxis", []))

    def cur_gidx(self) -> int:
        """Grid index for newly added chart."""
        if self.options is None:
            return 0
        return len(self.options.get("grid", []))

    def update_unrect_series(
        self,
        grids: list[opts.GridOpts | dict] | None = None,
    ):
        """Update the positions for series from non-RectChart.

        Params:
        ----------------------------
        grids: List of GridOpts that should be refered by non-RectCharts to
          determine their positions.
        """
        grids = grids or self.options["grid"]
        for ser in self.options.get("series", []):
            # Only series for non-RectChart will be added `grid_index`.
            grid_index = ser.get("grid_index")
            if grid_index is not None:
                grid_opts = grids[grid_index]
                grid_ = (
                    grid_opts.opts
                    if not isinstance(grid_opts, dict)
                    else grid_opts
                )
                for ele in ["top", "bottom", "left", "right", "width"]:
                    ser[ele] = grid_[ele]

    def add_chart(
        self,
        chart: Chart,
        grid_opts: opts.GridOpts | dict = None,
    ) -> Self:
        """Add chart.

        Params:
        ----------------------------
        chart: Chart.
        grid_opts: Chart's grid options.
          Remember to set the `option['grid']` later via `set_plain_layout`
          or `set_global_opts` if not passed.

        Return:
        ----------------------------
        self
        """
        # Update data series.
        for ser in chart.options.get("series", []):
            # Update xaxis and yaxis index for RectChart.
            if isinstance(chart, RectChart):
                if ser["xAxisIndex"] is None:
                    ser["xAxisIndex"] = self.cur_xidx()
                else:
                    ser["xAxisIndex"] += self.cur_xidx()
                if ser["yAxisIndex"] is None:
                    ser["yAxisIndex"] = self.cur_yidx()
                else:
                    ser["yAxisIndex"] += self.cur_yidx()
            # Update positions for non-RectChart which doesn't containing
            # `grid_index` to related to the grid.
            else:
                grid_ = (
                    grid_opts.opts
                    if not isinstance(grid_opts, dict)
                    else grid_opts
                )
                for ele in ["top", "bottom", "left", "right", "width"]:
                    ser[ele] = grid_[ele]
                # Add `grid_index` so to relate to the grid.
                ser["grid_index"] = self.cur_gidx()

            # Add yAaxisIndex as the prefix so to distinct the stack group.
            if ser.get("stack", None) is not None:
                ser["stack"] = f"{ser['yAxisIndex']}_{ser['stack']}"

        self.add(
            chart,
            grid_opts or opts.GridOpts(),
            grid_index=self.cur_gidx(),
            is_control_axis_index=True,
        )
        self.grid_xy.append((self.cur_xidx(), self.cur_yidx()))

        return self

    def set_global_opts(self, **kwargs) -> Self:
        """Set global options.

        Params:
        ----------------------------
        datazoom_opts: DataZoomOpts, equivalent dict or list of those.
        tooltip_opts: TooltipOpts or equivalent dict.
        axispointer_opts: AxisPointerOpts or equivalent dict.
        legend_opts: LegendOpts, equivalent dict or list of those.
        title_opts: TitleOpts, equivalent dict or list of those.
        grid_opts: List of GridOpts or equivalent dict of those.

        Return:
        ----------------------------
        Self
        """
        mapper = {
            "datazoom_opts": ("dataZoom", opts.DataZoomOpts),
            "tooltip_opts": ("tooltip", opts.TooltipOpts),
            "axispointer_opts": ("axisPointer", opts.AxisPointerOpts),
            "legend_opts": ("legend", opts.LegendOpts),
            "title_opts": ("title", opts.TitleOpts),
            "grid_opts": ("grid", opts.GridOpts),
            "color": ("color", lambda x: x),
        }
        for key, opt in kwargs.items():
            # set_trace()
            opt_key, opt_wrapper = mapper.get(key, (None, None))
            if opt_key is None:
                continue

            if isinstance(opt, dict):
                opt = opt_wrapper(**opt)
            elif (
                isinstance(opt, list)
                and len(opt) > 0
                and isinstance(opt[0], dict)
            ):
                opt = [opt_wrapper(**ele) for ele in opt]

            # `option["title"]` accepts only dict, while other options
            # could handle both dict and XXXOpts, legend for example.
            if opt_key == "title":
                opt = opt.opts[0]

            # Replace the options value directly.
            # As it seems that:
            # 1. All options accept both list and dict/Opt, while
            # 1.1 Some options, datazoom and title for example, will render
            #   all the components.
            # 1.2 Other options, tooltip for example, only render the exact
            #   first one.
            if not isinstance(opt, list):
                opt = [opt]
            self.options[opt_key] = opt

            # Update non-RectChart's positions.
            if opt_key == "grid":
                self.update_unrect_series()

        return self

    def set_datazoom(
        self,
        xaxis_index: list | str = "all",
        yaxis_index: list | str | None = None,
        xzoom_type: list = ("inside", "slider"),
        yzoom_type: list = ("inside",),
        *,
        datazoom_opts: dict | None = None,
    ) -> Self:
        """Set the datazoom.

        Params:
        -------------------------
        xaxis_index: List of the xaxis-index that would zoom in or zoom
          along the xaxis together.
          "all": All the xaxis.
        yaxis_index: List of the yaxis-index that would zoom in or zoom
          along the yaxis together.
          "all": All the charts.
        xzoom_type: List of datazoom type for xaxis.
        yzoom_type: List of datazoom type for yaxis.
        datazoom_opts: Dict of arguments of opts.DataZoomOpts.

        Return:
        -------------------------
        Self
        """
        conf = {
            "is_show": True,
            "type_": "inside",
            "is_disabled": False,
            "is_realtime": True,
            "is_zoom_lock": False,
            # Prefered percentage range settings.
            "range_start": 90,
            "range_end": 100,
            # min_span=20,
            "max_span": 100,
            # start_value=20,
            # end_value=100,
            # The `min_value_span` will determine the really `range_start` as
            # `range_start` and `range_end` are close.
            "min_value_span": 20,
            # max_value_span=100,
            # Control xaxis.
            "orient": "horizontal",
            "xaxis_index": [0, 1],
            # pos_top="90%",
            # pos_bottom="10%",
            # pos_left="90%",
            # pos_right="10%",
        }
        if datazoom_opts is not None:
            conf.update(datazoom_opts)

        if xaxis_index == "all":
            xaxis_index = list(range(self.cur_xidx()))
        if yaxis_index == "all":
            yaxis_index = list(range(self.cur_yidx()))

        datazoom_opts = []
        if xaxis_index is not None:
            for type_ in xzoom_type:
                conf["type_"] = type_
                conf["orient"] = "horizontal"
                conf["xaxis_index"] = xaxis_index
                conf["yaxis_index"] = None
                datazoom_opts.append(opts.DataZoomOpts(**conf))
        if yaxis_index is not None:
            for type_ in yzoom_type:
                conf["type_"] = type_
                conf["orient"] = "vertical"
                conf["xaxis_index"] = None
                conf["yaxis_index"] = yaxis_index
                datazoom_opts.append(opts.DataZoomOpts(**conf))

        self.set_global_opts(datazoom_opts=datazoom_opts)
        return self

    def set_tooltip(
        self,
        formatter: str | None = None,
        *,
        tooltip_opts: dict | None = None,
        axispointer_opts: dict | None = None,
    ) -> Self:
        """Set the tooltip.

        Params:
        ----------------------------
        formatter: String or JsCode for the tooltip formatter.
        tooltip_opts: Dict of arguments of opts.TooltipOpts.
        axispointer_opts: Dict of arguments of opts.AxisPointerOpts.

        Return:
        ----------------------------
        Self
        """
        tooltip_conf = {
            "is_show": True,
            "trigger": "axis",  # Instead of `item`
            "trigger_on": "mousemove|click",
            "axis_pointer_type": "cross",
            "is_show_content": True,
            "is_always_show_content": False,
            "show_delay": 0,
            "formatter": formatter,
            "background_color": "rgba(245, 245, 245, 0.5)",
            "border_width": 1,
            "border_color": "#ccc",
            "textstyle_opts": opts.TextStyleOpts(
                color="#000",
                font_weight="bolder",
                font_size=10,
                line_height=8,
            ),
        }
        if tooltip_opts is not None:
            tooltip_conf.update(tooltip_opts)
        tooltip_opts = opts.TooltipOpts(**tooltip_conf)

        # Only valid when the `axis_pointer_type = "cross"`.
        axispointer_conf = {
            "is_show": True,
            "link": [{"xAxisIndex": "all"}],
            "label": opts.LabelOpts(
                is_show=True,
                # position = "left",
                color="#FFFFFF",
                font_size=10,
                font_style="normal",
                font_weight="bold",
                background_color="rgba(0,0,0,0.5)",
                border_color="FFFFFF",
                border_width=1,
            ),
        }
        if axispointer_opts is not None:
            axispointer_conf.update(axispointer_opts)
        axispointer_opts = opts.AxisPointerOpts(**axispointer_conf)

        self.set_global_opts(
            tooltip_opts=tooltip_opts,
            axispointer_opts=axispointer_opts,
        )
        return self

    def set_legend(
        self,
        pos_top: int | str = 20,
        pos_left: int | str = "center",
        pos_bottom: int | str | None = None,
        pos_right: int | str | None = None,
        *,
        legend_opts: dict | None = None,
    ) -> Self:
        """Set the legend.

        Params:
        -------------------------
        pos_top: Position from the top.
        pos_left:
        pos_right:
        pos_bottom:
        legend_opts: Dict of arguments of opts.LegendOpts.
        """
        conf = {
            "is_show": True,
            "pos_top": pos_top,
            "pos_left": pos_left,
            "pos_bottom": pos_bottom,
            "pos_right": pos_right,
            "background_color": "rgba(0, 0, 0, 0.1)",
        }
        if legend_opts is not None:
            conf.update(legend_opts)
        legend_opts = opts.LegendOpts(**conf)

        self.set_global_opts(legend_opts=legend_opts)
        return self

    def set_title(
        self,
        title: str = "",
        pos_top: int | str = 0,
        pos_left: int | str = "center",
        font_size: int = 18,
        *,
        title_opts: dict | None = None,
    ) -> Self:
        """Set the title.

        Params:
        -------------------------
        title: The title content.
        pos_top: Position from the top.
        pos_left:
        font_size:
        title_opts: Dict of arguments of opts.TitleOpts.

        Return:
        -------------------------
        Self
        """
        conf = {
            "is_show": True,
            "title": title,
            "pos_top": pos_top,
            "pos_left": pos_left,
            "title_textstyle_opts": opts.TextStyleOpts(
                color="#111111",
                font_weight="normal",
                font_size=font_size,
            ),
        }
        if title_opts is not None:
            conf.update(title_opts)
        title_opts = opts.TitleOpts(**conf)

        self.set_global_opts(title_opts=title_opts)
        return self

    def set_color_map(self, color: list[str] | None = None) -> Self:
        """Set the color map.

        Params:
        -------------------------
        color: Color map, namely list of color.

        Return:
        -------------------------
        Self
        """
        self.set_global_opts(
            color=color
            or [
                "#5470c6",
                "#91cc75",
                "#fac858",
                "#ee6666",
                "#73c0de",
                "#3ba272",
                "#fc8452",
                "#9a60b4",
                "#ea7ccc",
            ]
        )
        return self

    def auto_margin(self) -> list[int]:
        """Estimate the margin."""
        # T, B, L, R
        margin = [10, 10, 60, 10]
        right_slider = 0
        right_axis = 0
        bottom_slider = 0
        for dz in self.options["dataZoom"]:
            if not isinstance(dz, dict):
                dz = dz.opts
            if (
                dz["show"]
                and dz["type"] == "slider"
                and dz["orient"] == "horizontal"
            ):
                bottom_slider = 50
            if (
                dz["show"]
                and dz["type"] == "slider"
                and dz["orient"] == "vertical"
            ):
                right_slider = 50
        for yax in self.options["yAxis"]:
            if not isinstance(yax, dict):
                yax = yax.opts
            if yax["show"] and yax["position"] == "right":
                right_axis = 50

        margin[1] += bottom_slider
        margin[3] += right_slider + right_axis

        return margin

    def set_plain_layout(
        self,
        heights: list[int] | int = 100,
        widths: list[int] | int = 100,
        margin: list[int] | None = None,
        gap: list[int] | None = None,
    ) -> Self:
        """Apply plain layout.

        Params:
        -------------------------
        heights: The percentage of heights of each row.
        widths: The percentage of widths of each columns.
        margin: Margin with unit px.

        Return:
        -------------------------
        Self
        """
        if isinstance(heights, int):
            heights = [heights]
        if isinstance(widths, int):
            widths = [widths]

        # Calculate the length of one percent of height and width.
        margin = margin or self.auto_margin()
        gap = gap or [20, 40]
        hunit = (
            int(self.height[:-2])
            - margin[0]
            - margin[1]
            - (len(heights) - 1) * gap[0]
        ) / 100
        wunit = (
            int(self.width[:-2])
            - margin[2]
            - margin[3]
            - (len(widths) - 1) * gap[1]
        ) / 100

        grids = []
        pos_top = margin[0]
        for hele in heights:
            pos_left = margin[2]
            for wele in widths:
                width = int(wele * wunit)
                height = int(hele * hunit)
                grid_opts = opts.GridOpts(
                    is_show=True,
                    pos_left=pos_left,
                    pos_top=pos_top,
                    width=width,
                    height=height,
                    is_contain_label=False,
                )
                pos_left += width + gap[1]
                grids.append(grid_opts)
            pos_top += height + gap[0]

        self.set_global_opts(grid_opts=grids)
        return self

    def set_defualt_opts(self) -> Self:
        """Set with the default global options."""
        self.set_datazoom(xaxis_index="all")
        self.set_tooltip()
        self.set_legend()
        self.set_title()
        self.set_color_map()
        return self
