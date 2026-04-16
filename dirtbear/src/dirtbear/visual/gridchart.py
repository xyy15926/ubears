#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: gridchart.py
#   Author: xyy15926
#   Created: 2026-04-15 22:56:58
#   Updated: 2026-04-16 14:59:00
#   Description:
# ---------------------------------------------------------

# %%
import logging
from typing import Dict, List, Tuple

from pyecharts.charts.chart import Chart
from pyecharts import options as opts
from pyecharts.charts import Grid
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
    cur_grid_index: Int.
      Current grid index for the chart that will be `add`ed.
    cur_xaxis_index: Int.
      Current xaxis index of the chart that will be `add`ed.
    cur_yaxis_index: Int.
      Current yaxis index of the chart that will be `add`ed.
    """
    def __init__(
        self,
        init_opts: opts.InitOpts | Dict = None,
        render_opts: opts.RenderOpts | Dict = None,
    ):
        """Init GridChart."""
        super().__init__(init_opts or {}, render_opts or {})
        self.cur_xaxis_index = 0
        self.cur_yaxis_index = 0
        self.cur_grid_index = 0
        self.charts = []

    def add_chart(
        self,
        chart: Chart,
        grid_opts: opts.GridOpts | Dict,
        xaxis_count: int = 1,
        yaxis_count: int = 1,
    ):
        """Add chart.

        Params:
        ----------------------------
        chart: Chart.
        grid_opts: Chart's grid options.
        xaxis_count: The number of xaxis in the chart.
        yaxis_count: The number of yaxis in the chart.

        Return:
        ----------------------------
        self
        """
        self.add(
            chart, grid_opts,
            grid_index=self.cur_grid_index,
            is_control_axis_index=True,
        )
        xaxs = list(range(self.cur_xaxis_index, self.cur_xaxis_index + xaxis_count))
        yaxs = list(range(self.cur_yaxis_index, self.cur_yaxis_index + yaxis_count))
        self.charts.append((chart, xaxs, yaxs))
        self.cur_grid_index += 1
        self.cur_xaxis_index += xaxis_count
        self.cur_yaxis_index += yaxis_count
        return self

    def set_global_opts(self, **kwargs):
        """Set global options.

        Params:
        ----------------------------
        datazoom_opts: DataZoomOpts, equivalent dict or list of those.
        tooltip_opts: TooltipOpts or equivalent dict.
        axispointer_opts: AxisPointerOpts or equivalent dict.
        legend_opts: LegendOpts, equivalent dict or list of those.
        title_opts: TitleOpts, equivalent dict or list of those.

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
        }
        for key, opt in kwargs.items():
            # set_trace()
            opt_key, opt_wrapper = mapper.get(key, (None, None))
            if opt_key is None:
                continue
            if isinstance(opt, dict):
                opt = opt_wrapper(**opt)
            elif (isinstance(opt, list)
                  and len(opt) > 0
                  and isinstance(opt[0], dict)):
                opt = [opt_wrapper(**ele) for ele in opt]

            # `option["title"]` accepts only dict, while other options could handle
            # both dict and XXXOpts, legend for example.
            if opt_key == "title":
                opt = opt.opts[0]

            # Replace the options value entirely for both scalar and list-value options.
            if isinstance(self.options[opt_key], list):
                if isinstance(opt, list):
                    self.options[opt_key] = opt
                else:
                    self.options[opt_key] = [opt, ]
            else:
                self.options[opt_key] = opt

        return self

    def set_datazoom(
        self,
        grid_x: List | str = "all",
        grid_y: List | str = None,
        *,
        datazoom_opts: Dict = None,
    ):
        """Set the datazoom.

        Params:
        -------------------------
        grid_x: List of the grid_index of the chart that would zoom in or zoom
          along the xaxis together.
          "all": All the charts.
        grid_y: List of the grid_index of the chart that would zoom in or zoom
          along the yaxis together.
          "all": All the charts.
        datazoom_opts: Dict of arguments of opts.DataZoomOpts.

        Return:
        -------------------------
        Self
        """
        conf = dict(
            is_show=True,
            type_="inside",
            is_disabled=False,
            is_realtime=True,
            is_zoom_lock=False,

            # Prefered percentage range settings.
            range_start=100,
            range_end=100,
            # min_span=20,
            # max_span=40,
            # start_value=20,
            # end_value=100,
            # The `min_value_span` will determine the really `range_start` as the
            # `range_start = 100`.
            min_value_span=20,
            max_value_span=100,

            # Control xaxis.
            orient="horizontal",
            xaxis_index=[0, 1],
            # pos_top="90%",
            # pos_bottom="10%",
            # pos_left="90%",
            # pos_right="10%",
        )
        if datazoom_opts is not None:
            conf.update(datazoom_opts)

        xaxis_index = None
        yaxis_index = None
        if grid_x == "all":
            xaxis_index = list(range(self.cur_xaxis_index))
        elif isinstance(grid_x, (tuple, list)):
            for idx in grid_x:
                xaxis_index += self.charts[idx][-2]
        if grid_y == "all":
            yaxis_index = list(range(self.cur_yaxis_index))
        elif isinstance(grid_y, (tuple, list)):
            for idx in grid_y:
                yaxis_index += self.charts[idx][-1]

        datazoom_opts = []
        if xaxis_index is not None:
            for type_ in ["inside", "slider"]:
                conf["type_"] = type_
                conf["orient"] = "horizontal"
                conf["xaxis_index"] = xaxis_index
                datazoom_opts.append(opts.DataZoomOpts(**conf))
        if yaxis_index is not None:
            for type_ in ["inside", "slider"]:
                conf["type_"] = type_
                conf["orient"] = "vertical"
                conf["yaxis_index"] = yaxis_index
                datazoom_opts.append(opts.DataZoomOpts(**conf))

        self.set_global_opts(datazoom_opts = datazoom_opts)
        return self

    def set_tooltip(
        self,
        formatter: str = None,
        *,
        tooltip_opts: Dict = None,
        axispointer_opts: Dict = None,
    ):
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
        tooltip_conf = dict(
            is_show = True,
            trigger = "axis",                 # Instead of `item`
            trigger_on= "mousemove|click",
            axis_pointer_type = "cross",
            is_show_content = True,
            is_always_show_content = False,
            show_delay = 0,
            formatter = formatter,
            background_color = "rgba(245, 245, 245, 0.5)",
            border_width = 1,
            border_color = "#ccc",
            textstyle_opts = opts.TextStyleOpts(
                color = "#000",
                font_weight = "bolder",
                font_size = 10,
                line_height = 8,
            ),
        )
        if tooltip_opts is not None:
            tooltip_conf.update(tooltip_opts)
        tooltip_opts = opts.TooltipOpts(**tooltip_conf)

        # Only valid when the `axis_pointer_type = "cross"`.
        axispointer_conf = dict(
            is_show = True,
            link = [{"xAxisIndex": "all"}],
            label = opts.LabelOpts(
                is_show = True,
                # position = "left",
                color = "#FFFFFF",
                font_size = 10,
                font_style = "normal",
                font_weight = "bold",
                background_color = "rgba(0,0,0,0.5)",
                border_color = "FFFFFF",
                border_width = 1,
            ),
        )
        if axispointer_opts is not None:
            axispointer_conf.update(axispointer_opts)
        axispointer_opts = opts.AxisPointerOpts(**axispointer_conf)

        self.set_global_opts(
            tooltip_opts = tooltip_opts,
            axispointer_opts = axispointer_opts,
        )
        return self

    def set_legend(
        self,
        pos_top: int | str = 20,
        pos_left: int | str = "center",
        pos_bottom: int | str = None,
        pos_right: int | str = None,
        *,
        legend_opts: Dict = None,
    ):
        """Set the legend.

        Params:
        -------------------------
        pos_top: Position from the top.
        pos_left:
        pos_right:
        pos_bottom:
        legend_opts: Dict of arguments of opts.LegendOpts.
        """
        conf = dict(
            is_show = True,
            pos_top = pos_top,
            pos_left = pos_left,
            pos_bottom = pos_bottom,
            pos_right = pos_right,
            background_color = "rgba(0, 0, 0, 0.1)",
        )
        if legend_opts is not None:
            conf.update(legend_opts)
        legend_opts = opts.LegendOpts(**conf)

        self.set_global_opts(legend_opts = legend_opts)
        return self

    def set_title(
        self,
        title: str = "",
        pos_top: int | str = 0,
        pos_left: int | str = "center",
        font_size: int = 18,
        *,
        title_opts: Dict = None,
    ):
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
        conf = dict(
            is_show = True,
            title = title,
            pos_top = pos_top,
            pos_left = pos_left,
            title_textstyle_opts = opts.TextStyleOpts(
                color = "#111111",
                font_weight = "normal",
                font_size = font_size,
            )
        )
        if title_opts is not None:
            conf.update(title_opts)
        title_opts = opts.TitleOpts(**conf)

        self.set_global_opts(title_opts = title_opts)
        return self

    def set_defualt_opts(self):
        """Set with the default global options."""
        self.set_datazoom()
        self.set_tooltip()
        self.set_legend()
        self.set_title()
        return self
