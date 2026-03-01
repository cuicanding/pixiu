"""ECharts K线与时间线组件（使用 reflex-echarts 官方封装）"""
from typing import Dict, List, Any
import reflex as rx
from reflex_echarts import echarts


def echarts_kline(
    option: Dict[str, Any],
    height: str = "300px",
) -> rx.Component:
    """单标的 K 线 + 择势时间线组件。

    使用 reflex-echarts 官方组件，option 为 ECharts 配置字典。
    """
    return rx.box(
        echarts(
            option=option,
            style={
                "width": "100%", 
                "height": height,
                "minWidth": "400px",
                "minHeight": "200px",
            },
        ),
        width="100%",
        height=height,
        min_height=height,
    )



def echarts_dual_kline(
    index_data: List[Dict[str, Any]],
    index_timeline: List[Dict[str, Any]],
    stock_data: List[Dict[str, Any]],
    stock_timeline: List[Dict[str, Any]],
    height: str = "600px",
) -> rx.Component:
    """双图：大盘 + 个股时间线，先用两个单图堆叠。"""
    return rx.vstack(
        rx.text("大盘择势", font_weight="bold", font_size="sm"),
        echarts_kline(index_data, index_timeline, height="280px"),
        rx.divider(),
        rx.text("个股择势", font_weight="bold", font_size="sm"),
        echarts_kline(stock_data, stock_timeline, height="280px"),
        spacing="1rem",
        width="100%",
    )
