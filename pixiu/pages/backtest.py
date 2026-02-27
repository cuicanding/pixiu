"""回测报告页面"""

import reflex as rx
from pixiu.state import State


def page() -> rx.Component:
    """回测报告页面"""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.heading("回测报告", size="6"),
                rx.spacer(),
                rx.button("返回", on_click=rx.redirect("/")),
                width="100%",
                margin_bottom="1rem",
            ),
            
            rx.box(
                rx.text("暂无回测结果，请先选择股票和策略进行分析。"),
                padding="2rem",
                text_align="center",
                color="gray.500",
            ),
            
            rx.spacer(),
            
            width="100%",
            max_width="1200px",
            margin="0 auto",
            padding="2rem",
        ),
        min_height="100vh",
        bg="gray.50",
    )
