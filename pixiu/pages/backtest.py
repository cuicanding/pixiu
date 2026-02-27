"""回测报告页面"""

import reflex as rx
from pixiu.state import State


def page() -> rx.Component:
    """回测报告页面"""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.button("返回", on_click=rx.redirect("/")),
                rx.spacer(),
                rx.heading("回测报告", size="6"),
                width="100%",
            ),
            
            rx.divider(),
            
            rx.cond(
                State.backtest_results_empty,
                rx.box(
                    rx.text("暂无回测结果，请先执行回测分析", color="gray.400"),
                    padding="4rem",
                    text_align="center",
                ),
                
                rx.text("回测完成，请查看结果", color="green.400"),
            ),
            
            rx.spacer(),
            
            spacing="4",
            width="100%",
            max_width="1000px",
            margin="0 auto",
            padding="2rem",
        ),
        min_height="100vh",
        bg="gray.950",
    )
