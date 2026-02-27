"""回测报告页面"""

import reflex as rx
from pixiu.state import State


def page() -> rx.Component:
    """回测报告页面"""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.heading("📋 回测报告", size="lg"),
                rx.spacer(),
                rx.button("返回", on_click=rx.redirect("/")),
                width="100%",
                margin_bottom="1rem",
            ),
            
            rx.cond(
                len(State.backtest_result.get("results", [])) > 0,
                rx.box(
                    rx.foreach(
                        State.backtest_result.get("results", []),
                        lambda result: _result_item(result),
                    ),
                ),
                rx.box(
                    rx.text("暂无回测结果，请先选择股票和策略进行分析。"),
                    padding="2rem",
                    text_align="center",
                    color="gray.500",
                ),
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


def _result_item(result: dict) -> rx.Component:
    """单个策略结果"""
    total_return = result.get("total_return", 0)
    color = "green.500" if total_return > 0 else "red.500"
    
    return rx.box(
        rx.heading(result.get("strategy", ""), size="md", margin_bottom="1rem"),
        
        rx.grid(
            _metric_card("总收益率", f"{total_return:.2%}", color),
            _metric_card("年化收益", f"{result.get('annualized_return', 0):.2%}"),
            _metric_card("最大回撤", f"{result.get('max_drawdown', 0):.2%}", "red.500"),
            _metric_card("夏普比率", f"{result.get('sharpe_ratio', 0):.2f}"),
            _metric_card("胜率", f"{result.get('win_rate', 0):.2%}"),
            columns="5",
            spacing="1rem",
            margin_bottom="1rem",
        ),
        
        rx.divider(),
        
        padding="1.5rem",
        bg="white",
        border_radius="md",
        shadow="md",
        margin_bottom="1rem",
    )


def _metric_card(title: str, value: str, color: str = "black") -> rx.Component:
    """指标卡片"""
    return rx.box(
        rx.vstack(
            rx.text(title, font_size="sm", color="gray.500"),
            rx.text(value, font_size="xl", font_weight="bold", color=color),
            align_items="center",
        ),
        padding="1rem",
        bg="white",
        border_radius="md",
        shadow="sm",
        text_align="center",
    )
