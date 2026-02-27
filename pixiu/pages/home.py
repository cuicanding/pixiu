"""首页"""

import reflex as rx
from pixiu.state import State


def page() -> rx.Component:
    """首页组件"""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.heading("Pixiu 量化分析", size="6"),
                rx.spacer(),
                rx.badge("v0.1.0", color_scheme="blue"),
                width="100%",
                margin_bottom="1rem",
            ),
            
            rx.hstack(
                rx.text("市场:", font_weight="bold"),
                rx.button("A股", on_click=State.set_market_a),
                rx.button("港股", on_click=State.set_market_hk),
                rx.button("美股", on_click=State.set_market_us),
                spacing="2",
                width="100%",
                margin_bottom="1rem",
            ),
            
            rx.hstack(
                rx.input(
                    placeholder="输入股票代码或名称搜索...",
                    value=State.search_keyword,
                    on_change=State.set_search_keyword,
                    width="70%",
                ),
                rx.button("搜索", on_click=State.search_stocks, color_scheme="blue"),
                width="100%",
                margin_bottom="0.5rem",
            ),
            
            rx.cond(
                State.is_loading,
                rx.hstack(rx.spinner(), rx.text(State.loading_message)),
            ),
            
            rx.box(
                rx.foreach(
                    State.search_results,
                    render_search_result,
                ),
                max_height="200px",
                overflow_y="auto",
                width="100%",
            ),
            
            rx.cond(
                State.selected_stock != "",
                rx.box(
                    rx.heading(State.selected_stock_name, size="5"),
                    rx.divider(margin_y="1rem"),
                    
                    rx.text("策略选择", font_weight="bold", margin_bottom="0.5rem"),
                    
                    rx.grid(
                        rx.foreach(
                            State.available_strategies,
                            render_strategy,
                        ),
                        columns="2",
                        spacing="3",
                        margin_bottom="1rem",
                    ),
                    
                    rx.hstack(
                        rx.button("开始分析", on_click=State.run_backtest, color_scheme="blue", size="3"),
                        rx.progress(value=State.progress, width="200px"),
                    ),
                    padding="1.5rem",
                    border="1px solid gray.200",
                    border_radius="lg",
                    margin_top="1rem",
                ),
            ),
            
            rx.spacer(),
            
            rx.hstack(
                rx.text("Pixiu 2024", color="gray.400"),
                rx.spacer(),
                rx.link("设置", href="/settings", color_scheme="blue"),
                width="100%",
            ),
        ),
        width="100%",
        max_width="1000px",
        margin="0 auto",
        padding="2rem",
        min_height="100vh",
        bg="gray.50",
    )


def render_search_result(stock: dict) -> rx.Component:
    """渲染搜索结果项"""
    return rx.box(
        rx.hstack(
            rx.text(stock["code"], font_weight="bold", width="100px"),
            rx.text(stock["name"]),
        ),
        padding="0.5rem",
        cursor="pointer",
        on_click=lambda: State.select_stock(stock["code"]),
        border_bottom="1px solid gray.200",
    )


def render_strategy(s: dict) -> rx.Component:
    """渲染策略项"""
    return rx.box(
        rx.text(s["name"], font_weight="bold"),
        rx.text(s["description"], font_size="sm", color="gray.500"),
        padding="0.75rem",
        border="1px solid gray.200",
        border_radius="md",
        cursor="pointer",
        on_click=lambda: State.toggle_strategy(s["name"]),
    )
