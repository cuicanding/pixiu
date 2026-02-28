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
                rx.badge("v0.2.0", color_scheme="cyan"),
                width="100%",
            ),
            
            rx.box(
                rx.text("使用说明", font_weight="bold"),
                rx.text("1. 选择市场：点击下方按钮切换 A股/港股/美股", font_size="0.875rem", color="gray.400"),
                rx.text("2. 搜索股票：输入股票代码或名称", font_size="0.875rem", color="gray.400"),
                rx.text("3. 选择股票后点击策略卡片", font_size="0.875rem", color="gray.400"),
                rx.text("4. 开始分析：点击按钮执行回测", font_size="0.875rem", color="gray.400"),
                bg="gray.900",
                padding="1rem",
                border_radius="0.5rem",
                width="100%",
            ),
            
            rx.hstack(
                rx.text("市场:", font_weight="bold"),
                rx.button(
                    "A股",
                    variant=rx.cond(State.current_market == "A股", "solid", "outline"),
                    color_scheme=rx.cond(State.current_market == "A股", "cyan", "gray"),
                    on_click=State.set_market_a,
                ),
                rx.button(
                    "港股",
                    variant=rx.cond(State.current_market == "港股", "solid", "outline"),
                    color_scheme=rx.cond(State.current_market == "港股", "cyan", "gray"),
                    on_click=State.set_market_hk,
                ),
                rx.button(
                    "美股",
                    variant=rx.cond(State.current_market == "美股", "solid", "outline"),
                    color_scheme=rx.cond(State.current_market == "美股", "cyan", "gray"),
                    on_click=State.set_market_us,
                ),
                spacing="2",
            ),
            
            rx.hstack(
                rx.input(
                    placeholder="输入股票代码或名称搜索...",
                    value=State.search_keyword,
                    on_change=State.set_search_keyword,
                    width="70%",
                ),
                rx.button("搜索", on_click=State.search_stocks, color_scheme="cyan"),
                width="100%",
            ),
            
            rx.cond(
                State.is_loading,
                rx.hstack(rx.spinner(), rx.text(State.loading_message)),
            ),
            
            rx.cond(
                State.error_message != "",
                rx.text(State.error_message, color="red"),
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
                    rx.heading(f"已选: {State.selected_stock} {State.selected_stock_name}", size="5"),
                    rx.divider(),
                    
                    rx.text("策略选择", font_weight="bold"),
                    rx.grid(
                        rx.foreach(
                            State.available_strategies,
                            render_strategy,
                        ),
                        columns="2",
                        spacing="3",
                    ),
                    
                    rx.hstack(
                        rx.button("开始分析", on_click=State.run_backtest, color_scheme="cyan", size="3"),
                        rx.progress(value=State.progress, width="200px"),
                    ),
                    padding="1.5rem",
                    border="1px solid gray.700",
                    border_radius="lg",
                ),
            ),
            
            rx.spacer(),
            
            rx.hstack(
                rx.text("Pixiu 2024", color="gray.400"),
                rx.spacer(),
                rx.link("设置", href="/settings", color_scheme="cyan"),
                width="100%",
            ),
            
            spacing="4",
            width="100%",
        ),
        width="100%",
        max_width="800px",
        margin="0 auto",
        padding="2rem",
        min_height="100vh",
        bg="gray.950",
    )


def render_search_result(stock: dict) -> rx.Component:
    """渲染搜索结果项"""
    return rx.box(
        rx.hstack(
            rx.text(stock["code"], font_weight="bold", min_width="80px"),
            rx.text(stock["name"]),
        ),
        padding="0.5rem",
        cursor="pointer",
        on_click=State.select_stock(stock["code"]),
    )


def render_strategy(s: dict) -> rx.Component:
    """渲染策略项"""
    return rx.box(
        rx.text(s["name"], font_weight="bold"),
        rx.text(s.get("description", ""), font_size="sm", color="gray.500"),
        padding="0.75rem",
        border="1px solid gray.700",
        border_radius="md",
        cursor="pointer",
        on_click=State.toggle_strategy(s["name"]),
    )
