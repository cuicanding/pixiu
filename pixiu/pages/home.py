"""首页"""

import reflex as rx
from pixiu.state import State


def page() -> rx.Component:
    """首页组件"""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.heading("📊 Pixiu 量化分析", size="lg"),
                rx.spacer(),
                rx.badge("v0.1.0", color_scheme="blue"),
                width="100%",
                margin_bottom="1rem",
            ),
            
            rx.hstack(
                rx.text("市场:", font_weight="bold"),
                rx.button_group(
                    rx.button(
                        "A股",
                        variant="solid" if State.current_market == "A股" else "outline",
                        on_click=lambda: State.set_market("A股"),
                    ),
                    rx.button(
                        "港股",
                        variant="solid" if State.current_market == "港股" else "outline",
                        on_click=lambda: State.set_market("港股"),
                    ),
                    rx.button(
                        "美股",
                        variant="solid" if State.current_market == "美股" else "outline",
                        on_click=lambda: State.set_market("美股"),
                    ),
                ),
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
                rx.button(
                    "搜索",
                    on_click=State.search_stocks,
                    color_scheme="blue",
                ),
                width="100%",
                margin_bottom="0.5rem",
            ),
            
            rx.box(
                rx.foreach(
                    State.search_results,
                    lambda stock: rx.box(
                        rx.hstack(
                            rx.text(stock["code"], font_weight="bold", width="100px"),
                            rx.text(stock["name"]),
                            rx.spacer(),
                        ),
                        padding="0.75rem",
                        cursor="pointer",
                        on_click=lambda: State.select_stock(stock["code"]),
                        _hover={"bg": "gray.100"},
                        border_bottom="1px solid",
                        border_color="gray.200",
                    ),
                ),
                max_height="200px",
                overflow_y="auto",
                width="100%",
                border_radius="md",
                border="1px solid",
                border_color="gray.200",
            ),
            
            rx.cond(
                State.is_loading,
                rx.hstack(
                    rx.spinner(),
                    rx.text(State.loading_message, color="gray.500"),
                    margin_y="1rem",
                ),
            ),
            
            rx.cond(
                State.selected_stock != "",
                rx.box(
                    rx.vstack(
                        rx.heading(f"📈 {State.selected_stock_name} ({State.selected_stock})", size="md"),
                        
                        rx.divider(margin_y="1rem"),
                        
                        rx.heading("策略选择", size="sm", margin_bottom="0.5rem"),
                        
                        rx.grid(
                            rx.foreach(
                                State.available_strategies,
                                lambda s: rx.box(
                                    rx.checkbox(
                                        s["name"],
                                        checked=State.selected_strategies.contains(s["name"]),
                                        on_change=lambda: State.toggle_strategy(s["name"]),
                                    ),
                                    rx.text(s["description"], font_size="xs", color="gray.500"),
                                    padding="0.75rem",
                                    border="1px solid",
                                    border_color="gray.200",
                                    border_radius="md",
                                    cursor="pointer",
                                    on_click=lambda: State.toggle_strategy(s["name"]),
                                    _hover={"border_color": "blue.300"},
                                ),
                            ),
                            columns="2",
                            spacing="0.75rem",
                            margin_bottom="1rem",
                        ),
                        
                        rx.hstack(
                            rx.button(
                                "▶ 开始分析",
                                on_click=State.run_backtest,
                                is_disabled=len(State.selected_strategies) == 0,
                                color_scheme="blue",
                                size="lg",
                            ),
                            rx.spacer(),
                            rx.progress(
                                value=State.progress,
                                width="200px",
                            ),
                        ),
                    ),
                    padding="1.5rem",
                    border_radius="lg",
                    border="1px solid",
                    border_color="gray.200",
                    margin_top="1rem",
                ),
            ),
            
            rx.spacer(),
            
            rx.hstack(
                rx.text("Pixiu © 2024", color="gray.400", font_size="sm"),
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
