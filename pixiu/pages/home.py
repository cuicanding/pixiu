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
                rx.text("2. 搜索股票：输入股票代码或名称，如 000001 或 腾讯", font_size="0.875rem", color="gray.400"),
                rx.text("3. 选择策略后点击开始回测", font_size="0.875rem", color="gray.400"),
                margin_bottom="1rem",
            ),
            
            rx.divider(),
            
            rx.hstack(
                rx.text("市场:", font_weight="bold"),
                rx.button(
                    "A股",
                    variant="solid" if State.current_market == "A股" else "outline",
                    color_scheme="cyan" if State.current_market == "A股" else "gray",
                    on_click=State.set_market_a,
                    border_radius="md",
                ),
                rx.button(
                    "港股",
                    variant="solid" if State.current_market == "港股" else "outline",
                    color_scheme="cyan" if State.current_market == "港股" else "gray",
                    on_click=State.set_market_hk,
                    border_radius="md",
                ),
                rx.button(
                    "美股",
                    variant="solid" if State.current_market == "美股" else "outline",
                    color_scheme="cyan" if State.current_market == "美股" else "gray",
                    on_click=State.set_market_us,
                    border_radius="md",
                ),
                rx.badge(State.current_market + " ✓", color_scheme="green", variant="solid"),
                spacing="2",
            ),
            
            rx.divider(),
            
            rx.hstack(
                rx.input(
                    placeholder="输入股票代码或名称搜索...",
                    value=State.search_keyword,
                    on_change=State.set_search_keyword,
                    width="300px",
                ),
                rx.button(
                    "搜索",
                    on_click=State.search_stocks,
                    is_loading=State.is_loading,
                    color_scheme="cyan",
                ),
                spacing="2",
            ),
            
            rx.cond(
                State.error_message != "",
                rx.callout(
                    State.error_message,
                    icon="alert",
                    color_scheme="red",
                ),
            ),
            
            rx.cond(
                State.search_results.length() > 0,
                rx.box(
                    rx.text("搜索结果:", font_weight="bold", margin_bottom="0rem"),
                    rx.foreach(
                        State.search_results,
                        render_search_result,
                    ),
                    max_height="200px",
                    overflow_y="auto",
                    width="100%",
                    padding="0.5rem",
                    border="1px solid gray.700",
                    border_radius="md",
                ),
            ),
            
            rx.divider(),
            
            rx.text("选择策略:", font_weight="bold"),
            
            rx.box(
                rx.foreach(
                    State.available_strategies,
                    render_strategy,
                ),
                display="grid",
                grid_template_columns="repeat(2, 1fr)",
                gap="0.5rem",
                width="100%",
            ),
            
            rx.divider(),
            
            rx.hstack(
                rx.text("初始资金:", font_weight="bold"),
                rx.input(
                    value=State.initial_capital,
                    on_change=State.set_initial_capital,
                    width="120px",
                ),
                rx.text("手续费率:", font_weight="bold"),
                rx.input(
                    value=State.commission_rate,
                    on_change=State.set_commission_rate,
                    width="80px",
                ),
                spacing="4",
            ),
            
            rx.button(
                "开始回测分析",
                on_click=State.run_backtest,
                is_loading=State.is_loading,
                is_disabled=(State.selected_stock == "") | (State.selected_strategies.length() == 0),
                color_scheme="cyan",
                size="lg",
                width="100%",
                margin_top="1rem",
            ),
            
            rx.cond(
                State.backtest_results.length() > 0,
                rx.box(
                    rx.hstack(
                        rx.heading("回测结果", size="5"),
                        rx.spacer(),
                        rx.link("查看详细报告", href="/backtest", color_scheme="cyan"),
                        width="100%",
                    ),
                    rx.foreach(
                        State.backtest_results,
                        render_backtest_result,
                    ),
                    margin_top="1rem",
                    padding="1rem",
                    border="1px solid gray.700",
                    border_radius="md",
                ),
            ),
            
            rx.hstack(
                rx.text("已选股票:", color="gray.400"),
                rx.cond(
                    State.selected_stock != "",
                    rx.badge(f"{State.selected_stock} {State.selected_stock_name}", color_scheme="green"),
                    rx.text("未选择", color="gray.500"),
                ),
                rx.spacer(),
                rx.text(f"已选策略: {State.selected_strategies.length()}个", color="gray.400"),
                rx.spacer(),
                rx.link("设置", href="/settings", color_scheme="cyan"),
                width="100%",
                margin_top="1rem",
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
        on_click=lambda: State.select_stock(stock["code"]),
        _hover={"bg": "gray.800"},
        border_radius="md",
    )


def render_strategy(s: dict) -> rx.Component:
    """渲染策略项"""
    return rx.box(
        rx.hstack(
            rx.checkbox(
                is_checked=State.selected_strategies.contains(s["name"]),
                on_change=lambda: State.toggle_strategy(s["name"]),
            ),
            rx.vstack(
                rx.text(s["name"], font_weight="bold"),
                rx.text(s.get("description", ""), font_size="sm", color="gray.500"),
                spacing="0",
                align_items="start",
            ),
        ),
        padding="0.75rem",
        border="1px solid gray.700",
        border_radius="md",
        cursor="pointer",
    )


def render_backtest_result(result: dict) -> rx.Component:
    """渲染回测结果项"""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text(result.get("strategy", ""), font_weight="bold"),
                rx.badge(
                    f"收益: {result.get('total_return', 00):.2%}",
                    color_scheme="green" if result.get("total_return", 0) > 0 else "red",
                ),
            ),
            rx.hstack(
                rx.text(f"年化收益: {result.get('annualized_return', 00):.2%}", font_size="sm"),
                rx.text(f"最大回撤: {result.get('max_drawdown', 00):.2%}", font_size="sm"),
                rx.text(f"夏普比率: {result.get('sharpe_ratio', 0):.2f}", font_size="sm"),
                rx.text(f"胜率: {result.get('win_rate', 0):.2%}", font_size="sm"),
            ),
        ),
        padding="0.5rem",
        border_radius="md",
    )
