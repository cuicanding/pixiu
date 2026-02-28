"""首页"""

import reflex as rx
from pixiu.state import State


def step_indicator() -> rx.Component:
    """Step progress indicator."""
    steps = [
        ("选择市场", State.STEP_MARKET),
        ("搜索股票", State.STEP_SEARCH),
        ("择势分析", State.STEP_REGIME),
        ("选择策略", State.STEP_STRATEGY),
        ("配置参数", State.STEP_CONFIG),
        ("查看结果", State.STEP_RESULT),
    ]
    return rx.hstack(
        *[
            rx.box(
                rx.hstack(
                    rx.badge(
                        str(i + 1),
                        color_scheme="cyan" if State.current_step > i else "gray",
                        variant="solid" if State.current_step > i else "outline",
                    ),
                    rx.text(
                        name,
                        font_size="0.75rem",
                        color="cyan.400" if State.current_step > i else "gray.500",
                    ),
                    spacing="2",
                ),
                padding_x="0.5rem",
            )
            for i, (name, _) in enumerate(steps)
        ],
        spacing="1",
        width="100%",
        overflow_x="auto",
        padding_y="0.5rem",
    )


def step_market_selection() -> rx.Component:
    """Step 1: Market selection."""
    return rx.box(
        rx.vstack(
            rx.text("选择市场", font_size="lg", font_weight="bold"),
            rx.text("选择要分析的股票市场", font_size="sm", color="gray.400"),
            rx.hstack(
                rx.button(
                    "A股",
                    size="lg",
                    variant=rx.cond(State.current_market == "A股", "solid", "outline"),
                    color_scheme=rx.cond(State.current_market == "A股", "cyan", "gray"),
                    on_click=State.set_market_a,
                ),
                rx.button(
                    "港股",
                    size="lg",
                    variant=rx.cond(State.current_market == "港股", "solid", "outline"),
                    color_scheme=rx.cond(State.current_market == "港股", "cyan", "gray"),
                    on_click=State.set_market_hk,
                ),
                rx.button(
                    "美股",
                    size="lg",
                    variant=rx.cond(State.current_market == "美股", "solid", "outline"),
                    color_scheme=rx.cond(State.current_market == "美股", "cyan", "gray"),
                    on_click=State.set_market_us,
                ),
                spacing="4",
            ),
            spacing="3",
        ),
        padding="1.5rem",
        border="1px solid gray.700",
        border_radius="lg",
        width="100%",
    )


def render_search_result(stock: dict) -> rx.Component:
    """Render a search result item."""
    is_selected = State.selected_stock == stock["code"]
    return rx.box(
        rx.hstack(
            rx.badge(stock["code"], color_scheme="gray"),
            rx.text(stock["name"], font_weight="medium"),
            rx.spacer(),
            rx.cond(
                is_selected,
                rx.badge("已选", color_scheme="green"),
                rx.text("点击选择", font_size="sm", color="gray.500"),
            ),
        ),
        padding="0.75rem",
        border=rx.cond(is_selected, "1px solid cyan.500", "1px solid transparent"),
        border_radius="md",
        bg=rx.cond(is_selected, "gray.800", "transparent"),
        cursor="pointer",
        on_click=State.select_stock(stock["code"]),
        _hover={"bg": "gray.800"},
    )


def step_stock_search() -> rx.Component:
    """Step 2: Search and select stock."""
    return rx.box(
        rx.vstack(
            rx.text("搜索股票", font_size="lg", font_weight="bold"),
            rx.text(f"当前市场: {State.current_market}", font_size="sm", color="gray.400"),
            
            rx.hstack(
                rx.input(
                    placeholder="输入股票代码或名称...",
                    value=State.search_keyword,
                    on_change=State.set_search_keyword,
                    width="100%",
                    size="lg",
                ),
                rx.button(
                    "搜索",
                    on_click=State.search_stocks,
                    color_scheme="cyan",
                    size="lg",
                    is_loading=State.is_loading,
                ),
                width="100%",
                spacing="2",
            ),
            
            rx.cond(
                State.is_loading,
                rx.hstack(rx.spinner(), rx.text(State.loading_message)),
            ),
            
            rx.cond(
                State.error_message != "",
                rx.text(State.error_message, color="red.400"),
            ),
            
            rx.box(
                rx.foreach(State.search_results, render_search_result),
                max_height="250px",
                overflow_y="auto",
                width="100%",
            ),
            
            rx.cond(
                State.selected_stock != "",
                rx.box(
                    rx.hstack(
                        rx.text("已选择:", color="gray.400"),
                        rx.badge(State.selected_stock, color_scheme="cyan"),
                        rx.text(State.selected_stock_name, font_weight="bold"),
                    ),
                    padding="0.75rem",
                    bg="gray.800",
                    border_radius="md",
                ),
            ),
            
            spacing="3",
        ),
        padding="1.5rem",
        border="1px solid gray.700",
        border_radius="lg",
        width="100%",
    )


def step_regime_analysis() -> rx.Component:
    """Step 3: Market and stock regime analysis."""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text("择势分析", font_size="lg", font_weight="bold"),
                rx.spacer(),
                rx.button(
                    "开始分析",
                    on_click=State.analyze_regime,
                    color_scheme="cyan",
                    is_loading=State.is_loading,
                ),
            ),
            
            rx.text("分析市场状态和个股走势，智能推荐策略", font_size="sm", color="gray.400"),
            
            rx.cond(
                State.is_loading,
                rx.vstack(
                    rx.spinner(size="lg"),
                    rx.text(State.loading_message),
                    align_items="center",
                    padding="2rem",
                ),
            ),
            
            rx.cond(
                State.stock_regime != "unknown",
                rx.vstack(
                    rx.grid(
                        rx.box(
                            rx.text("大盘状态", font_size="sm", color="gray.400"),
                            rx.hstack(
                                rx.badge(
                                    rx.cond(State.market_regime == "trend", "趋势", "震荡"),
                                    color_scheme=rx.cond(State.market_regime == "trend", "green", "yellow"),
                                ),
                                rx.text(
                                    rx.cond(State.market_regime == "trend", "适合跟踪策略", "适合均值回归"),
                                    font_size="sm",
                                ),
                            ),
                            padding="1rem",
                            bg="gray.800",
                            border_radius="md",
                        ),
                        rx.box(
                            rx.text("个股状态", font_size="sm", color="gray.400"),
                            rx.hstack(
                                rx.badge(
                                    rx.cond(State.stock_regime == "trend", "趋势", "震荡"),
                                    color_scheme=rx.cond(State.stock_regime == "trend", "green", "yellow"),
                                ),
                                rx.text(
                                    rx.cond(State.stock_regime == "trend", "趋势明显", "横盘整理"),
                                    font_size="sm",
                                ),
                            ),
                            padding="1rem",
                            bg="gray.800",
                            border_radius="md",
                        ),
                        columns="2",
                        spacing="4",
                        width="100%",
                    ),
                    
                    rx.box(
                        rx.text("分析指标", font_size="sm", color="gray.400", margin_bottom="0.5rem"),
                        rx.hstack(
                            rx.text(f"ADX: ", font_size="sm"),
                            rx.text(State.regime_analysis.get("adx", 0), font_size="sm"),
                            rx.text("|", color="gray.600"),
                            rx.text(f"MA斜率: ", font_size="sm"),
                            rx.text(State.regime_analysis.get("ma_slope", 0), font_size="sm"),
                            rx.text("|", color="gray.600"),
                            rx.text(f"波动率: ", font_size="sm"),
                            rx.text(State.regime_analysis.get("volatility", 0), font_size="sm"),
                        ),
                        padding="0.75rem",
                        bg="gray.900",
                        border_radius="md",
                    ),
                    
                    rx.cond(
                        State.using_mock_data,
                        rx.badge("使用模拟数据", color_scheme="yellow", variant="subtle"),
                    ),
                    
                    spacing="4",
                ),
            ),
            
            spacing="3",
        ),
        padding="1.5rem",
        border="1px solid gray.700",
        border_radius="lg",
        width="100%",
    )


def render_strategy(s: dict) -> rx.Component:
    """Render a strategy card with selection state."""
    name = s["name"]
    
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.checkbox(
                    is_checked=State.selected_strategies.contains(name),
                ),
                rx.text(name, font_weight="bold"),
                rx.cond(
                    State.recommended_strategies.contains(name),
                    rx.badge("推荐", color_scheme="green", size="sm"),
                ),
                width="100%",
                justify="between",
            ),
            rx.text(
                s.get("description", ""),
                font_size="sm",
                color="gray.500",
            ),
            align_items="start",
            spacing="1",
        ),
        padding="0.75rem",
        border=rx.cond(
            State.selected_strategies.contains(name),
            "2px solid cyan.500",
            rx.cond(
                State.recommended_strategies.contains(name),
                "1px solid green.700",
                "1px solid gray.700"
            )
        ),
        border_radius="md",
        cursor="pointer",
        bg=rx.cond(State.selected_strategies.contains(name), "gray.800", "transparent"),
        on_click=State.toggle_strategy(name),
        _hover={"bg": "gray.800"},
    )


def step_strategy_selection() -> rx.Component:
    """Step 4: Strategy selection with recommendations."""
    return rx.box(
        rx.vstack(
            rx.text("选择策略", font_size="lg", font_weight="bold"),
            
            rx.cond(
                State.recommended_strategies.length() > 0,
                rx.box(
                    rx.text("基于择势分析推荐:", font_size="sm", color="green.400", margin_bottom="0.5rem"),
                    rx.text(
                        rx.cond(
                            State.market_regime == "trend",
                            rx.cond(
                                State.stock_regime == "trend",
                                "大盘+个股均为趋势行情，推荐趋势跟踪策略",
                                "大盘趋势但个股震荡，推荐网格或RSI策略"
                            ),
                            rx.cond(
                                State.stock_regime == "trend",
                                "大盘震荡但个股有趋势，可尝试趋势策略",
                                "大盘+个股均震荡，推荐均值回归策略"
                            )
                        ),
                        font_size="sm",
                        color="gray.400",
                    ),
                    padding="0.75rem",
                    bg="green.900",
                    border_radius="md",
                    margin_bottom="1rem",
                ),
            ),
            
            rx.grid(
                rx.foreach(State.available_strategies, render_strategy),
                columns="2",
                spacing="3",
                width="100%",
            ),
            
            rx.hstack(
                rx.text(f"已选 ", color="gray.400"),
                rx.text(State.selected_strategies.length(), color="gray.400"),
                rx.text(" 个策略", color="gray.400"),
            ),
            
            spacing="3",
        ),
        padding="1.5rem",
        border="1px solid gray.700",
        border_radius="lg",
        width="100%",
    )


def step_config() -> rx.Component:
    """Step 5: Backtest configuration."""
    return rx.box(
        rx.vstack(
            rx.text("配置参数", font_size="lg", font_weight="bold"),
            
            rx.vstack(
                rx.hstack(
                    rx.text("初始资金:", width="100px"),
                    rx.input(
                        value=State.initial_capital,
                        on_change=State.set_initial_capital,
                        type="number",
                        width="200px",
                    ),
                ),
                rx.hstack(
                    rx.text("手续费率:", width="100px"),
                    rx.input(
                        value=State.commission_rate,
                        on_change=State.set_commission_rate,
                        type="number",
                        width="200px",
                    ),
                ),
                rx.hstack(
                    rx.text("仓位比例:", width="100px"),
                    rx.input(
                        value=State.position_size,
                        on_change=State.set_position_size,
                        type="number",
                        width="200px",
                    ),
                ),
                spacing="3",
            ),
            
            rx.button(
                "开始回测",
                on_click=State.run_backtest,
                color_scheme="cyan",
                size="lg",
                width="100%",
                is_loading=State.is_loading,
            ),
            
            rx.cond(
                State.is_loading,
                rx.vstack(
                    rx.progress(value=State.progress, width="100%"),
                    rx.text(State.loading_message, color="gray.400"),
                ),
            ),
            
            spacing="4",
        ),
        padding="1.5rem",
        border="1px solid gray.700",
        border_radius="lg",
        width="100%",
    )


def render_backtest_result(result: dict) -> rx.Component:
    """Render a single backtest result card."""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text(result["strategy"], font_weight="bold", font_size="lg"),
                rx.spacer(),
                rx.badge(
                    result["total_return"],
                    color_scheme=rx.cond(result["total_return"] > 0, "green", "red"),
                ),
            ),
            
            rx.grid(
                rx.box(
                    rx.text("年化收益", font_size="sm", color="gray.400"),
                    rx.text(result["annualized_return"], font_weight="bold"),
                ),
                rx.box(
                    rx.text("最大回撤", font_size="sm", color="gray.400"),
                    rx.text(result["max_drawdown"], color="red.400"),
                ),
                rx.box(
                    rx.text("夏普比率", font_size="sm", color="gray.400"),
                    rx.text(result["sharpe_ratio"]),
                ),
                rx.box(
                    rx.text("胜率", font_size="sm", color="gray.400"),
                    rx.text(result["win_rate"]),
                ),
                columns="4",
                spacing="4",
            ),
            
            spacing="3",
        ),
        padding="1rem",
        bg="gray.800",
        border_radius="md",
        width="100%",
    )


def step_results() -> rx.Component:
    """Step 6: Display backtest results."""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text("回测结果", font_size="lg", font_weight="bold"),
                rx.spacer(),
                rx.button(
                    "重新开始",
                    on_click=State.reset_flow,
                    variant="outline",
                    color_scheme="gray",
                ),
            ),
            
            rx.foreach(State.backtest_results, render_backtest_result),
            
            rx.cond(
                State.backtest_results.length() == 0,
                rx.text("暂无回测结果", color="gray.400"),
            ),
            
            spacing="4",
        ),
        padding="1.5rem",
        border="1px solid gray.700",
        border_radius="lg",
        width="100%",
    )


def page() -> rx.Component:
    """Main page with wizard flow."""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.heading("Pixiu 量化分析", size="6"),
                rx.spacer(),
                rx.badge("v0.3.0", color_scheme="cyan"),
                width="100%",
            ),
            
            step_indicator(),
            rx.divider(),
            
            rx.cond(
                State.current_step == State.STEP_MARKET,
                step_market_selection(),
            ),
            rx.cond(
                State.current_step == State.STEP_SEARCH,
                step_stock_search(),
            ),
            rx.cond(
                State.current_step == State.STEP_REGIME,
                step_regime_analysis(),
            ),
            rx.cond(
                State.current_step == State.STEP_STRATEGY,
                step_strategy_selection(),
            ),
            rx.cond(
                State.current_step == State.STEP_CONFIG,
                step_config(),
            ),
            rx.cond(
                State.current_step == State.STEP_RESULT,
                step_results(),
            ),
            
            rx.hstack(
                rx.button(
                    "上一步",
                    on_click=State.prev_step,
                    variant="outline",
                    is_disabled=State.current_step == 1,
                ),
                rx.spacer(),
                rx.button(
                    "下一步",
                    on_click=State.next_step,
                    color_scheme="cyan",
                    is_disabled=State.current_step >= State.max_step,
                ),
                width="100%",
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
