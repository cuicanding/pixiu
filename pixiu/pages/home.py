"""首页"""

import reflex as rx
from pixiu.state import State
from pixiu.components.explain_button import explain_button, explain_modal


def step_indicator() -> rx.Component:
    """Step progress indicator."""
    steps = [
        ("选择市场", State.STEP_MARKET),
        ("搜索股票", State.STEP_SEARCH),
        ("择势分析", State.STEP_REGIME),
        ("选择策略", State.STEP_STRATEGY),
        ("查看结果", State.STEP_RESULT),
    ]
    return rx.hstack(
        *[
            rx.box(
                rx.hstack(
                    rx.badge(
                        str(i + 1),
                        color_scheme=rx.cond(State.current_step > i, "cyan", "gray"),
                        variant=rx.cond(State.current_step > i, "solid", "outline"),
                    ),
                    rx.text(
                        name,
                        font_size="0.75rem",
                        color=rx.cond(State.current_step > i, "cyan.400", "gray.500"),
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
                    size="3",
                    variant=rx.cond(State.current_market == "A股", "solid", "outline"),
                    color_scheme=rx.cond(State.current_market == "A股", "cyan", "gray"),
                    on_click=State.set_market_a,
                ),
                rx.button(
                    "港股",
                    size="3",
                    variant=rx.cond(State.current_market == "港股", "solid", "outline"),
                    color_scheme=rx.cond(State.current_market == "港股", "cyan", "gray"),
                    on_click=State.set_market_hk,
                ),
                rx.button(
                    "美股",
                    size="3",
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
                    size="3",
                ),
                rx.button(
                    "搜索",
                    on_click=State.search_stocks,
                    color_scheme="cyan",
                    size="3",
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
            
            rx.divider(margin_y="1rem"),
            
            rx.text("回测时间范围", font_size="lg", font_weight="bold"),
            
            rx.vstack(
                rx.text("快捷选项", font_size="sm", color="gray.400"),
                rx.hstack(
                    rx.button(
                        "近3个月",
                        size="2",
                        variant=rx.cond(State.quick_range == "3m", "solid", "outline"),
                        color_scheme="cyan",
                        on_click=State.set_quick_range("3m"),
                    ),
                    rx.button(
                        "近6个月",
                        size="2",
                        variant=rx.cond(State.quick_range == "6m", "solid", "outline"),
                        color_scheme="cyan",
                        on_click=State.set_quick_range("6m"),
                    ),
                    rx.button(
                        "近12个月",
                        size="2",
                        variant=rx.cond(State.quick_range == "12m", "solid", "outline"),
                        color_scheme="cyan",
                        on_click=State.set_quick_range("12m"),
                    ),
                    rx.button(
                        "近2年",
                        size="2",
                        variant=rx.cond(State.quick_range == "24m", "solid", "outline"),
                        color_scheme="cyan",
                        on_click=State.set_quick_range("24m"),
                    ),
                    spacing="2",
                    flex_wrap="wrap",
                ),
                spacing="2",
            ),
            
            rx.vstack(
                rx.text("年度选项", font_size="sm", color="gray.400"),
                rx.hstack(
                    rx.button(
                        "今年",
                        size="2",
                        variant=rx.cond(
                            rx.cond(State.time_range_mode == "year", State.year_range == "this_year", False),
                            "solid",
                            "outline",
                        ),
                        color_scheme="cyan",
                        on_click=State.set_year_range("this_year"),
                    ),
                    rx.button(
                        "去年",
                        size="2",
                        variant=rx.cond(
                            rx.cond(State.time_range_mode == "year", State.year_range == "last_year", False),
                            "solid",
                            "outline",
                        ),
                        color_scheme="cyan",
                        on_click=State.set_year_range("last_year"),
                    ),
                    rx.button(
                        "2023年",
                        size="2",
                        variant=rx.cond(
                            rx.cond(State.time_range_mode == "year", State.year_range == "2023", False),
                            "solid",
                            "outline",
                        ),
                        color_scheme="cyan",
                        on_click=State.set_year_range("2023"),
                    ),
                    rx.button(
                        "2024年",
                        size="2",
                        variant=rx.cond(
                            rx.cond(State.time_range_mode == "year", State.year_range == "2024", False),
                            "solid",
                            "outline",
                        ),
                        color_scheme="cyan",
                        on_click=State.set_year_range("2024"),
                    ),
                    spacing="2",
                    flex_wrap="wrap",
                ),
                spacing="2",
            ),
            
            rx.vstack(
                rx.text("自定义日期", font_size="sm", color="gray.400"),
                rx.hstack(
                    rx.box(
                        rx.text("开始日期", font_size="xs", color="gray.500"),
                        rx.input(
                            type="date",
                            value=State.custom_start_date,
                            on_change=State.set_custom_start,
                            size="2",
                        ),
                    ),
                    rx.box(
                        rx.text("结束日期", font_size="xs", color="gray.500"),
                        rx.input(
                            type="date",
                            value=State.custom_end_date,
                            on_change=State.set_custom_end,
                            size="2",
                        ),
                    ),
                    spacing="4",
                ),
                spacing="2",
            ),
            
            rx.box(
                rx.hstack(
                    rx.text("当前时间范围:", color="gray.400", font_size="sm"),
                    rx.text(State.backtest_start_date, color="cyan.400", font_size="sm"),
                    rx.text("至", color="gray.400", font_size="sm"),
                    rx.text(State.backtest_end_date, color="cyan.400", font_size="sm"),
                ),
                padding="0.75rem",
                bg="gray.800",
                border_radius="md",
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
                    rx.spinner(size="3"),
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
                            rx.hstack(
                                rx.text(f"ADX: ", font_size="sm"),
                                rx.text(State.regime_analysis.get("adx", 0), font_size="sm"),
                                explain_button("adx", State.regime_analysis.get("adx", "")),
                                spacing="1",
                                align_items="center",
                            ),
                            rx.text("|", color="gray.600"),
                            rx.hstack(
                                rx.text(f"MA斜率: ", font_size="sm"),
                                rx.text(State.regime_analysis.get("ma_slope", 0), font_size="sm"),
                                explain_button("ma_slope", State.regime_analysis.get("ma_slope", "")),
                                spacing="1",
                                align_items="center",
                            ),
                            rx.text("|", color="gray.600"),
                            rx.hstack(
                                rx.text(f"波动率: ", font_size="sm"),
                                rx.text(State.regime_analysis.get("volatility", 0), font_size="sm"),
                                explain_button("volatility", State.regime_analysis.get("volatility", "")),
                                spacing="1",
                                align_items="center",
                            ),
                            spacing="2",
                            align_items="center",
                        ),
                        padding="0.75rem",
                        bg="gray.900",
                        border_radius="md",
                    ),
                    
                    rx.cond(
                        State.using_mock_data,
                        rx.badge("使用模拟数据", color_scheme="yellow", variant="soft"),
                    ),
                    
                    # 大盘图表
                    rx.cond(
                        State.market_chart != "",
                        rx.box(
                            rx.text(
                                "大盘指数走势图 (上证指数)",
                                font_size="sm",
                                color="cyan.400",
                                margin_bottom="0.5rem",
                            ),
                            rx.image(
                                src=f"data:image/png;base64,{State.market_chart}",
                                width="100%",
                                border_radius="md",
                            ),
                            rx.hstack(
                                rx.text("大盘ADX: ", font_size="xs", color="gray.500"),
                                rx.text(State.market_index_data.get("adx", 0), font_size="xs", color="purple.400"),
                                rx.text(" | 大盘状态: ", font_size="xs", color="gray.500"),
                                rx.text(
                                    rx.cond(State.market_regime == "trend", "趋势行情", "震荡行情"),
                                    font_size="xs",
                                    color=rx.cond(State.market_regime == "trend", "green.400", "yellow.400"),
                                ),
                                spacing="1",
                                margin_top="0.5rem",
                            ),
                            width="100%",
                            margin_top="1rem",
                        ),
                    ),
                    
                    # 个股图表
                    rx.cond(
                        State.regime_chart != "",
                        rx.box(
                            rx.text(
                                f"{State.selected_stock_name} ({State.selected_stock}) 个股走势图",
                                font_size="sm",
                                color="cyan.400",
                                margin_bottom="0.5rem",
                            ),
                            rx.image(
                                src=f"data:image/png;base64,{State.regime_chart}",
                                width="100%",
                                border_radius="md",
                            ),
                            rx.hstack(
                                rx.text("个股ADX: ", font_size="xs", color="gray.500"),
                                rx.text(State.regime_analysis.get("adx", 0), font_size="xs", color="purple.400"),
                                rx.text(" | 个股状态: ", font_size="xs", color="gray.500"),
                                rx.text(
                                    rx.cond(State.stock_regime == "trend", "趋势行情", "震荡行情"),
                                    font_size="xs",
                                    color=rx.cond(State.stock_regime == "trend", "green.400", "yellow.400"),
                                ),
                                spacing="1",
                                margin_top="0.5rem",
                            ),
                            width="100%",
                            margin_top="1rem",
                        ),
                    ),
                    
                    rx.cond(
                        State.recommended_strategies.length() > 0,
                        rx.box(
                            rx.hstack(
                                rx.icon("lightbulb", color="yellow.400"),
                                rx.text("推荐策略: ", font_weight="bold", color="yellow.400"),
                                rx.foreach(
                                    State.recommended_strategies,
                                    lambda s: rx.badge(s, color_scheme="yellow", variant="soft")
                                ),
                            ),
                            padding="1rem",
                            bg="gray.800",
                            border_radius="md",
                            margin_top="1rem",
                        ),
                    ),
                    
                    rx.button(
                        "🤖 AI解释择势",
                        on_click=State.explain_regime_timeline,
                        color_scheme="purple",
                        size="2",
                        margin_top="0.5rem",
                        is_loading=State.ai_explaining,
                    ),
                    
                    rx.hstack(
                        rx.spacer(),
                        rx.button(
                            "下一步：选择策略",
                            on_click=State.go_to_strategy_step,
                            color_scheme="cyan",
                            size="3",
                        ),
                        margin_top="1rem",
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
                rx.text(name, font_weight="bold"),
                rx.spacer(),
                rx.cond(
                    State.recommended_strategies.contains(name),
                    rx.badge("推荐", color_scheme="green", size="1"),
                ),
                rx.badge(
                    rx.cond(
                        State.selected_strategies.contains(name),
                        "已选",
                        ""
                    ),
                    color_scheme=rx.cond(
                        State.selected_strategies.contains(name),
                        "cyan",
                        "gray"
                    ),
                    size="1",
                ),
                width="100%",
                align_items="center",
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
                    rx.hstack(
                        rx.icon("brain", color="cyan.400"),
                        rx.text("智能推荐", font_size="sm", font_weight="bold", color="cyan.400"),
                        explain_button("regime_recommendation", ""),
                        spacing="2",
                        align_items="center",
                    ),
                    
                    rx.box(
                        rx.text(State.regime_summary, font_size="sm", color="white", margin_bottom="0.5rem"),
                        rx.text(State.regime_explanation, font_size="xs", color="gray.400", white_space="pre-wrap"),
                        margin_top="0.5rem",
                        padding="0.75rem",
                        bg="gray.900",
                        border_radius="md",
                    ),
                    
                    rx.hstack(
                        rx.text("推荐策略:", font_size="xs", color="gray.500"),
                        rx.foreach(
                            State.recommended_strategies,
                            lambda s: rx.badge(s, color_scheme="cyan", variant="soft", size="1")
                        ),
                        spacing="2",
                        margin_top="0.5rem",
                    ),
                    
                    padding="1rem",
                    bg="gray.800",
                    border_radius="md",
                    border="1px solid cyan.700",
                    margin_bottom="1rem",
                ),
            ),
            
            rx.text("点击策略卡片选择/取消:", font_size="sm", color="gray.400"),
            
            rx.grid(
                rx.foreach(State.available_strategies, render_strategy),
                columns="2",
                spacing="3",
                width="100%",
            ),
            
            rx.hstack(
                rx.text(f"已选 ", color="gray.400"),
                rx.text(State.selected_strategies.length(), color="cyan.400", font_weight="bold"),
                rx.text(" 个策略", color="gray.400"),
            ),
            
            rx.button(
                "开始回测",
                on_click=State.run_backtest,
                color_scheme="cyan",
                size="3",
                width="100%",
                is_loading=State.is_loading,
                margin_top="1rem",
            ),
            
            rx.cond(
                State.is_loading,
                rx.vstack(
                    rx.progress(value=State.progress, width="100%"),
                    rx.text(State.loading_message, color="gray.400"),
                ),
            ),
            
            spacing="3",
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
                    color_scheme="gray",
                ),
                rx.button(
                    "🤖 AI评估",
                    on_click=State.evaluate_current_backtest(result["strategy"]),
                    color_scheme="purple",
                    size="1",
                    is_loading=State.ai_explaining,
                ),
            ),
            
            rx.grid(
                rx.box(
                    rx.hstack(
                        rx.text("年化收益", font_size="sm", color="gray.400"),
                        explain_button("annualized_return", result["annualized_return"]),
                        spacing="1",
                        align_items="center",
                    ),
                    rx.text(result["annualized_return"], font_weight="bold"),
                ),
                rx.box(
                    rx.hstack(
                        rx.text("最大回撤", font_size="sm", color="gray.400"),
                        explain_button("max_drawdown", result["max_drawdown"]),
                        spacing="1",
                        align_items="center",
                    ),
                    rx.text(result["max_drawdown"], color="red.400"),
                ),
                rx.box(
                    rx.hstack(
                        rx.text("夏普比率", font_size="sm", color="gray.400"),
                        explain_button("sharpe_ratio", result["sharpe_ratio"]),
                        spacing="1",
                        align_items="center",
                    ),
                    rx.text(result["sharpe_ratio"]),
                ),
                rx.box(
                    rx.hstack(
                        rx.text("胜率", font_size="sm", color="gray.400"),
                        explain_button("win_rate", result["win_rate"]),
                        spacing="1",
                        align_items="center",
                    ),
                    rx.text(result["win_rate"]),
                ),
                columns="4",
                spacing="4",
            ),
            
            rx.cond(
                result["chart"] != "",
                rx.box(
                    rx.text(
                        f"{result['strategy']} 策略回测图表",
                        font_size="sm",
                        color="cyan.400",
                        margin_bottom="0.5rem",
                    ),
                    rx.image(
                        src=f"data:image/png;base64,{result['chart']}",
                        width="100%",
                        border_radius="md",
                    ),
                    rx.hstack(
                        rx.text("图表说明: ", font_size="xs", color="gray.500"),
                        rx.text("红/绿K线=股价涨跌, ", font_size="xs", color="gray.400"),
                        rx.text("红三角=买入, ", font_size="xs", color="red.400"),
                        rx.text("绿三角=卖出, ", font_size="xs", color="green.400"),
                        rx.text("蓝线=资金曲线, ", font_size="xs", color="blue.400"),
                        rx.text("红填充=回撤", font_size="xs", color="red.300"),
                        spacing="1",
                        margin_top="0.5rem",
                    ),
                    margin_top="1rem",
                ),
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
            
            explain_modal(),
            
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
