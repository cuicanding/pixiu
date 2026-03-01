"""量化实验 / 策略发现向导页面"""
import reflex as rx

from pixiu.state import State
from pixiu.components.echarts_kline import echarts_kline


def _step_header(title: str, description: str) -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.text(title, font_size="lg", font_weight="bold"),
            rx.spacer(),
        ),
        rx.text(description, font_size="sm", color="gray.400"),
        spacing="1",
        margin_bottom="1rem",
    )


def _turning_point_card(tp: dict) -> rx.Component:
    """横向转折点卡片组件"""
    return rx.box(
        # 第一行：序号 + 日期 + 方向变化 + 强度
        rx.hstack(
            # 序号
            rx.badge(
                f"T{tp.get('index', 1)}",
                color_scheme="cyan",
                variant="solid",
                font_size="0.875rem",
                font_weight="bold",
            ),
            # 日期
            rx.text(f"{tp.get('date', '')}", font_weight="bold", font_size="0.875rem"),
            # 方向变化（用颜色表示）
            rx.hstack(
                rx.text(
                    rx.cond(tp.get('from_direction') == "up", "上涨", "下跌"),
                    color=rx.cond(tp.get('from_direction') == "up", "#ef4444", "#22c55e"),
                    font_weight="bold", 
                    font_size="0.875rem"
                ),
                rx.text("→", color="#6b7280", font_size="0.875rem"),
                rx.text(
                    rx.cond(tp.get('to_direction') == "up", "上涨", "下跌"),
                    color=rx.cond(tp.get('to_direction') == "up", "#ef4444", "#22c55e"),
                    font_weight="bold", 
                    font_size="0.875rem"
                ),
                spacing="2",
                align="center",
            ),
            # 强度变化
            rx.hstack(
                rx.text("强度:", font_size="0.7rem", color="#6b7280"),
                rx.text(
                    tp.get('to_strength', 5),
                    font_weight="bold", 
                    font_size="0.875rem",
                    color="#10b981"  # 固定颜色
                ),
                rx.text("/10", font_size="0.75rem", color="#a0a0b0"),
                spacing="0",
            ),
            # 转折前天数
            rx.hstack(
                rx.text("转折前:", font_size="0.7rem", color="#6b7280"),
                rx.text(
                    tp.get('prev_duration', 0),
                    font_weight="bold", 
                    font_size="0.875rem"
                ),
                rx.text("天", font_size="0.75rem", color="#a0a0b0"),
                spacing="0",
            ),
            spacing="4",
            align="center",
            width="100%",
        ),
        # 第二行：转折理由
        rx.hstack(
            rx.text("信号:", font_size="0.7rem", color="#6b7280"),
            rx.text(
                tp.get('reason', ''),
                font_size="0.75rem",
                color="#a0a0b0",
            ),
            spacing="1",
            margin_top="0.25rem",
        ),
        padding="0.75rem",
        border_radius="0.5rem",
        bg="#1f1f2e",
        border=rx.cond(tp.get('to_direction') == "up", "1px solid #ef4444", "1px solid #22c55e"),
        width="100%",
        margin_bottom="0.5rem",
    )


def step1_select_market_stock() -> rx.Component:
    """Step 1: 选择时间、市场、股票（复用现有 State 字段和逻辑）"""
    return rx.box(
        _step_header("Step 1 · 选择时间、市场、股票", "先选定市场、标的和实验时间区间，这是整个实验的基础。"),
        rx.vstack(
            # 时间范围：复用 home 页的时间选择组件
            rx.box(
                rx.text("时间范围", font_size="sm", color="gray.400", margin_bottom="0.5rem"),
                rx.hstack(
                    rx.button(
                        "近1年",
                        size="2",
                        variant=rx.cond(
                            rx.cond(State.time_range_mode == "quick", State.quick_range == "12m", False),
                            "solid",
                            "outline",
                        ),
                        color_scheme="cyan",
                        on_click=State.set_quick_range("12m"),
                    ),
                    rx.button(
                        "近3年",
                        size="2",
                        variant=rx.cond(
                            rx.cond(State.time_range_mode == "quick", State.quick_range == "36m", False),
                            "solid",
                            "outline",
                        ),
                        color_scheme="cyan",
                        on_click=State.set_quick_range("36m"),
                    ),
                    spacing="2",
                    flex_wrap="wrap",
                ),
            ),

            # 市场选择
            rx.box(
                rx.text("市场", font_size="sm", color="gray.400", margin_bottom="0.5rem"),
                rx.hstack(
                    rx.button(
                        "A股",
                        on_click=State.set_market_a,
                        variant=rx.cond(State.current_market == "A股", "solid", "outline"),
                        color_scheme="cyan",
                        size="2",
                    ),
                    rx.button(
                        "港股",
                        on_click=State.set_market_hk,
                        variant=rx.cond(State.current_market == "港股", "solid", "outline"),
                        color_scheme="cyan",
                        size="2",
                    ),
                    rx.button(
                        "美股",
                        on_click=State.set_market_us,
                        variant=rx.cond(State.current_market == "美股", "solid", "outline"),
                        color_scheme="cyan",
                        size="2",
                    ),
                    spacing="2",
                ),
            ),

            # 股票搜索与选择（复用 State.search_stocks / select_stock）
            rx.box(
                rx.text("股票搜索", font_size="sm", color="gray.400", margin_bottom="0.5rem"),
                rx.hstack(
                    rx.input(
                        placeholder="输入代码或名称...",
                        value=State.search_keyword,
                        on_change=State.set_search_keyword,
                        size="2",
                    ),
                    rx.button(
                        "搜索",
                        on_click=State.search_stocks,
                        color_scheme="cyan",
                        size="2",
                        is_loading=State.is_loading,
                    ),
                    spacing="2",
                ),
                rx.cond(
                    State.error_message != "",
                    rx.text(State.error_message, color="red.400", font_size="sm", margin_top="0.5rem"),
                ),
                rx.vstack(
                    rx.foreach(
                        State.search_results,
                        lambda s: rx.button(
                            f"{s['name']} ({s['code']})",
                            variant=rx.cond(State.selected_stock == s["code"], "solid", "ghost"),
                            size="2",
                            width="100%",
                            justify_content="flex-start",
                            on_click=State.select_stock(s["code"]),
                        ),
                    ),
                    spacing="1",
                    margin_top="0.5rem",
                ),
            ),
        ),
        padding="1.5rem",
        border="1px solid gray.700",
        border_radius="lg",
        width="100%",
    )


def step2_regime_analysis() -> rx.Component:
    """Step 2: 择势分析 + 时间线 ECharts 可视化"""
    return rx.box(
        _step_header("Step 2 · 择势分析", "分析大盘与个股的趋势/震荡阶段，并用ECharts时间线展示关键区间与转折点。"),
        rx.vstack(
            rx.hstack(
                rx.button(
                    "开始择势分析",
                    on_click=State.analyze_regime,
                    color_scheme="cyan",
                    is_loading=State.is_loading,
                ),
                rx.spacer(),
            ),
            rx.cond(
                State.is_loading,
                rx.vstack(
                    rx.spinner(size="3"),
                    rx.text(State.loading_message, font_size="sm", color="gray.400"),
                    align_items="center",
                    padding="1.5rem",
                ),
            ),
            # ECharts 图表区域：大盘和个股分两个图展示，添加颜色图例说明
            rx.cond(
                (~State.is_loading) & (State.regime_chart_option != {}),
                rx.vstack(
                    # 颜色图例说明
                    rx.hstack(
                        rx.box(
                            rx.text("颜色说明:", font_size="sm", font_weight="bold", margin_right="1rem"),
                        ),
                        rx.hstack(
                            rx.box(width="20px", height="12px", bg="rgba(239, 68, 68, 0.35)", border_radius="sm"),
                            rx.text("上涨趋势", font_size="xs", color="gray.300"),
                            spacing="1",
                        ),
                        rx.hstack(
                            rx.box(width="20px", height="12px", bg="rgba(34, 197, 94, 0.35)", border_radius="sm"),
                            rx.text("下跌趋势", font_size="xs", color="gray.300"),
                            spacing="1",
                        ),
                        rx.hstack(
                            rx.box(width="20px", height="12px", bg="rgba(245, 158, 11, 0.35)", border_radius="sm"),
                            rx.text("震荡", font_size="xs", color="gray.300"),
                            spacing="1",
                        ),
                        spacing="3",
                        align_items="center",
                        padding="0.5rem 1rem",
                        bg="gray.800",
                        border_radius="md",
                    ),
                    
                    # 大盘图
                    rx.box(
                        echarts_kline(State.regime_chart_option.get("market", {}), height="360px"),
                        width="100%",
                    ),
                    
                    # 大盘转折点详情（与个股相同样式）
                    rx.cond(
                        State.market_turning_points,
                        rx.box(
                            rx.vstack(
                                rx.hstack(
                                    rx.text("📊 大盘转折点", font_size="1.25rem", font_weight="bold"),
                                    rx.spacer(),
                                    rx.badge(
                                        rx.text(State.market_turning_points.length(), font_size="xs"),
                                        color_scheme="cyan",
                                        variant="outline",
                                    ),
                                ),
                                rx.divider(),
                                # 横向排列转折点卡片
                                rx.foreach(
                                    State.market_turning_points,
                                    lambda tp: _turning_point_card(tp),
                                ),
                            ),
                            padding="1rem",
                            bg="#12121a",
                            border_radius="0.75rem",
                            border="1px solid #2a2a3a",
                            margin_bottom="1rem",
                            width="100%",
                        ),
                    ),
                    
                    # 个股图（带择势背景）
                    rx.box(
                        echarts_kline(State.regime_chart_option.get("stock", {}), height="400px"),
                        width="100%",
                    ),
                    
                    # 个股转折点详情卡片（横向展示）
                    rx.cond(
                        State.regime_timeline.get('turning_points'),
                        rx.box(
                            rx.vstack(
                                rx.hstack(
                                    rx.text("📅 个股转折点详情", font_size="1.25rem", font_weight="bold"),
                                    rx.spacer(),
                                    rx.badge(
                                        rx.text(State.regime_timeline.get('turning_points', []).length(), font_size="xs"),
                                        color_scheme="cyan",
                                        variant="outline",
                                    ),
                                ),
                                rx.divider(),
                                rx.foreach(
                                    State.regime_timeline.get('turning_points', []),
                                    lambda tp: _turning_point_card(tp),
                                ),
                            ),
                            padding="1rem",
                            bg="#12121a",
                            border_radius="0.75rem",
                            border="1px solid #2a2a3a",
                            margin_top="1rem",
                            width="100%",
                        ),
                    ),
                    
                    spacing="3",
                    margin_top="1rem",
                    width="100%",
                ),
                rx.cond(
                    ~State.is_loading,
                    rx.text(
                        "暂无时间线数据，请先选择股票并执行择势分析。",
                        font_size="sm",
                        color="gray.500",
                        margin_top="0.5rem",
                    ),
                ),
            ),

        ),
        padding="1.5rem",
        border="1px solid gray.700",
        border_radius="lg",
        width="100%",
    )


def step3_training_strategies() -> rx.Component:
    """Step 3: 训练集策略推荐与回测（占位，后续实现）"""
    return rx.box(
        _step_header("Step 3 · 训练集策略发现", "在训练集上针对不同择势组合推荐/回测策略，这一版先占位。"),
        rx.text("TODO: 在这里展示按择势组合拆分的训练集回测结果和策略对比。", color="gray.500"),
        padding="1.5rem",
        border="1px solid gray.700",
        border_radius="lg",
        width="100%",
    )


def step4_validation() -> rx.Component:
    """Step 4: 验证集回测与 AI 解释（占位，后续实现）"""
    return rx.box(
        _step_header("Step 4 · 验证集评估与解释", "在验证集上验证规则表现，并用 AI 生成回测报告。"),
        rx.text("TODO: 在这里展示验证集回测结果与 AI 报告。", color="gray.500"),
        padding="1.5rem",
        border="1px solid gray.700",
        border_radius="lg",
        width="100%",
    )


def experiment_steps() -> rx.Component:
    """实验向导主内容：四步流程"""
    return rx.vstack(
        rx.hstack(
            rx.badge("Step 1", color_scheme="cyan"),
            rx.text("选择时间/市场/股票", font_weight="bold"),
            rx.spacer(),
            align_items="center",
        ),
        step1_select_market_stock(),
        rx.divider(margin_y="4"),

        rx.hstack(
            rx.badge("Step 2", color_scheme="cyan"),
            rx.text("择势分析与时间线", font_weight="bold"),
            rx.spacer(),
            align_items="center",
        ),
        step2_regime_analysis(),
        rx.divider(margin_y="4"),

        rx.hstack(
            rx.badge("Step 3", color_scheme="cyan"),
            rx.text("训练集策略发现", font_weight="bold"),
            rx.spacer(),
            align_items="center",
        ),
        step3_training_strategies(),
        rx.divider(margin_y="4"),

        rx.hstack(
            rx.badge("Step 4", color_scheme="cyan"),
            rx.text("验证集评估与解释", font_weight="bold"),
            rx.spacer(),
            align_items="center",
        ),
        step4_validation(),
        spacing="4",
        width="100%",
    )


def experiment_page() -> rx.Component:
    """量化实验 / 策略发现主页面。当前重点：打通 Step 1 & Step 2。"""
    return rx.box(
        rx.html("<script src='https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js'></script>"),
        rx.vstack(
            rx.text(
                "量化实验 · 择势驱动策略发现",
                font_size="xl",
                font_weight="bold",
            ),
            rx.text(
                "按照\"选择时间与标的 → 择势分析 → 训练集策略发现 → 验证集评估\"的流程实验策略。",
                font_size="sm",
                color="gray.400",
            ),
            experiment_steps(),
            spacing="4",
            padding_y="2rem",
            width="100%",
        ),
        width="100%",
        padding_x="2rem",
    )
