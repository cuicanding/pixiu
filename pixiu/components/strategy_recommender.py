"""策略推荐组件"""
import reflex as rx

STRATEGY_REGIME_MAP = {
    "trend": ["趋势强度策略", "均线交叉策略", "最优执行策略"],
    "range": ["网格交易策略", "RSI策略", "波动率套利策略"],
    "any": ["随机过程策略", "卡尔曼滤波策略"]
}


def strategy_recommender(regime: str, available_strategies: list, selected_strategies: list, on_toggle) -> rx.Component:
    recommended = STRATEGY_REGIME_MAP.get(regime, [])
    return rx.vstack(
        rx.hstack(rx.text("推荐策略", font_weight="bold"), rx.badge(f"基于{regime}行情", color_scheme="blue")),
        rx.foreach(available_strategies, lambda s: _strategy_item(s, recommended, selected_strategies, on_toggle)),
        spacing="2", width="100%"
    )


def _strategy_item(strategy: dict, recommended: list, selected: list, on_toggle) -> rx.Component:
    is_recommended = strategy["name"] in recommended
    is_selected = strategy["name"] in selected
    return rx.box(
        rx.hstack(
            rx.checkbox(is_checked=is_selected, on_change=lambda: on_toggle(strategy["name"])),
            rx.vstack(rx.text(strategy["name"], font_weight="medium"), rx.text(strategy["description"], font_size="sm", color="gray"), spacing="1", align_items="start"),
            rx.cond(is_recommended, rx.badge("推荐", color_scheme="green", size="sm"), rx.box()),
            justify="space_between", width="100%"
        ),
        padding="0.5rem", border_radius="md", bg="#252532" if is_selected else "transparent",
        border=f"1px solid {'#10b981' if is_recommended else '#333'}",
    )
