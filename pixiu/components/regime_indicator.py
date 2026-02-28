"""择势状态指示器组件"""
import reflex as rx


def regime_indicator(regime: str, adx: float = 0, ma_slope: float = 0, volatility: float = 0) -> rx.Component:
    regime_color = "#10b981" if regime == "trend" else "#f59e0b"
    regime_text = "趋势" if regime == "trend" else "震荡"
    regime_icon = "📈" if regime == "trend" else "📊"
    
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text(f"{regime_icon} {regime_text}行情", font_size="lg", font_weight="bold"),
                rx.badge(regime_text, color_scheme="green" if regime == "trend" else "yellow"),
                justify="space_between", width="100%"
            ),
            rx.divider(),
            rx.hstack(
                rx.vstack(rx.text("ADX", font_size="sm", color="gray"), rx.text(f"{adx:.1f}", font_weight="bold")),
                rx.vstack(rx.text("MA斜率", font_size="sm", color="gray"), rx.text(f"{ma_slope:.4f}", font_weight="bold")),
                rx.vstack(rx.text("波动率", font_size="sm", color="gray"), rx.text(f"{volatility:.4f}", font_weight="bold")),
                justify="space_between", width="100%"
            ),
            spacing="2",
        ),
        padding="1rem", border_radius="lg", bg="#1a1a24", border=f"2px solid {regime_color}",
    )
