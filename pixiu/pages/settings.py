"""Settings page - clean form layout."""

import reflex as rx
from pixiu.state import State


def page() -> rx.Component:
    """Settings page with dark theme."""
    return rx.box(
        rx.vstack(
            # Header
            rx.hstack(
                rx.button(
                    "← 返回",
                    on_click=rx.redirect("/"),
                    variant="ghost",
                    color_scheme="cyan",
                ),
                rx.spacer(),
                rx.text(
                    "系统设置",
                    font_size="1.5rem",
                    font_weight="bold",
                    color="#ffffff",
                ),
                rx.spacer(),
                width="100%",
            ),
            
            rx.divider(border_color="#2a2a3a"),
            
            # API Configuration
            rx.box(
                rx.vstack(
                    rx.text("GLM API Key", color="#a0a0b0", font_size="0.875rem"),
                    rx.input(
                        placeholder="输入 GLM API Key",
                        value=State.glm_api_key,
                        on_change=State.set_glm_api_key,
                        type="password",
                        bg="#1a1a24",
                        border="1px solid #2a2a3a",
                        width="100%",
                    ),
                    rx.text(
                        "用于生成 AI 智能分析报告。可在 zhipuai.cn 获取",
                        font_size="0.75rem",
                        color="#6b7280",
                    ),
                    spacing="0.5rem",
                    width="100%",
                ),
                bg="#12121a",
                padding="1.5rem",
                border_radius="0.5rem",
                border="1px solid #2a2a3a",
                width="100%",
            ),
            
            # Backtest Parameters
            rx.box(
                rx.vstack(
                    rx.text("回测参数", color="#a0a0b0", font_size="0.875rem"),
                    rx.grid(
                        rx.vstack(
                            rx.text("初始资金", color="#a0a0b0", font_size="0.75rem"),
                            rx.input(
                                value=str(State.initial_capital),
                                on_change=State.set_initial_capital,
                                placeholder="100000",
                                bg="#1a1a24",
                                border="1px solid #2a2a3a",
                                width="100%",
                            ),
                            spacing="0.5rem",
                            width="100%",
                        ),
                        rx.vstack(
                            rx.text("手续费率", color="#a0a0b0", font_size="0.75rem"),
                            rx.input(
                                value=str(State.commission_rate),
                                on_change=State.set_commission_rate,
                                placeholder="0.0003",
                                bg="#1a1a24",
                                border="1px solid #2a2a3a",
                                width="100%",
                            ),
                            spacing="0.5rem",
                            width="100%",
                        ),
                        rx.vstack(
                            rx.text("仓位比例", color="#a0a0b0", font_size="0.75rem"),
                            rx.input(
                                value=str(State.position_size),
                                on_change=State.set_position_size,
                                placeholder="0.95",
                                bg="#1a1a24",
                                border="1px solid #2a2a3a",
                                width="100%",
                            ),
                            spacing="0.5rem",
                            width="100%",
                        ),
                        columns="3",
                        spacing="1rem",
                    ),
                    spacing="1rem",
                    width="100%",
                ),
                bg="#12121a",
                padding="1.5rem",
                border_radius="0.5rem",
                border="1px solid #2a2a3a",
                width="100%",
            ),
            
            # Save Button
            rx.button(
                "保存设置",
                on_click=State.save_settings,
                color_scheme="cyan",
                size="lg",
                margin_top="1rem",
            ),
            
            rx.spacer(),
            
            spacing="1.5rem",
            width="100%",
            max_width="600px",
            margin="0 auto",
            padding="2rem",
        ),
        min_height="100vh",
        bg="#0a0a0f",
    )
