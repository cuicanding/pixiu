"""设置页面"""

import reflex as rx
from pixiu.state import State


def page() -> rx.Component:
    """设置页面"""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.heading("系统设置", size="6"),
                rx.spacer(),
                rx.link("返回首页", href="/", color_scheme="cyan"),
                width="100%",
            ),
            
            rx.divider(),
            
            rx.box(
                rx.heading("API 配置", size="5"),
                rx.text("GLM API Key", color="gray.400"),
                rx.input(
                    placeholder="输入 GLM API Key",
                    value=State.glm_api_key,
                    on_change=State.set_glm_api_key,
                    type="password",
                    width="100%",
                ),
                rx.text("用于生成 AI 智能分析报告", color="gray.500", font_size="0.75rem"),
                margin_top="1rem",
            ),
            
            rx.box(
                rx.heading("回测参数", size="5"),
                rx.grid(
                    rx.box(
                        rx.text("初始资金", color="gray.400"),
                        rx.input(
                            value=State.initial_capital,
                            on_change=State.set_initial_capital,
                            width="100%",
                        ),
                    ),
                    rx.box(
                        rx.text("手续费率", color="gray.400"),
                        rx.input(
                            value=State.commission_rate,
                            on_change=State.set_commission_rate,
                            width="100%",
                        ),
                    ),
                    rx.box(
                        rx.text("仓位比例", color="gray.400"),
                        rx.input(
                            value=State.position_size,
                            on_change=State.set_position_size,
                            width="100%",
                        ),
                    ),
                    columns="3",
                    spacing="4",
                    margin_top="1rem",
                ),
                margin_top="1.5rem",
            ),
            
            rx.button("保存设置", on_click=State.save_settings, color_scheme="cyan", size="3"),
            
            spacing="4",
            width="100%",
            max_width="600px",
            margin="0 auto",
            padding="2rem",
        ),
        min_height="100vh",
        bg="gray.950",
    )
