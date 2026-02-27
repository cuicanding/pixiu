"""首页"""

import reflex as rx


def page() -> rx.Component:
    """首页组件"""
    return rx.box(
        rx.vstack(
            rx.heading("📊 Pixiu 量化分析", size="lg"),
            rx.text("正在初始化..."),
            padding="2rem",
        ),
        min_height="100vh",
        bg="gray.50",
    )
