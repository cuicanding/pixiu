"""Explanation button and modal components for AI-powered metric explanations."""
import reflex as rx
from pixiu.state import State


def explain_button(concept: str, value: str = "") -> rx.Component:
    """Create an explanation button for a metric.
    
    Args:
        concept: The concept to explain (e.g., 'sharpe_ratio', 'total_return')
        value: The current value of the metric
    """
    return rx.button(
        rx.icon(tag="circle_help", size=16),
        variant="ghost",
        size="1",
        color="#6b7280",
        on_click=lambda: State.explain_concept(concept, value),
        cursor="pointer",
        _hover={"color": "#00d9ff"},
    )


def metric_with_explain(
    title: str,
    value: str,
    concept: str,
    subtitle: str = "",
    color: str = "cyan",
) -> rx.Component:
    """Display a metric with an explanation button.
    
    Args:
        title: Metric label
        value: Metric value to display
        concept: The concept key for explanation
        subtitle: Optional subtitle/description
        color: Accent color
    """
    color_map = {
        "cyan": "#00d9ff",
        "green": "#10b981",
        "red": "#ef4444",
        "orange": "#f59e0b",
        "purple": "#7c3aed",
    }
    
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text(
                    title,
                    font_size="0.875rem",
                    color="#a0a0b0",
                ),
                explain_button(concept, value),
                spacing="2",
                align_items="center",
            ),
            rx.text(
                value,
                font_size="1.5rem",
                font_weight="bold",
                color=color_map.get(color, "#00d9ff"),
            ),
            rx.cond(
                subtitle != "",
                rx.text(
                    subtitle,
                    font_size="0.75rem",
                    color="#6b7280",
                ),
            ),
            spacing="2",
            align_items="flex_start",
        ),
        bg="#1a1a24",
        padding="1rem",
        border_radius="0.5rem",
        border="1px solid #2a2a3a",
        width="100%",
    )


def explain_modal() -> rx.Component:
    """Create a modal for displaying AI explanations."""
    return rx.dialog.root(
        rx.dialog.content(
            rx.dialog.title(
                rx.hstack(
                    rx.text("AI 指标解释", font_size="1.25rem", color="white"),
                    rx.spacer(),
                    rx.dialog.close(
                        rx.button(
                            rx.icon(tag="x", size=20),
                            variant="ghost",
                            color="#6b7280",
                        ),
                    ),
                    width="100%",
                ),
            ),
            rx.dialog.description(
                rx.vstack(
                    rx.cond(
                        State.ai_explaining,
                        rx.vstack(
                            rx.spinner(color="#00d9ff", size="3"),
                            rx.text("AI 正在生成解释...", color="#6b7280"),
                            spacing="4",
                            align_items="center",
                            padding="2rem",
                        ),
                        rx.box(
                            rx.text(
                                State.current_explanation,
                                color="#e0e0e0",
                                font_size="0.95rem",
                                line_height="1.7",
                                white_space="pre-wrap",
                            ),
                            bg="#1a1a24",
                            padding="1rem",
                            border_radius="0.5rem",
                            border="1px solid #2a2a3a",
                            width="100%",
                        ),
                    ),
                    spacing="4",
                    width="100%",
                ),
            ),
            bg="#0f0f14",
            border="1px solid #2a2a3a",
            border_radius="0.75rem",
            max_width="600px",
            width="90%",
        ),
        open=State.explain_modal_open,
        on_open_change=State.close_explain_modal,
    )
