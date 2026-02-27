"""Metric display card component."""
import reflex as rx
from typing import Optional


def metric_card(
    title: str,
    value: str,
    subtitle: Optional[str] = None,
    color: str = "cyan",
) -> rx.Component:
    """Display a metric with title and value.
    
    Args:
        title: Metric label
        value: Metric value to display
        subtitle: Optional subtitle/description
        color: Accent color (cyan, green, red, orange, purple)
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
            rx.text(
                title,
                font_size="0.875rem",
                color="#a0a0b0",
            ),
            rx.text(
                value,
                font_size="1.5rem",
                font_weight="bold",
                color=color_map.get(color, "#00d9ff"),
            ),
            rx.cond(
                subtitle != None,
                rx.text(
                    subtitle,
                    font_size="0.75rem",
                    color="#6b7280",
                ),
            ),
            spacing="0.5rem",
            align_items="flex_start",
        ),
        bg="#1a1a24",
        padding="1rem",
        border_radius="0.5rem",
        border="1px solid #2a2a3a",
        width="100%",
    )
