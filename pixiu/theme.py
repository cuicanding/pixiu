"""Dark tech theme configuration for Pixiu."""

import reflex as rx

COLORS = {
    "bg_primary": "#0a0a0f",
    "bg_secondary": "#12121a",
    "bg_card": "#1a1a24",
    "bg_hover": "#252532",
    "accent_cyan": "#00d9ff",
    "accent_purple": "#7c3aed",
    "accent_green": "#10b981",
    "accent_red": "#ef4444",
    "accent_orange": "#f59e0b",
    "text_primary": "#ffffff",
    "text_secondary": "#a0a0b0",
    "text_muted": "#6b7280",
    "border": "#2a2a3a",
}


def get_theme() -> rx.Component:
    """Get the dark theme configuration."""
    return rx.theme(
        appearance="dark",
        has_background=True,
        radius="medium",
        accent_color="cyan",
    )
