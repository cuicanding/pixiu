"""Strategy selection card component."""
import reflex as rx


def strategy_card(
    strategy: dict,
    is_selected: bool,
    on_click_handler,
) -> rx.Component:
    """Display a selectable strategy card.
    
    Args:
        strategy: Dict with 'name' and 'description' keys
        is_selected: Whether this strategy is selected
        on_click_handler: Event handler for toggle
    """
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text(
                    "V" if is_selected else "O",
                    color="#10b981" if is_selected else "#6b7280",
                    font_size="1.25rem",
                ),
                rx.text(
                    strategy["name"],
                    font_weight="bold",
                    color="#ffffff",
                ),
                spacing="0.5rem",
                width="100%",
            ),
            rx.text(
                strategy.get("description", ""),
                font_size="0.75rem",
                color="#a0a0b0",
            ),
            spacing="0.5rem",
            align_items="flex_start",
        ),
        padding="1rem",
        border_radius="0.5rem",
        cursor="pointer",
        border=f"2px solid {'#10b981' if is_selected else '#2a2a3a'}",
        bg="#1a1a24" if is_selected else "transparent",
        on_click=on_click_handler,
        _hover={
            "border_color": "#00d9ff",
        },
    )
