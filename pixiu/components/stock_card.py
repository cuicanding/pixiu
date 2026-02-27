"""Stock search result card component."""
import reflex as rx


def stock_card(stock: dict, on_click_handler) -> rx.Component:
    """Display a stock search result.
    
    Args:
        stock: Dict with 'code' and 'name' keys
        on_click_handler: Event handler for selection
    """
    return rx.box(
        rx.hstack(
            rx.text(
                stock["code"],
                font_weight="bold",
                color="#00d9ff",
                min_width="80px",
            ),
            rx.text(
                stock["name"],
                color="#ffffff",
            ),
            rx.spacer(),
            rx.text(
                stock.get("market", ""),
                font_size="0.75rem",
                color="#6b7280",
            ),
            width="100%",
        ),
        padding="0.75rem 1rem",
        cursor="pointer",
        border_radius="0.375rem",
        on_click=on_click_handler,
        _hover={
            "bg": "#252532",
        },
    )
