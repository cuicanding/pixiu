"""Pixiu quantitative analysis application."""

import reflex as rx
from pixiu.pages.home import page as home_page
from pixiu.pages.backtest import page as backtest_page
from pixiu.pages.settings import page as settings_page
from pixiu.pages.experiment import experiment_page
from pixiu.state import State


app = rx.App(
    theme=rx.theme(
        appearance="dark",
        has_background=True,
        accent_color="cyan",
    ),
)

app.add_page(home_page, route="/", title="Pixiu 量化分析", on_load=State.on_load)
app.add_page(backtest_page, route="/backtest", title="回测报告 - Pixiu", on_load=State.on_load)
app.add_page(settings_page, route="/settings", title="设置 - Pixiu", on_load=State.on_load)
app.add_page(experiment_page, route="/experiment", title="量化实验 - Pixiu", on_load=State.on_load)
