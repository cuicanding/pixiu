"""Pixiu 量化分析应用入口"""

import reflex as rx

from pixiu.pages.home import page as home_page
from pixiu.pages.backtest import page as backtest_page
from pixiu.pages.settings import page as settings_page

app = rx.App()
app.add_page(home_page, route="/", title="Pixiu 量化分析")
app.add_page(backtest_page, route="/backtest", title="回测报告 - Pixiu")
app.add_page(settings_page, route="/settings", title="设置 - Pixiu")
