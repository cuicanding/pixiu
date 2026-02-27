"""Pixiu 量化分析应用入口"""

import reflex as rx

from pixiu.pages.home import page as home_page

app = rx.App()
app.add_page(home_page, route="/", title="Pixiu 量化分析")
