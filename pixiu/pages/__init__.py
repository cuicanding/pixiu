"""Pixiu 页面模块"""

from .home import page as home_page
from .backtest import page as backtest_page
from .settings import page as settings_page

__all__ = ["home_page", "backtest_page", "settings_page"]
