"""全局状态管理"""

import reflex as rx
from typing import List, Dict


class State(rx.State):
    """应用全局状态"""
    
    is_loading: bool = False
    loading_message: str = ""
    progress: int = 0
    
    current_market: str = "A股"
    search_keyword: str = ""
    search_results: List[Dict] = []
    selected_stock: str = ""
    selected_stock_name: str = ""
    
    available_strategies: List[Dict] = [
        {"name": "均线策略", "description": "基于移动平均线的交易策略"},
        {"name": "RSI策略", "description": "基于相对强弱指标的交易策略"},
    ]
    selected_strategies: List[str] = []
    
    backtest_result: Dict = {}

    def set_market_a(self):
        self.current_market = "A股"
    
    def set_market_hk(self):
        self.current_market = "港股"
    
    def set_market_us(self):
        self.current_market = "美股"

    def set_search_keyword(self, keyword: str):
        self.search_keyword = keyword

    async def search_stocks(self):
        self.is_loading = True
        self.loading_message = "搜索中..."
        yield
        
        # 模拟搜索结果
        self.search_results = [
            {"code": "000001", "name": "平安银行"},
            {"code": "000002", "name": "万科A"},
        ]
        self.is_loading = False
        yield

    async def select_stock(self, code: str):
        self.selected_stock = code
        for s in self.search_results:
            if s["code"] == code:
                self.selected_stock_name = s["name"]
                break
        yield

    async def toggle_strategy(self, strategy_name: str):
        if strategy_name in self.selected_strategies:
            self.selected_strategies.remove(strategy_name)
        else:
            self.selected_strategies.append(strategy_name)
        yield

    async def run_backtest(self):
        self.is_loading = True
        self.loading_message = "执行回测..."
        self.progress = 0
        yield
        
        # 模拟回测
        self.progress = 50
        yield
        
        self.progress = 100
        self.backtest_result = {"total_return": 15.5, "sharpe": 1.2}
        self.is_loading = False
        self.loading_message = ""
        yield
