"""全局状态管理"""

import reflex as rx
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime

from pixiu.services.database import Database
from pixiu.services.data_service import DataService
from pixiu.services.backtest_service import BacktestEngine, BacktestConfig
from pixiu.strategies import get_all_strategies, get_strategy
from pixiu.config import config as app_config


class State(rx.State):
    """应用全局状态"""
    
    is_loading: bool = False
    loading_message: str = ""
    progress: float = 0.0
    
    current_market: str = "A股"
    search_keyword: str = ""
    search_results: List[Dict] = []
    selected_stock: str = ""
    selected_stock_name: str = ""
    
    stock_data: Dict = {}
    chart_data: Dict = {}
    
    available_strategies: List[Dict] = []
    selected_strategies: List[str] = []
    
    backtest_result: Dict = {}
    backtest_config: Dict = {
        "initial_capital": 100000,
        "commission_rate": 0.0003,
        "position_size": 0.95
    }
    
    ai_report: str = ""
    ai_generating: bool = False
    
    def __init__(self):
        super().__init__()
        self.db = Database(str(app_config.data_dir / "stocks.db"))
        self.data_service = DataService(self.db)
        self._load_strategies()
    
    def _load_strategies(self):
        """加载所有可用策略"""
        strategies = get_all_strategies()
        self.available_strategies = [
            {
                "name": s.name,
                "description": s.description,
                "params": s.params
            }
            for s in strategies
        ]
    
    @rx.background
    async def search_stocks(self):
        """搜索股票"""
        async with self:
            self.is_loading = True
            self.loading_message = "搜索中..."
        
        results = await self.data_service.search_stocks(
            self.search_keyword,
            self.current_market
        )
        
        async with self:
            self.search_results = [
                {"code": s.code, "name": s.name, "market": s.market}
                for s in results
            ]
            self.is_loading = False
    
    @rx.background
    async def select_stock(self, code: str):
        """选择股票"""
        async with self:
            self.is_loading = True
            self.loading_message = "加载数据..."
            self.selected_stock = code
        
        df = await self.data_service.get_cached_data(code)
        
        if df.empty:
            success, count = await self.data_service.download_and_save(
                code, self.current_market
            )
            if success:
                df = await self.data_service.get_cached_data(code)
        
        async with self:
            if not df.empty:
                for s in self.search_results:
                    if s["code"] == code:
                        self.selected_stock_name = s["name"]
                        break
                
                self.stock_data = {
                    "dates": df.index.strftime("%Y-%m-%d").tolist(),
                    "close": df["close"].tolist(),
                    "volume": df["volume"].tolist()
                }
                self._update_chart_data(df)
            self.is_loading = False
    
    def _update_chart_data(self, df: pd.DataFrame):
        """更新图表数据"""
        self.chart_data = {
            "data": [{
                "x": df.index.strftime("%Y-%m-%d").tolist(),
                "y": df["close"].tolist(),
                "type": "scatter",
                "mode": "lines",
                "name": "收盘价"
            }],
            "layout": {
                "title": f"{self.selected_stock_name} 价格走势",
                "xaxis": {"title": "日期"},
                "yaxis": {"title": "价格"},
                "showlegend": True
            }
        }
    
    def toggle_strategy(self, strategy_name: str):
        """切换策略选择"""
        if strategy_name in self.selected_strategies:
            self.selected_strategies.remove(strategy_name)
        else:
            self.selected_strategies.append(strategy_name)
    
    @rx.background
    async def run_backtest(self):
        """执行回测"""
        async with self:
            self.is_loading = True
            self.loading_message = "执行回测..."
            self.progress = 0.0
        
        df = pd.DataFrame(self.stock_data)
        df["trade_date"] = pd.to_datetime(df["dates"])
        df = df.set_index("trade_date")
        
        results = []
        total = len(self.selected_strategies)
        
        for i, strategy_name in enumerate(self.selected_strategies):
            strategy = get_strategy(strategy_name)
            if strategy:
                df_with_signals = strategy.generate_signals(df.reset_index())
                
                engine = BacktestEngine(BacktestConfig(**self.backtest_config))
                result = engine.run(df_with_signals)
                
                results.append({
                    "strategy": strategy_name,
                    "result": result
                })
            
            async with self:
                self.progress = (i + 1) / total * 100
        
        async with self:
            self.backtest_result = {
                "results": [
                    {
                        "strategy": r["strategy"],
                        "total_return": r["result"].total_return,
                        "annualized_return": r["result"].annualized_return,
                        "max_drawdown": r["result"].max_drawdown,
                        "sharpe_ratio": r["result"].sharpe_ratio,
                        "win_rate": r["result"].win_rate,
                    }
                    for r in results
                ]
            }
            self.is_loading = False
    
    def set_market(self, market: str):
        """设置市场"""
        self.current_market = market
    
    def set_search_keyword(self, keyword: str):
        """设置搜索关键词"""
        self.search_keyword = keyword
