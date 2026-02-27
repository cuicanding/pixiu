"""Global state management - connected to real services."""

import reflex as rx
from typing import List, Dict, Optional
from pathlib import Path

from pixiu.services.database import Database
from pixiu.services.data_service import DataService
from pixiu.services.backtest_service import BacktestEngine, BacktestConfig
from pixiu.services.ai_service import AIService
from pixiu.strategies import get_all_strategies, get_strategy
from pixiu.config import config


class State(rx.State):
    """Application global state with real service integration."""
    
    is_loading: bool = False
    loading_message: str = ""
    progress: int = 0
    error_message: str = ""
    
    current_market: str = "A股"
    search_keyword: str = ""
    search_results: List[Dict] = []
    
    selected_stock: str = ""
    selected_stock_name: str = ""
    
    available_strategies: List[Dict] = []
    selected_strategies: List[str] = []
    
    initial_capital: float = 100000.0
    commission_rate: float = 0.0003
    position_size: float = 0.95
    
    backtest_results: List[Dict] = []
    
    ai_report: str = ""
    ai_generating: bool = False
    
    glm_api_key: str = ""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_strategies()
        self._load_settings()
    
    def _load_strategies(self):
        """Load available strategies from registry."""
        try:
            strategies = get_all_strategies()
            self.available_strategies = [
                {
                    "name": s.name,
                    "description": s.description,
                }
                for s in strategies
            ]
        except Exception:
            self.available_strategies = [
                {"name": "趋势强度策略", "description": "基于导数分析的趋势跟踪"},
                {"name": "波动率套利策略", "description": "基于波动率的均值回归"},
                {"name": "卡尔曼滤波策略", "description": "基于状态空间估计的信号"},
            ]
    
    def _load_settings(self):
        """Load settings from config."""
        self.glm_api_key = getattr(config, 'glm_api_key', "") or ""
        self.initial_capital = getattr(config, 'initial_capital', 100000.0)
        self.commission_rate = getattr(config, 'commission_rate', 0.0003)
        self.position_size = getattr(config, 'position_size', 0.95)
    
    def set_market_a(self):
        self.current_market = "A股"
    
    def set_market_hk(self):
        self.current_market = "港股"
    
    def set_market_us(self):
        self.current_market = "美股"
    
    def set_search_keyword(self, keyword: str):
        self.search_keyword = keyword
    
    async def search_stocks(self):
        """Search stocks using real DataService."""
        if not self.search_keyword:
            return
        
        self.is_loading = True
        self.error_message = ""
        self.search_results = []
        yield
        
        try:
            db_path = Path("data/stocks.db")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            db = Database(str(db_path))
            data_service = DataService(db)
            
            results = await data_service.search_stocks(
                self.search_keyword,
                self.current_market
            )
            
            self.search_results = [
                {"code": s.code, "name": s.name, "market": s.market}
                for s in results[:10]
            ]
        except Exception as e:
            self.error_message = f"搜索失败: {str(e)}"
        finally:
            self.is_loading = False
            yield
    
    async def select_stock(self, code: str):
        """Select a stock and load its data."""
        self.selected_stock = code
        self.selected_stock_name = ""
        
        for stock in self.search_results:
            if stock["code"] == code:
                self.selected_stock_name = stock["name"]
                break
        
        yield
    
    def toggle_strategy(self, strategy_name: str):
        """Toggle strategy selection."""
        if strategy_name in self.selected_strategies:
            self.selected_strategies = [s for s in self.selected_strategies if s != strategy_name]
        else:
            self.selected_strategies = self.selected_strategies + [strategy_name]
    
    async def run_backtest(self):
        """Run backtest using real BacktestEngine."""
        if not self.selected_stock or not self.selected_strategies:
            self.error_message = "请先选择股票和策略"
            yield
            return
        
        self.is_loading = True
        self.progress = 0
        self.error_message = ""
        self.backtest_results = []
        yield
        
        try:
            db = Database("data/stocks.db")
            data_service = DataService(db)
            
            self.loading_message = "加载股票数据..."
            yield
            
            df = await data_service.get_cached_data(self.selected_stock)
            
            if df is None or (hasattr(df, 'empty') and df.empty):
                success, _ = await data_service.download_and_save(
                    self.selected_stock,
                    self.current_market
                )
                if success:
                    df = await data_service.get_cached_data(self.selected_stock)
            
            if df is None or (hasattr(df, 'empty') and df.empty):
                self.error_message = "无法获取股票数据"
                self.is_loading = False
                yield
                return
            
            backtest_config = BacktestConfig(
                initial_capital=self.initial_capital,
                commission_rate=self.commission_rate,
                position_size=self.position_size,
            )
            
            total = len(self.selected_strategies)
            for i, strategy_name in enumerate(self.selected_strategies):
                self.loading_message = f"回测策略: {strategy_name}"
                self.progress = int((i / total) * 80) if total > 0 else 0
                yield
                
                strategy = get_strategy(strategy_name)
                engine = BacktestEngine(backtest_config)
                result = engine.run(df, strategy)
                
                trades = []
                for t in result.trades[:50]:
                    trades.append({
                        "date": t.date.strftime("%Y-%m-%d"),
                        "type": t.trade_type,
                        "price": float(t.price),
                        "shares": float(t.shares),
                        "pnl": float(t.pnl) if t.pnl else 0,
                    })
                
                self.backtest_results = self.backtest_results + [{
                    "strategy": strategy_name,
                    "total_return": float(result.total_return),
                    "annualized_return": float(result.annualized_return),
                    "max_drawdown": float(result.max_drawdown),
                    "sharpe_ratio": float(result.sharpe_ratio),
                    "win_rate": float(result.win_rate),
                    "profit_loss_ratio": float(result.profit_loss_ratio),
                    "trades": trades,
                }]
            
            self.progress = 100
            yield
            
            return rx.redirect("/backtest")
            
        except Exception as e:
            self.error_message = f"回测失败: {str(e)}"
        finally:
            self.is_loading = False
            self.loading_message = ""
            yield
    
    async def generate_ai_report(self):
        """Generate AI analysis report."""
        if not self.glm_api_key:
            self.error_message = "请先在设置中配置 GLM API Key"
            yield
            return
        
        if not self.backtest_results:
            self.error_message = "请先执行回测"
            yield
            return
        
        self.ai_generating = True
        self.ai_report = ""
        yield
        
        try:
            ai_service = AIService(self.glm_api_key)
            self.ai_report = await ai_service.generate_analysis_report(
                self.selected_stock,
                self.selected_stock_name,
                self.backtest_results,
            )
        except Exception as e:
            self.error_message = f"AI 报告生成失败: {str(e)}"
            self.ai_report = ""
        finally:
            self.ai_generating = False
            yield
    
    def set_glm_api_key(self, key: str):
        self.glm_api_key = key
    
    def set_initial_capital(self, value: str):
        try:
            self.initial_capital = float(value)
        except ValueError:
            pass
    
    def set_commission_rate(self, value: str):
        try:
            self.commission_rate = float(value)
        except ValueError:
            pass
    
    def set_position_size(self, value: str):
        try:
            self.position_size = float(value)
        except ValueError:
            pass
    
    def save_settings(self):
        """Save settings to config and environment."""
        import os
        os.environ["GLM_API_KEY"] = self.glm_api_key
        if hasattr(config, 'glm_api_key'):
            config.glm_api_key = self.glm_api_key
        if hasattr(config, 'initial_capital'):
            config.initial_capital = self.initial_capital
        if hasattr(config, 'commission_rate'):
            config.commission_rate = self.commission_rate
        if hasattr(config, 'position_size'):
            config.position_size = self.position_size
    
    def clear_error(self):
        """Clear error message."""
        self.error_message = ""
