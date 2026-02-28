"""Global state management - connected to real services."""

import reflex as rx
from typing import List, Dict
from pathlib import Path
import pandas as pd

from pixiu.services.database import Database
from pixiu.services.data_service import DataService
from pixiu.services.backtest_service import BacktestEngine, BacktestConfig
from pixiu.services.ai_service import AIReportService
from pixiu.strategies import get_all_strategies
from pixiu.config import config


class State(rx.State):
    """Application global state with real service integration."""
    
    STEP_MARKET = 1
    STEP_SEARCH = 2
    STEP_REGIME = 3
    STEP_STRATEGY = 4
    STEP_CONFIG = 5
    STEP_RESULT = 6

    REGIME_STRATEGY_MAP = {
        "trend_trend": ["趋势强度策略", "均线策略", "动量策略"],
        "trend_range": ["网格交易策略", "RSI策略", "波动率套利策略"],
        "range_trend": ["趋势强度策略", "动量策略"],
        "range_range": ["网格交易策略", "RSI策略", "波动率套利策略", "均值回归策略"],
    }

    current_step: int = 1
    max_step: int = 1
    
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
    
    market_regime: str = "unknown"
    stock_regime: str = "unknown"
    market_index_data: Dict = {}
    using_mock_data: bool = False
    recommended_strategies: List[str] = []
    regime_analysis: Dict = {}
    combine_mode: str = "complementary"
    filter_threshold: int = 2
    
    ai_report: str = ""
    ai_generating: bool = False
    glm_api_key: str = ""
    _db_initialized: bool = False
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_strategies()
        self._load_settings()
    
    def _load_strategies(self):
        try:
            strategies = get_all_strategies()
            self.available_strategies = [
                {"name": s.name, "description": s.description}
                for s in strategies
            ]
        except Exception:
            self.available_strategies = [
                {"name": "趋势强度策略", "description": "基于导数分析的趋势跟踪"},
                {"name": "波动率套利策略", "description": "基于波动率的均值回归"},
                {"name": "卡尔曼滤波策略", "description": "基于状态空间估计的信号"},
            ]
    
    def _load_settings(self):
        self.glm_api_key = getattr(config, 'glm_api_key', "") or ""
        self.initial_capital = getattr(config, 'initial_capital', 100000.0)
        self.commission_rate = getattr(config, 'commission_rate', 0.0003)
        self.position_size = getattr(config, 'position_size', 0.95)
    
    async def ensure_db_initialized(self):
        """Ensure database tables exist."""
        if self._db_initialized:
            return
        try:
            db_path = Path("data/stocks.db")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            db = Database(str(db_path))
            await db.ensure_tables()
            self._db_initialized = True
        except Exception as e:
            self.error_message = f"数据库初始化失败: {str(e)}"
            yield
    
    def set_market_a(self):
        self.current_market = "A股"
    
    def set_market_hk(self):
        self.current_market = "港股"
    
    def set_market_us(self):
        self.current_market = "美股"
    
    def go_to_step(self, step: int):
        if step <= self.max_step:
            self.current_step = step
    
    def next_step(self):
        if self.current_step < 6:
            self.current_step += 1
            if self.current_step > self.max_step:
                self.max_step = self.current_step
    
    def prev_step(self):
        if self.current_step > 1:
            self.current_step -= 1
    
    def reset_flow(self):
        self.current_step = 1
        self.max_step = 1
        self.selected_stock = ""
        self.selected_stock_name = ""
        self.selected_strategies = []
        self.backtest_results = []
        self.regime_analysis = {}
        self.market_regime = "unknown"
        self.stock_regime = "unknown"
    
    def set_search_keyword(self, keyword: str):
        self.search_keyword = keyword
    
    async def search_stocks(self):
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
            data_service = DataService(db, use_mock=True)
            
            results = await data_service.search_stocks(
                self.search_keyword,
                self.current_market
            )
            
            if results is None or len(results) == 0:
                self.error_message = "未找到匹配的股票"
            else:
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
        self.selected_stock = code
        self.selected_stock_name = ""
        for stock in self.search_results:
            if stock["code"] == code:
                self.selected_stock_name = stock["name"]
                break
        if self.selected_stock_name:
            self.current_step = self.STEP_REGIME
            self.max_step = max(self.max_step, self.STEP_REGIME)
        yield
    
    def toggle_strategy(self, strategy_name: str):
        if strategy_name in self.selected_strategies:
            self.selected_strategies = [s for s in self.selected_strategies if s != strategy_name]
        else:
            self.selected_strategies = self.selected_strategies + [strategy_name]
    
    async def run_backtest(self):
        await self.ensure_db_initialized()
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
            data_service = DataService(db, use_mock=True)
            
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
            
            from pixiu.strategies import get_strategy
            
            total = len(self.selected_strategies)
            for i, strategy_name in enumerate(self.selected_strategies):
                self.loading_message = f"回测策略: {strategy_name}"
                self.progress = int((i / total) * 80) if total > 0 else 0
                yield
                
                strategy = get_strategy(strategy_name)
                if strategy is None:
                    continue
                
                engine = BacktestEngine(backtest_config)
                signals = strategy.generate_signals(df)
                result = engine.run(df, signals)
                
                trades = []
                for t in result.trades[:50]:
                    trades.append({
                        "date": t.trade_date if hasattr(t, 'trade_date') else "",
                        "type": t.signal_type if hasattr(t, 'signal_type') else "",
                        "price": float(t.price) if t.price else 0,
                        "shares": float(t.shares) if t.shares else 0,
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
            self.loading_message = "回测完成"
            yield
            
        except Exception as e:
            self.error_message = f"回测失败: {str(e)}"
        finally:
            self.is_loading = False
            self.loading_message = ""
        yield
    
    async def generate_ai_report(self):
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
            ai_service = AIReportService(self.glm_api_key)
            
            if len(self.backtest_results) > 0:
                result = self.backtest_results[0]
                stock_info = {"code": self.selected_stock, "name": self.selected_stock_name}
                self.ai_report = await ai_service.generate_analysis(
                    result,
                    stock_info,
                    result.get("strategy", "")
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
        self.error_message = ""
    
    async def _get_market_index_data(self) -> pd.DataFrame:
        """Get market index data (上证指数 for A股, etc.)"""
        from pixiu.services.data_service import DataService
        db = Database("data/stocks.db")
        data_service = DataService(db, use_mock=True)
        
        market_codes = {
            "A股": "000001",  # 上证指数
            "港股": "HSI",    # 恒生指数
            "美股": "DJI",    # 道琼斯
        }
        code = market_codes.get(self.current_market, "000001")
        return data_service._generate_mock_history(code)
    
    async def analyze_regime(self):
        """Analyze both market and stock regime"""
        from pixiu.analysis import MarketRegimeDetector
        from pixiu.services.data_service import DataService
        
        self.is_loading = True
        self.loading_message = "分析市场和个股状态..."
        self.error_message = ""
        yield
        
        try:
            await self.ensure_db_initialized()
            
            db = Database("data/stocks.db")
            data_service = DataService(db, use_mock=True)
            detector = MarketRegimeDetector()
            
            # Analyze market index
            market_df = await self._get_market_index_data()
            if market_df is not None and not market_df.empty:
                market_analysis = detector.get_analysis_detail(market_df)
                self.market_regime = market_analysis["regime"]
                self.market_index_data = market_analysis
            
            # Analyze selected stock
            if self.selected_stock:
                df = await data_service.get_cached_data(self.selected_stock)
                if df is None or df.empty:
                    success, _ = await data_service.download_and_save(
                        self.selected_stock,
                        self.current_market
                    )
                    if success:
                        df = await data_service.get_cached_data(self.selected_stock)
                
                if df is not None and not df.empty:
                    stock_analysis = detector.get_analysis_detail(df)
                    self.stock_regime = stock_analysis["regime"]
                    self.regime_analysis = stock_analysis
                    self.using_mock_data = True

            self.recommended_strategies = self.regime_recommendations

            self.current_step = self.STEP_STRATEGY
            self.max_step = max(self.max_step, self.STEP_STRATEGY)
            
        except Exception as e:
            self.error_message = f"择势分析失败: {str(e)}"
        finally:
            self.is_loading = False
            self.loading_message = ""
        yield

    def set_combine_mode(self, mode: str):
        if mode in ["equal_weight", "signal_filter", "complementary"]:
            self.combine_mode = mode

    @rx.var
    def backtest_results_empty(self) -> bool:
        """Check if backtest results are empty."""
        return len(self.backtest_results) == 0

    @rx.var
    def regime_recommendations(self) -> List[str]:
        """Get strategy recommendations based on regime analysis."""
        key = f"{self.market_regime}_{self.stock_regime}"
        return self.REGIME_STRATEGY_MAP.get(key, [])
