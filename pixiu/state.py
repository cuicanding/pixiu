"""Global state management - connected to real services."""

import reflex as rx
from typing import List, Dict
from pathlib import Path
import pandas as pd
import sys
from datetime import datetime, timedelta

def debug_log(msg):
    """Debug logging to stdout"""
    print(f"[DEBUG] {msg}", file=sys.stderr, flush=True)

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

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
        "trend_trend": ["趋势强度策略", "均线交叉策略"],
        "trend_range": ["网格交易策略", "RSI策略"],
        "range_trend": ["趋势强度策略", "卡尔曼滤波策略"],
        "range_range": ["网格交易策略", "RSI策略", "波动率套利策略"],
    }
    
    REGIME_EXPLANATION = {
        "trend_trend": """大盘和个股都处于趋势行情。
特点：价格有明确方向，波动有序。
推荐逻辑：趋势跟踪策略可以顺势而为，捕捉方向性收益。
适合策略：趋势强度策略（基于导数判断趋势方向）、均线交叉策略（金叉做多死叉做空）。""",
        
        "trend_range": """大盘趋势但个股震荡。
特点：个股在大趋势中上下波动，没有明确方向。
推荐逻辑：震荡行情中趋势策略容易反复止损，应使用均值回归策略。
适合策略：网格交易（高抛低吸）、RSI策略（超买卖超买卖）。""",
        
        "range_trend": """大盘震荡但个股有趋势。
特点：个股走独立行情，有明确方向。
推荐逻辑：个股趋势明确，可尝试趋势跟踪策略。
适合策略：趋势强度策略、卡尔曼滤波策略（估计真实价格偏离）。""",
        
        "range_range": """大盘和个股都震荡。
特点：价格在一定区间内波动，没有明确方向。
推荐逻辑：震荡行情最适合均值回归类策略。
适合策略：网格交易、RSI、波动率套利（捕捉过度波动后的回归）。""",
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
    backtest_charts: Dict[str, str] = {}
    
    market_regime: str = "unknown"
    stock_regime: str = "unknown"
    market_index_data: Dict = {}
    using_mock_data: bool = False
    recommended_strategies: List[str] = []
    regime_analysis: Dict = {}
    combine_mode: str = "complementary"
    filter_threshold: int = 2
    regime_chart: str = ""
    market_chart: str = ""
    _db_initialized: bool = False
    
    explain_modal_open: bool = False
    current_explanation: str = ""
    ai_explaining: bool = False
    glm_api_key: str = ""
    _db_initialized: bool = False
    _db_initialized: bool = False
    
    time_range_mode: str = "quick"
    quick_range: str = "12m"
    year_range: str = ""
    custom_start_date: str = ""
    custom_end_date: str = ""
    backtest_start_date: str = ""
    backtest_end_date: str = ""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_strategies()
        self._load_settings()
        self._update_date_range()
    
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
    
    def _update_date_range(self):
        """Update backtest start/end dates based on current mode."""
        today = datetime.now()
        self.backtest_end_date = today.strftime("%Y-%m-%d")
        
        if self.time_range_mode == "quick":
            if self.quick_range == "1m":
                start = today - timedelta(days=30)
            elif self.quick_range == "3m":
                start = today - timedelta(days=90)
            elif self.quick_range == "6m":
                start = today - timedelta(days=180)
            elif self.quick_range == "12m":
                start = today - timedelta(days=365)
            elif self.quick_range == "24m":
                start = today - timedelta(days=730)
            elif self.quick_range == "36m":
                start = today - timedelta(days=1095)
            else:
                start = today - timedelta(days=365)
            self.backtest_start_date = start.strftime("%Y-%m-%d")
        elif self.time_range_mode == "year":
            if self.year_range == "this_year":
                self.backtest_start_date = f"{today.year}-01-01"
            elif self.year_range == "last_year":
                self.backtest_start_date = f"{today.year - 1}-01-01"
                self.backtest_end_date = f"{today.year - 1}-12-31"
            elif self.year_range == "2023":
                self.backtest_start_date = "2023-01-01"
                self.backtest_end_date = "2023-12-31"
            elif self.year_range == "2024":
                self.backtest_start_date = "2024-01-01"
                self.backtest_end_date = "2024-12-31"
            elif self.year_range == "last_3_years":
                self.backtest_start_date = f"{today.year - 3}-01-01"
            elif self.year_range == "last_5_years":
                self.backtest_start_date = f"{today.year - 5}-01-01"
            else:
                start = today - timedelta(days=365)
                self.backtest_start_date = start.strftime("%Y-%m-%d")
        elif self.time_range_mode == "custom":
            self.backtest_start_date = self.custom_start_date
            self.backtest_end_date = self.custom_end_date
    
    def set_quick_range(self, range_value: str):
        self.time_range_mode = "quick"
        self.quick_range = range_value
        self._update_date_range()
    
    def set_year_range(self, range_value: str):
        self.time_range_mode = "year"
        self.year_range = range_value
        self._update_date_range()
    
    def set_custom_start(self, date_str: str):
        self.custom_start_date = date_str
        if self.custom_end_date:
            self.time_range_mode = "custom"
            self._update_date_range()
    
    def set_custom_end(self, date_str: str):
        self.custom_end_date = date_str
        if self.custom_start_date:
            self.time_range_mode = "custom"
            self._update_date_range()
    
    async def _ensure_db_initialized(self):
        """Ensure database tables exist. Internal method without yield."""
        if self._db_initialized:
            return True
        try:
            db_path = Path("data/stocks.db")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            db = Database(str(db_path))
            await db.ensure_tables()
            self._db_initialized = True
            return True
        except Exception as e:
            self.error_message = f"数据库初始化失败: {str(e)}"
            return False
    
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
        debug_log(f"[选择股票] 用户选择: {code}")
        self.selected_stock = code
        self.selected_stock_name = ""
        for stock in self.search_results:
            if stock["code"] == code:
                self.selected_stock_name = stock["name"]
                break
        if self.selected_stock_name:
            self.current_step = self.STEP_REGIME
            self.max_step = max(self.max_step, self.STEP_REGIME)
            debug_log(f"[选择股票] 完成: {code} - {self.selected_stock_name}, 跳转到步骤 {self.current_step}")
        yield
    
    def toggle_strategy(self, strategy_name: str):
        if strategy_name in self.selected_strategies:
            self.selected_strategies = [s for s in self.selected_strategies if s != strategy_name]
        else:
            self.selected_strategies = self.selected_strategies + [strategy_name]
    
    async def run_backtest(self):
        debug_log(f"[回测] 开始执行, 股票: {self.selected_stock}, 策略: {self.selected_strategies}")
        
        await self._ensure_db_initialized()
        if not self.selected_stock or not self.selected_strategies:
            self.error_message = "请先选择股票和策略"
            debug_log(f"[回测] 参数不足: stock={self.selected_stock}, strategies={self.selected_strategies}")
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
            
            debug_log(f"[回测] 获取股票数据: {self.selected_stock}")
            df = await data_service.get_cached_data(self.selected_stock)
            
            if df is None or (hasattr(df, 'empty') and df.empty):
                debug_log("[回测] 缓存无数据，尝试下载...")
                success, _ = await data_service.download_and_save(
                    self.selected_stock,
                    self.current_market
                )
                if success:
                    df = await data_service.get_cached_data(self.selected_stock)
            
            if df is None or (hasattr(df, 'empty') and df.empty):
                self.error_message = "无法获取股票数据"
                self.is_loading = False
                debug_log("[回测] 无法获取股票数据")
                yield
                return
            
            debug_log(f"[回测] 数据加载成功: {len(df)} 条")
            
            backtest_config = BacktestConfig(
                initial_capital=self.initial_capital,
                commission_rate=self.commission_rate,
                position_size=self.position_size,
            )
            
            from pixiu.strategies import get_strategy
            from pixiu.services.chart_service import generate_backtest_chart
            
            total = len(self.selected_strategies)
            self.backtest_charts = {}
            for i, strategy_name in enumerate(self.selected_strategies):
                self.loading_message = f"回测策略: {strategy_name}"
                self.progress = int((i / total) * 80) if total > 0 else 0
                yield
                
                debug_log(f"[回测] 执行策略 {i+1}/{total}: {strategy_name}")
                strategy = get_strategy(strategy_name)
                if strategy is None:
                    debug_log(f"[回测] 策略不存在: {strategy_name}")
                    continue
                
                engine = BacktestEngine(backtest_config)
                df_with_signals = strategy.generate_signals(df)
                
                if 'signal' not in df_with_signals.columns:
                    debug_log(f"[回测] 策略 {strategy_name} 未生成signal列")
                    continue
                
                signal_series = df_with_signals['signal'].values if hasattr(df_with_signals['signal'], 'values') else df_with_signals['signal']
                result = engine.run(df_with_signals, signal_series)
                
                debug_log(f"[回测] 策略 {strategy_name} 完成: 收益={result.total_return:.2f}%, 夏普={result.sharpe_ratio:.2f}")
                
                chart_base64_value = ""
                try:
                    chart_base64 = generate_backtest_chart(
                        df_with_signals,
                        result.trades,
                        result.equity_curve,
                        result.drawdown_curve
                    )
                    self.backtest_charts[strategy_name] = chart_base64
                    chart_base64_value = chart_base64
                    debug_log(f"[回测] 策略 {strategy_name} 图表生成成功, 大小: {len(chart_base64)}")
                    yield
                except Exception as chart_err:
                    debug_log(f"[回测] 图表生成失败: {chart_err}")
                    import traceback
                    traceback.print_exc()
                
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
                    "chart": chart_base64_value,
                }]
            
            self.progress = 100
            self.loading_message = "回测完成"
            self.current_step = self.STEP_RESULT
            self.max_step = max(self.max_step, self.STEP_RESULT)
            debug_log(f"[回测] 全部完成，跳转到步骤 {self.current_step}, 结果数: {len(self.backtest_results)}")
            yield
            
        except Exception as e:
            self.error_message = f"回测失败: {str(e)}"
            debug_log(f"[回测] 失败: {e}")
            import traceback
            traceback.print_exc()
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
        data_service = DataService(db, use_mock=False)
        
        market_codes = {
            "A股": "000001",
            "港股": "HSI",
            "美股": "DJI",
        }
        code = market_codes.get(self.current_market, "000001")
        df = await data_service.fetch_stock_history(code, self.current_market)
        self.using_mock_data = data_service.use_mock
        return df
    
    async def analyze_regime(self):
        """Analyze both market and stock regime"""
        from pixiu.analysis import MarketRegimeDetector
        from pixiu.services.data_service import DataService
        from pixiu.services.chart_service import generate_regime_chart
        
        debug_log(f"[择势分析] 开始执行, 股票: {self.selected_stock}, 市场: {self.current_market}")
        debug_log(f"[择势分析] 时间范围: {self.backtest_start_date} ~ {self.backtest_end_date}")
        
        self.is_loading = True
        self.loading_message = "分析市场和个股状态..."
        self.error_message = ""
        yield
        
        try:
            await self._ensure_db_initialized()
            debug_log("[择势分析] 数据库初始化完成")
            
            db = Database("data/stocks.db")
            data_service = DataService(db, use_mock=False)
            detector = MarketRegimeDetector()
            
            # 获取大盘真实数据
            debug_log(f"[择势分析] 获取大盘指数数据...")
            market_codes = {"A股": "000001", "港股": "HSI", "美股": "DJI"}
            market_code = market_codes.get(self.current_market, "000001")
            
            self.loading_message = f"获取大盘数据 ({market_code})..."
            yield
            
            market_df = await data_service.fetch_stock_history(
                market_code, 
                self.current_market if self.current_market != "A股" else "index",
                self.backtest_start_date,
                self.backtest_end_date
            )
            
            if market_df is not None and not market_df.empty:
                market_analysis = detector.get_analysis_detail(market_df)
                self.market_regime = market_analysis["regime"]
                self.market_index_data = market_analysis
                debug_log(f"[择势分析] 大盘状态: {self.market_regime}, ADX: {market_analysis.get('adx'):.2f}")
                
                # 生成大盘图表
                try:
                    self.market_chart = generate_regime_chart(market_df, market_analysis)
                    debug_log(f"[择势分析] 大盘图表生成成功")
                except Exception as e:
                    debug_log(f"[择势分析] 大盘图表生成失败: {e}")
            else:
                debug_log("[择势分析] 大盘数据获取失败，使用模拟数据")
                market_df = data_service._generate_mock_history(market_code)
                market_analysis = detector.get_analysis_detail(market_df)
                self.market_regime = market_analysis["regime"]
                self.market_index_data = market_analysis
                self.using_mock_data = True
            
            # 获取个股数据
            debug_log(f"[择势分析] 分析个股: {self.selected_stock}")
            self.loading_message = f"获取个股数据 ({self.selected_stock})..."
            yield
            
            if self.selected_stock:
                df = await data_service.fetch_stock_history(
                    self.selected_stock, 
                    self.current_market,
                    self.backtest_start_date,
                    self.backtest_end_date
                )
                debug_log(f"[择势分析] 获取到数据: {len(df) if df is not None and not df.empty else 0} 条")
                
                if df is not None and not df.empty:
                    stock_analysis = detector.get_analysis_detail(df)
                    self.stock_regime = stock_analysis["regime"]
                    self.regime_analysis = stock_analysis
                    
                    # 生成个股图表
                    try:
                        self.regime_chart = generate_regime_chart(df, stock_analysis)
                        debug_log(f"[择势分析] 个股图表生成成功")
                    except Exception as e:
                        debug_log(f"[择势分析] 个股图表生成失败: {e}")
                    
                    debug_log(f"[择势分析] 个股状态: {self.stock_regime}, ADX: {stock_analysis.get('adx'):.2f}")
                else:
                    debug_log("[择势分析] 无数据，使用模拟数据")
                    df = data_service._generate_mock_history(self.selected_stock)
                    stock_analysis = detector.get_analysis_detail(df)
                    self.stock_regime = stock_analysis["regime"]
                    self.regime_analysis = stock_analysis
                    self.regime_chart = generate_regime_chart(df, stock_analysis)
                    self.using_mock_data = True
                    debug_log(f"[择势分析] 个股状态(模拟): {self.stock_regime}")

            self.recommended_strategies = self.regime_recommendations
            debug_log(f"[择势分析] 推荐策略: {self.recommended_strategies}")
            debug_log(f"[择势分析] 完成，停留在择势分析步骤查看结果")
            
        except Exception as e:
            self.error_message = f"择势分析失败: {str(e)}"
            debug_log(f"[择势分析] 失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_loading = False
            self.loading_message = ""
        yield

    def set_combine_mode(self, mode: str):
        if mode in ["equal_weight", "signal_filter", "complementary"]:
            self.combine_mode = mode

    def go_to_strategy_step(self):
        """跳转到策略选择步骤，自动选中推荐策略"""
        self.selected_strategies = self.recommended_strategies.copy() if self.recommended_strategies else []
        self.current_step = self.STEP_STRATEGY
        self.max_step = max(self.max_step, self.STEP_STRATEGY)

    @rx.var
    def backtest_results_empty(self) -> bool:
        """Check if backtest results are empty."""
        return len(self.backtest_results) == 0

    @rx.var
    def regime_recommendations(self) -> List[str]:
        """Get strategy recommendations based on regime analysis."""
        key = f"{self.market_regime}_{self.stock_regime}"
        return self.REGIME_STRATEGY_MAP.get(key, [])
    
    @rx.var
    def regime_explanation(self) -> str:
        """Get explanation for regime-based recommendations."""
        key = f"{self.market_regime}_{self.stock_regime}"
        return self.REGIME_EXPLANATION.get(key, "")
    
    @rx.var
    def regime_summary(self) -> str:
        """Get short summary of current regime."""
        market_text = "趋势行情" if self.market_regime == "trend" else "震荡行情"
        stock_text = "趋势行情" if self.stock_regime == "trend" else "震荡行情"
        return f"大盘: {market_text} | 个股: {stock_text}"

    def get_backtest_chart(self, strategy: str) -> str:
        """Get backtest chart for a specific strategy."""
        return self.backtest_charts.get(strategy, "")

    async def explain_concept(self, concept: str, value: str = ""):
        """Generate AI explanation for a concept."""
        self.explain_modal_open = True
        self.ai_explaining = True
        self.current_explanation = ""
        yield
        
        try:
            from pixiu.services.ai_service import AIReportService
            from pixiu.services.explain_prompts import get_prompt
            
            prompt = get_prompt(concept, value=value, regime=self.stock_regime)
            self.current_explanation = await AIReportService(self.glm_api_key)._call_api(prompt)
        except Exception as e:
            self.current_explanation = f"解释生成失败: {str(e)}"
        finally:
            self.ai_explaining = False
        yield

    def close_explain_modal(self):
        """Close the explanation modal."""
        self.explain_modal_open = False
