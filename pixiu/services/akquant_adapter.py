"""AKQuant适配器"""
from typing import Dict, Optional
import pandas as pd

try:
    import akquant as aq
    from akquant import Strategy
    AKQUANT_AVAILABLE = True
except ImportError:
    AKQUANT_AVAILABLE = False
    aq = None
    Strategy = None

from pixiu.models.backtest import BacktestResult, Trade


class AKQuantAdapter:
    """AKQuant高性能回测适配器"""
    
    def __init__(self):
        if not AKQUANT_AVAILABLE:
            import warnings
            warnings.warn("AKQuant not installed, falling back to built-in engine")
    
    def run_backtest(self, df: pd.DataFrame, strategy, config: Dict) -> Optional[BacktestResult]:
        if not AKQUANT_AVAILABLE:
            return self._fallback_backtest(df, strategy, config)
        
        try:
            result = aq.run_backtest(
                data=self._prepare_data(df),
                strategy=self._wrap_strategy(strategy),
                initial_cash=config.get('initial_capital', 100000),
                symbol=config.get('symbol', 'stock')
            )
            return self._convert_result(result)
        except Exception as e:
            import warnings
            warnings.warn(f"AKQuant backtest failed: {e}, using fallback")
            return self._fallback_backtest(df, strategy, config)
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        if 'trade_date' in result.columns:
            result = result.rename(columns={'trade_date': 'date'})
        return result
    
    def _wrap_strategy(self, strategy):
        if not AKQUANT_AVAILABLE:
            return None
        pixiu_strategy = strategy
        class WrappedStrategy(Strategy):
            def on_bar(self, bar):
                df = pd.DataFrame({'close': [bar.close], 'open': [bar.open], 'high': [bar.high], 'low': [bar.low], 'volume': [bar.volume]})
                signal = pixiu_strategy.generate_signals(df).iloc[-1]
                if signal == 1:
                    pos_size = int(self.cash * 0.95 / bar.close)
                    if pos_size > 0:
                        self.buy(symbol=bar.symbol, quantity=pos_size)
                elif signal == -1:
                    pos = self.get_position(bar.symbol)
                    if pos > 0:
                        self.close_position(symbol=bar.symbol)
        return WrappedStrategy
    
    def _convert_result(self, aq_result) -> BacktestResult:
        return BacktestResult(
            total_return=getattr(aq_result, 'total_return_pct', 0),
            annualized_return=getattr(aq_result, 'annualized_return', 0),
            max_drawdown=getattr(aq_result, 'max_drawdown_pct', 0),
            sharpe_ratio=getattr(aq_result, 'sharpe_ratio', 0),
            win_rate=(getattr(aq_result, 'win_rate', 0) or 0) / 100,
            profit_loss_ratio=getattr(aq_result, 'profit_factor', 0),
            calmar_ratio=getattr(aq_result, 'calmar_ratio', 0),
            total_trades=int(getattr(aq_result, 'trade_count', 0) or 0),
            start_date=str(getattr(aq_result, 'start_time', '')),
            end_date=str(getattr(aq_result, 'end_time', '')),
            trades=[], equity_curve=[], drawdown_curve=[]
        )
    
    def _fallback_backtest(self, df: pd.DataFrame, strategy, config: Dict) -> BacktestResult:
        from pixiu.services.backtest_service import BacktestEngine, BacktestConfig
        backtest_config = BacktestConfig(
            initial_capital=config.get('initial_capital', 100000),
            commission_rate=config.get('commission_rate', 0.0003),
            position_size=0.95
        )
        signals = strategy.generate_signals(df)
        if isinstance(signals, pd.DataFrame):
            signals = signals['signal']
        engine = BacktestEngine(backtest_config)
        return engine.run(df, signals)
