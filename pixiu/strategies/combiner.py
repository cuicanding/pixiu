"""策略组合器"""
from typing import List, Dict
import pandas as pd


class StrategyCombiner:
    """策略组合器
    
    提供三种组合模式：
    1. equal_weight: 等权组合
    2. signal_filter: 信号过滤
    3. complementary: 互补策略
    """
    
    COMBINE_MODES = ["equal_weight", "signal_filter", "complementary"]
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            "mode": "complementary",
            "filter_threshold": 2,
            "trend_strategies": ["均线交叉策略"],
            "range_strategies": ["RSI策略", "网格交易策略"]
        }
    
    def _extract_signal(self, signal) -> pd.Series:
        """从信号中提取Series，支持DataFrame和Series"""
        if isinstance(signal, pd.DataFrame):
            return signal['signal']
        return signal
    
    def equal_weight(self, signals: List[pd.Series]) -> pd.Series:
        """等权组合：所有策略信号取平均，>0买入，<0卖出"""
        if not signals:
            return pd.Series(0, index=[])
        
        extracted = [self._extract_signal(s) for s in signals]
        combined = sum(extracted) / len(extracted)
        result = pd.Series(0, index=extracted[0].index)
        result[combined > 0] = 1
        result[combined < 0] = -1
        return result
    
    def signal_filter(self, signals: List[pd.Series], threshold: int = 2) -> pd.Series:
        """信号过滤：N个以上策略一致时才执行"""
        if not signals:
            return pd.Series(0, index=[])
        
        extracted = [self._extract_signal(s) for s in signals]
        buy_votes = sum((s == 1).astype(int) for s in extracted)
        sell_votes = sum((s == -1).astype(int) for s in extracted)
        
        result = pd.Series(0, index=extracted[0].index)
        result[buy_votes >= threshold] = 1
        result[sell_votes >= threshold] = -1
        return result
    
    def complementary(
        self,
        df: pd.DataFrame,
        regime: str,
        trend_strategies: List,
        range_strategies: List
    ) -> pd.Series:
        """互补策略：根据市场状态自动切换策略组"""
        if regime == "trend":
            strategies = trend_strategies
        else:
            strategies = range_strategies
        
        if not strategies:
            return pd.Series(0, index=df.index)
        
        signals = [s.generate_signals(df) for s in strategies]
        return self.equal_weight(signals)
    
    def combine(
        self,
        signals: List[pd.Series],
        regime: str = None,
        df: pd.DataFrame = None,
        trend_strategies: List = None,
        range_strategies: List = None
    ) -> pd.Series:
        """通用组合接口"""
        mode = self.config.get("mode", "equal_weight")
        
        if mode == "equal_weight":
            return self.equal_weight(signals)
        elif mode == "signal_filter":
            return self.signal_filter(signals, self.config.get("filter_threshold", 2))
        elif mode == "complementary":
            return self.complementary(
                df, regime, trend_strategies or [], range_strategies or []
            )
        
        raise ValueError(f"Unknown combine mode: {mode}")
