"""均线交叉策略"""
import pandas as pd
from pixiu.strategies.base import BaseStrategy
from pixiu.strategies import register_strategy


@register_strategy
class MACrossStrategy(BaseStrategy):
    """均线交叉策略
    
    适用于趋势行情：
    - 金叉（短期均线上穿长期均线）: 买入
    - 死叉（短期均线下穿长期均线）: 卖出
    """
    
    name = "均线交叉策略"
    description = "基于快慢均线交叉的趋势跟踪策略"
    regime = "trend"
    params = {
        "fast_period": 5,
        "slow_period": 20
    }
    
    def __init__(self, fast_period: int = 5, slow_period: int = 20):
        self.params = {
            "fast_period": fast_period,
            "slow_period": slow_period
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df['close']
        fast_period = self.params["fast_period"]
        slow_period = self.params["slow_period"]
        
        fast_ma = close.rolling(fast_period).mean()
        slow_ma = close.rolling(slow_period).mean()
        
        df['signal'] = 0
        
        gold_cross = (fast_ma.shift(1) <= slow_ma.shift(1)) & (fast_ma > slow_ma)
        death_cross = (fast_ma.shift(1) >= slow_ma.shift(1)) & (fast_ma < slow_ma)
        
        df.loc[gold_cross, 'signal'] = 1
        df.loc[death_cross, 'signal'] = -1
        
        return df
    
    def get_required_data(self) -> list:
        return ["close"]
