"""最优执行策略"""
import pandas as pd
import numpy as np
from pixiu.strategies.base import BaseStrategy
from pixiu.strategies import register_strategy


@register_strategy
class OptimalExecutionStrategy(BaseStrategy):
    """最优执行策略
    
    基于TWAP/VWAP执行算法：
    - 在趋势行情中分批建仓
    - 使用成交量加权平均价格作为执行基准
    """
    
    name = "最优执行策略"
    description = "基于TWAP/VWAP的最优执行算法策略"
    regime = "trend"
    params = {
        "execution_window": 5,
        "volume_threshold": 1.2
    }
    
    def __init__(self, execution_window: int = 5, volume_threshold: float = 1.2):
        self.params = {
            "execution_window": execution_window,
            "volume_threshold": volume_threshold
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        volume = df['volume']
        execution_window = self.params["execution_window"]
        volume_threshold = self.params["volume_threshold"]
        
        signals = pd.Series(0, index=df.index)
        
        avg_volume = volume.rolling(execution_window * 5).mean()
        vwap = (close * volume).rolling(execution_window * 5).sum() / \
               volume.rolling(execution_window * 5).sum()
        
        for i in range(execution_window * 5, len(close)):
            price_vs_vwap = (close.iloc[i] - vwap.iloc[i]) / vwap.iloc[i]
            vol_ratio = volume.iloc[i] / avg_volume.iloc[i]
            
            if price_vs_vwap < -0.01 and vol_ratio > volume_threshold:
                signals.iloc[i] = 1
            elif price_vs_vwap > 0.01 and vol_ratio > volume_threshold:
                signals.iloc[i] = -1
        
        return signals
    
    def get_required_data(self) -> list:
        return ["close", "volume"]
