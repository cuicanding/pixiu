"""RSI策略"""
import pandas as pd
import numpy as np
from typing import List

from pixiu.strategies.base import BaseStrategy
from pixiu.strategies import register_strategy


@register_strategy
class RSIStrategy(BaseStrategy):
    """RSI相对强弱指标策略
    
    适用于震荡行情：
    - RSI < oversold (30): 超卖，买入信号
    - RSI > overbought (70): 超买，卖出信号
    """
    
    name = "RSI策略"
    description = "基于RSI相对强弱指标的均值回归策略，适用于震荡行情"
    regime = "range"
    params = {
        "period": 14,
        "oversold": 30,
        "overbought": 70
    }
    
    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        self.params = {
            "period": period,
            "oversold": oversold,
            "overbought": overbought
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号
        
        Args:
            df: 包含close列的DataFrame
            
        Returns:
            添加signal列的DataFrame
            signal: 1=买入, -1=卖出, 0=持有
        """
        df = df.copy()
        close = df['close']
        period = self.params["period"]
        oversold = self.params["oversold"]
        overbought = self.params["overbought"]
        
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        df['rsi'] = rsi
        df['signal'] = 0
        df.loc[rsi < oversold, 'signal'] = 1
        df.loc[rsi > overbought, 'signal'] = -1
        
        return df
    
    def get_required_data(self) -> List[str]:
        return ["close"]
