"""随机过程策略"""
import pandas as pd
import numpy as np
from pixiu.strategies.base import BaseStrategy
from pixiu.strategies import register_strategy


@register_strategy
class StochasticStrategy(BaseStrategy):
    """随机过程策略
    
    基于几何布朗运动(GBM)建模：
    dS = μS dt + σS dW
    
    利用估计的漂移项和波动率预测价格偏离
    """
    
    name = "随机过程策略"
    description = "基于几何布朗运动的随机过程建模策略"
    regime = "any"
    params = {
        "lookback": 60,
        "z_threshold": 1.5
    }
    
    def __init__(self, lookback: int = 60, z_threshold: float = 1.5):
        self.params = {
            "lookback": lookback,
            "z_threshold": z_threshold
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        lookback = self.params["lookback"]
        z_threshold = self.params["z_threshold"]
        
        signals = pd.Series(0, index=df.index)
        
        for i in range(lookback, len(close)):
            window = close.iloc[i-lookback:i]
            returns = window.pct_change().dropna()
            
            mu = returns.mean() * 252
            sigma = returns.std() * np.sqrt(252)
            
            expected_return = mu * (1/252)
            actual_return = (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]
            
            z_score = (actual_return - expected_return) / (sigma / np.sqrt(252))
            
            if z_score < -z_threshold:
                signals.iloc[i] = 1
            elif z_score > z_threshold:
                signals.iloc[i] = -1
        
        return signals
    
    def get_required_data(self) -> list:
        return ["close"]
