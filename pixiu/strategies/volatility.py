"""波动率套利策略"""

import numpy as np
import pandas as pd
from scipy import integrate
from typing import List

from .base import BaseStrategy
from . import register_strategy


@register_strategy
class VolatilityStrategy(BaseStrategy):
    """基于波动率积分的均值回归策略
    
    数学原理：
    - 波动率 σ：价格标准差
    - 波动率积分 ∫σ dt：累积波动能量
    - 高累积波动后往往出现回归
    
    信号生成：
    - 波动率积分超过阈值 → 预期回归 → 反向操作
    """
    
    name = "波动率套利策略"
    description = "基于波动率积分分析，捕捉均值回归机会"
    params = {
        "window": 20,
        "entry_threshold": 2.0,
        "exit_threshold": 0.5
    }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        window = self.params.get("window", 20)
        entry_threshold = self.params.get("entry_threshold", 2.0)
        
        returns = df['close'].pct_change()
        df['volatility'] = returns.rolling(window=window).std() * np.sqrt(252)
        
        volatility_values = df['volatility'].fillna(0).values
        df['volatility_integral'] = 0.0
        
        for i in range(1, len(df)):
            x = np.arange(i)
            y = volatility_values[:i]
            mask = ~np.isnan(y)
            if mask.sum() > 1:
                df.loc[df.index[i], 'volatility_integral'] = integrate.trapezoid(y[mask], x[mask])
        
        vol_integral_normalized = df['volatility_integral'] / df['volatility_integral'].rolling(window*2).mean()
        
        df['signal'] = 0
        df.loc[vol_integral_normalized > entry_threshold, 'signal'] = -1
        df.loc[vol_integral_normalized < -entry_threshold, 'signal'] = 1
        
        df.loc[:window*2, 'signal'] = 0
        
        return df
    
    def get_required_data(self) -> List[str]:
        return ['close']
    
    def get_documentation(self) -> str:
        return """
## 波动率套利策略

### 数学原理

**波动率计算**
$$\\sigma_t = \\sqrt{\\frac{1}{N}\\sum_{i=0}^{N-1}(r_{t-i} - \\bar{r})^2}$$

**波动率积分（累积波动能量）**
$$E(t) = \\int_0^t \\sigma(\\tau) d\\tau$$

### 交易逻辑

高波动率累积后，市场往往出现均值回归：

- 积分值异常高 → 市场过度波动 → 预期回归 → 卖出
- 积分值异常低 → 市场过度平静 → 预期突破 → 买入
"""
