"""趋势强度策略"""

import numpy as np
import pandas as pd
from typing import List

from .base import BaseStrategy
from . import register_strategy


@register_strategy
class TrendStrengthStrategy(BaseStrategy):
    """基于价格导数判断趋势强度
    
    数学原理：
    - 一阶导数 f'(t)：价格变化率，判断趋势方向
    - 二阶导数 f''(t)：变化加速度，判断趋势强度
    
    信号生成：
    - f'(t) > 0 且 f''(t) > 0 → 强买 (1)
    - f'(t) < 0 且 f''(t) < 0 → 强卖 (-1)
    - 其他 → 持有 (0)
    """
    
    name = "趋势强度策略"
    description = "基于微积分导数分析价格趋势方向和强度"
    params = {
        "window": 20,
        "strength_threshold": 0.02
    }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        window = self.params.get("window", 20)
        
        df['price_derivative'] = np.gradient(df['close'])
        df['price_acceleration'] = np.gradient(df['price_derivative'])
        
        rolling_std = df['close'].rolling(window=window).std()
        df['trend_strength'] = df['price_derivative'] / rolling_std
        
        df['signal'] = np.where(
            (df['price_derivative'] > 0) & (df['price_acceleration'] > 0),
            1,
            np.where(
                (df['price_derivative'] < 0) & (df['price_acceleration'] < 0),
                -1,
                0
            )
        )
        
        df.iloc[:window, df.columns.get_loc('signal')] = 0
        
        return df
    
    def get_required_data(self) -> List[str]:
        return ['close']
    
    def get_documentation(self) -> str:
        return """
## 趋势强度策略

### 数学原理

本策略基于微积分中的导数概念：

**一阶导数（价格变化率）**
$$f'(t) = \\frac{dP}{dt} \\approx \\frac{P(t) - P(t-1)}{\\Delta t}$$

- f'(t) > 0：价格上升
- f'(t) < 0：价格下降

**二阶导数（加速度）**
$$f''(t) = \\frac{d^2P}{dt^2}$$

- f''(t) > 0：上升趋势加强
- f''(t) < 0：下降趋势加强

### 交易规则

| 条件 | 信号 | 说明 |
|------|------|------|
| f'(t) > 0 且 f''(t) > 0 | 买入 | 趋势向上且加速 |
| f'(t) < 0 且 f''(t) < 0 | 卖出 | 趋势向下且加速 |
| 其他 | 持有 | 趋势不明确 |
"""
