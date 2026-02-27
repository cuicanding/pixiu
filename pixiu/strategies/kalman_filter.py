"""卡尔曼滤波策略"""

import numpy as np
import pandas as pd
from typing import List

from .base import BaseStrategy
from . import register_strategy


@register_strategy
class KalmanFilterStrategy(BaseStrategy):
    """基于卡尔曼滤波的价格估计策略
    
    数学原理：
    状态空间模型：
    - 状态方程：x_t = A*x_{t-1} + w_t
    - 观测方程：y_t = H*x_t + v_t
    
    卡尔曼滤波递归估计真实价格，过滤噪声
    
    信号生成：
    - 观测价格 > 估计价格 + kσ → 卖出（价格偏高）
    - 观测价格 < 估计价格 - kσ → 买入（价格偏低）
    """
    
    name = "卡尔曼滤波策略"
    description = "使用卡尔曼滤波估计真实价格，捕捉价格偏离"
    params = {
        "process_variance": 1e-5,
        "measurement_variance": 1e-2,
        "signal_threshold": 1.5
    }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        Q = self.params.get("process_variance", 1e-5)
        R = self.params.get("measurement_variance", 1e-2)
        threshold = self.params.get("signal_threshold", 1.5)
        
        n = len(df)
        x = np.zeros(n)
        P = np.zeros(n)
        
        x[0] = df['close'].iloc[0]
        P[0] = 1.0
        
        for i in range(1, n):
            x_pred = x[i-1]
            P_pred = P[i-1] + Q
            
            K = P_pred / (P_pred + R)
            
            x[i] = x_pred + K * (df['close'].iloc[i] - x_pred)
            P[i] = (1 - K) * P_pred
        
        df['kalman_estimate'] = x
        df['kalman_variance'] = P
        df['kalman_std'] = np.sqrt(P)
        
        deviation = (df['close'] - df['kalman_estimate']) / df['kalman_std']
        
        df['signal'] = 0
        df.loc[deviation < -threshold, 'signal'] = 1
        df.loc[deviation > threshold, 'signal'] = -1
        
        return df
    
    def get_required_data(self) -> List[str]:
        return ['close']
    
    def get_documentation(self) -> str:
        return """
## 卡尔曼滤波策略

### 数学原理

卡尔曼滤波是一种最优递归估计算法，基于状态空间模型：

**预测步骤**
$$\\hat{x}_{t|t-1} = A\\hat{x}_{t-1}$$
$$P_{t|t-1} = AP_{t-1}A^T + Q$$

**更新步骤**
$$K_t = P_{t|t-1}H^T(HP_{t|t-1}H^T + R)^{-1}$$
$$\\hat{x}_t = \\hat{x}_{t|t-1} + K_t(y_t - H\\hat{x}_{t|t-1})$$

### 交易逻辑

- 观测价格显著低于滤波估计 → 价格被低估 → 买入
- 观测价格显著高于滤波估计 → 价格被高估 → 卖出
"""
