"""网格交易策略"""
import pandas as pd
from pixiu.strategies.base import BaseStrategy
from pixiu.strategies import register_strategy


@register_strategy
class GridTradingStrategy(BaseStrategy):
    """网格交易策略 - 在价格区间内高抛低吸"""
    
    name = "网格交易策略"
    description = "在价格区间内设置网格，低买高卖的均值回归策略"
    params = {
        "grid_size": 0.05,
        "grid_count": 10
    }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        grid_size = self.params.get("grid_size", 0.05)
        
        df['signal'] = 0
        
        position = 0
        last_trade_price = df['close'].iloc[0] if len(df) > 0 else 0
        
        for i in range(1, len(df)):
            current_price = df['close'].iloc[i]
            price_change = (current_price - last_trade_price) / last_trade_price if last_trade_price > 0 else 0
            
            if position == 0 and price_change <= -grid_size:
                df.iloc[i, df.columns.get_loc('signal')] = 1
                position = 1
                last_trade_price = current_price
            elif position > 0 and price_change >= grid_size:
                df.iloc[i, df.columns.get_loc('signal')] = -1
                position = 0
                last_trade_price = current_price
        
        return df
    
    def get_required_data(self) -> list:
        return ["close"]
    
    def get_documentation(self) -> str:
        return """
## 网格交易策略

### 原理

网格交易是一种经典的量化策略，在设定的价格区间内：
- 价格下跌一定幅度 → 买入
- 价格上涨一定幅度 → 卖出

### 参数

- grid_size: 网格大小，默认5%，即价格波动5%触发交易

### 适用场景

- 震荡行情
- 区间波动明显的股票
"""
