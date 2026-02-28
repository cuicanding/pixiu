"""网格交易策略"""
import pandas as pd
from pixiu.strategies.base import BaseStrategy
from pixiu.strategies import register_strategy


@register_strategy
class GridTradingStrategy(BaseStrategy):
    """网格交易策略
    
    适用于震荡行情：
    - 在价格下跌grid_size时买入
    - 在价格上涨grid_size时卖出
    """
    
    name = "网格交易策略"
    description = "在价格区间内设置网格，低买高卖的均值回归策略"
    regime = "range"
    params = {
        "grid_size": 0.02,
        "grid_count": 10
    }
    
    def __init__(self, grid_size: float = 0.02, grid_count: int = 10):
        self.params = {
            "grid_size": grid_size,
            "grid_count": grid_count
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        grid_size = self.params["grid_size"]
        
        signals = pd.Series(0, index=df.index)
        
        base_price = close.iloc[0]
        position = 0
        last_trade_price = base_price
        
        for i in range(1, len(close)):
            current_price = close.iloc[i]
            price_change = (current_price - last_trade_price) / last_trade_price
            
            if position == 0 and price_change <= -grid_size:
                signals.iloc[i] = 1
                position = 1
                last_trade_price = current_price
            elif position > 0 and price_change >= grid_size:
                signals.iloc[i] = -1
                position = 0
                last_trade_price = current_price
        
        return signals
    
    def get_required_data(self) -> list:
        return ["close"]
