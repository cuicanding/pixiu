"""策略基类模块"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import pandas as pd


class BaseStrategy(ABC):
    """所有策略必须继承此类"""
    
    name: str = ""
    description: str = ""
    params: Dict[str, Any] = {}
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号
        
        Args:
            df: 行情数据， containing open, high, low, close, volume
            
        Returns:
            添加signal列的DataFrame
            signal: 1=买入, -1=卖出, 0=持有
        """
        pass
    
    @abstractmethod
    def get_required_data(self) -> List[str]:
        """返回需要的数据列"""
        pass
    
    def validate_params(self) -> bool:
        """参数校验"""
        return True
    
    def get_documentation(self) -> str:
        """返回策略的数学原理说明"""
        return f"## {self.name}\n\n{self.description}"
    
    def get_param_schema(self) -> Dict[str, Any]:
        """返回参数schema用于UI生成"""
        return {
            "type": "object",
            "properties": {
                k: {"type": "number", "default": v}
                for k, v in self.params.items()
            }
        }
