"""策略模块"""

from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseStrategy

STRATEGY_REGISTRY: Dict[str, "BaseStrategy"] = {}


def register_strategy(cls):
    """装饰器：自动注册策略"""
    instance = cls()
    STRATEGY_REGISTRY[instance.name] = instance
    return cls


def get_all_strategies() -> List["BaseStrategy"]:
    """获取所有已注册策略"""
    return list(STRATEGY_REGISTRY.values())


def get_strategy(name: str) -> "BaseStrategy | None":
    """按名称获取策略"""
    return STRATEGY_REGISTRY.get(name)


from .base import BaseStrategy

# 自动导入并注册所有策略
from .classic.rsi import RSIStrategy
from .classic.ma_cross import MACrossStrategy
from .classic.grid_trading import GridTradingStrategy
from .advanced.stochastic import StochasticStrategy
from .advanced.optimal_execution import OptimalExecutionStrategy
from .trend_strength import TrendStrengthStrategy
from .volatility import VolatilityStrategy
from .kalman_filter import KalmanFilterStrategy
