"""回测结果模型"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Trade:
    """交易记录"""
    trade_date: str
    signal_type: str
    shares: int
    price: float
    amount: float = 0.0
    commission: float = 0.0


@dataclass
class BacktestResult:
    """回测结果"""
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_loss_ratio: float
    calmar_ratio: float
    total_trades: int
    start_date: str = ""
    end_date: str = ""
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
