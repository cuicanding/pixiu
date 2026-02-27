"""回测结果模型"""

from dataclasses import dataclass,from typing import List


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
    trades: List[Trade] = None
    equity_curve: List[float] = None
    drawdown_curve: List[float] = None
    
    def __post_init__(self):
        if self.trades is None:
            self.trades = []
        if self.equity_curve is None:
            self.equity_curve = []
        if self.drawdown_curve is None:
            self.drawdown_curve = []
