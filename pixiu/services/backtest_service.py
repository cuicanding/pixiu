"""回测引擎模块"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

from pixiu.models.backtest import BacktestResult, Trade
from pixiu.config import config


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 100000.0
    commission_rate: float = 0.0003
    slippage_rate: float = 0.0001
    position_size: float = 0.95
    risk_free_rate: float = 0.03


class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, backtest_config: Optional[BacktestConfig] = None):
        self.config = backtest_config or BacktestConfig(
            initial_capital=config.initial_capital,
            commission_rate=config.commission_rate,
            position_size=0.95,
            risk_free_rate=config.risk_free_rate
        )
    
    def run(self, df: pd.DataFrame, signals: Optional[pd.Series] = None) -> BacktestResult:
        """执行回测
        
        Args:
            df: 行情数据，必须包含 close, signal 列
            signals: 可选的信号序列，如果提供将覆盖df中的signal列
        """
        if signals is not None:
            df = df.copy()
            df['signal'] = signals
        
        cash = self.config.initial_capital
        shares = 0
        equity_curve: List[float] = []
        trades: List[Trade] = []
        
        for i, row in df.iterrows():
            signal = row['signal']
            price = row['close']
            trade_date = str(row.get('trade_date', i))
            
            if signal == 1 and cash > 0:
                shares_to_buy = int(cash * self.config.position_size / price)
                if shares_to_buy > 0:
                    actual_price = price * (1 + self.config.slippage_rate)
                    commission = shares_to_buy * actual_price * self.config.commission_rate
                    cost = shares_to_buy * actual_price + commission
                    
                    if cost <= cash:
                        cash -= cost
                        shares += shares_to_buy
                        trades.append(Trade(
                            trade_date=trade_date,
                            signal_type="BUY",
                            shares=shares_to_buy,
                            price=actual_price,
                            amount=shares_to_buy * actual_price,
                            commission=commission
                        ))
            
            elif signal == -1 and shares > 0:
                actual_price = price * (1 - self.config.slippage_rate)
                commission = shares * actual_price * self.config.commission_rate
                proceeds = shares * actual_price - commission
                
                cash += proceeds
                trades.append(Trade(
                    trade_date=trade_date,
                    signal_type="SELL",
                    shares=shares,
                    price=actual_price,
                    amount=shares * actual_price,
                    commission=commission
                ))
                shares = 0
            
            equity_curve.append(cash + shares * price)
        
        return self._calculate_metrics(df, trades, equity_curve)
    
    def _calculate_metrics(
        self,
        df: pd.DataFrame,
        trades: List[Trade],
        equity_curve: List[float]
    ) -> BacktestResult:
        """计算回测指标"""
        equity = np.array(equity_curve)
        
        total_return = (equity[-1] - equity[0]) / equity[0]
        
        days = len(equity)
        annualized_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_drawdown = drawdown.min()
        
        returns = np.diff(equity) / equity[:-1]
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = (np.mean(returns) * 252 - self.config.risk_free_rate) / (np.std(returns) * np.sqrt(252))
        else:
            sharpe_ratio = 0.0
        
        if len(trades) >= 2:
            trade_returns = []
            for i in range(0, len(trades) - 1, 2):
                if i + 1 < len(trades):
                    buy_trade = trades[i]
                    sell_trade = trades[i + 1]
                    if buy_trade.signal_type == "BUY" and sell_trade.signal_type == "SELL":
                        trade_return = (sell_trade.price - buy_trade.price) / buy_trade.price
                        trade_returns.append(trade_return)
            
            if trade_returns:
                win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
                wins = [r for r in trade_returns if r > 0]
                losses = [r for r in trade_returns if r < 0]
                profit_loss_ratio = np.mean(wins) / abs(np.mean(losses)) if losses else 0.0
            else:
                win_rate = 0.0
                profit_loss_ratio = 0.0
        else:
            win_rate = 0.0
            profit_loss_ratio = 0.0
        
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            profit_loss_ratio=profit_loss_ratio,
            calmar_ratio=calmar_ratio,
            total_trades=len(trades),
            start_date=str(df.iloc[0].get('trade_date', '')),
            end_date=str(df.iloc[-1].get('trade_date', '')),
            trades=trades,
            equity_curve=equity.tolist(),
            drawdown_curve=drawdown.tolist()
        )
