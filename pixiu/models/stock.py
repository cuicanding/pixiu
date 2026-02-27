"""股票数据模型"""

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class Stock:
    """股票基础信息"""
    code: str
    name: str
    market: str
    industry: Optional[str] = None
    list_date: Optional[date] = None
    updated_at: Optional[str] = None


@dataclass  
class DailyQuote:
    """日线行情数据"""
    code: str
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float = 0.0
    turnover_rate: Optional[float] = None
