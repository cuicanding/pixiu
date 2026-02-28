"""网格交易策略测试"""
import pytest
import pandas as pd
import numpy as np
from pixiu.strategies.classic.grid_trading import GridTradingStrategy


class TestGridTradingStrategy:
    
    @pytest.fixture
    def range_data(self):
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        close = 10 + np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.randn(100) * 0.1
        return pd.DataFrame({
            'trade_date': dates,
            'close': close,
            'high': close + 0.5,
            'low': close - 0.5,
            'open': close,
            'volume': 1000000
        })
    
    def test_init(self):
        strategy = GridTradingStrategy()
        assert strategy.name == "网格交易策略"
        assert strategy.regime == "range"
    
    def test_generate_signals(self, range_data):
        strategy = GridTradingStrategy()
        signals = strategy.generate_signals(range_data)
        assert len(signals) == len(range_data)
        assert set(signals.unique()).issubset({-1, 0, 1})
