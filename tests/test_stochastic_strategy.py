"""随机过程策略测试"""
import pytest
import pandas as pd
import numpy as np
from pixiu.strategies.advanced.stochastic import StochasticStrategy


class TestStochasticStrategy:
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        close = 10 + np.cumsum(np.random.randn(100) * 0.5)
        return pd.DataFrame({
            'trade_date': dates, 'close': close,
            'high': close + 0.5, 'low': close - 0.5, 'open': close, 'volume': 1000000
        })
    
    def test_init(self):
        strategy = StochasticStrategy()
        assert strategy.name == "随机过程策略"
        assert strategy.regime == "any"
    
    def test_generate_signals(self, sample_data):
        strategy = StochasticStrategy()
        signals = strategy.generate_signals(sample_data)
        assert len(signals) == len(sample_data)
