"""RSI策略测试"""
import pytest
import pandas as pd
import numpy as np
from pixiu.strategies.classic.rsi import RSIStrategy


class TestRSIStrategy:
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        close = 10 + np.cumsum(np.random.randn(100) * 0.5)
        return pd.DataFrame({
            'trade_date': dates,
            'open': close + np.random.randn(100) * 0.1,
            'high': close + np.abs(np.random.randn(100) * 0.2),
            'low': close - np.abs(np.random.randn(100) * 0.2),
            'close': close,
            'volume': np.random.randint(1000000, 10000000, 100)
        })
    
    def test_init(self):
        strategy = RSIStrategy()
        assert strategy.name == "RSI策略"
        assert strategy.regime == "range"
    
    def test_generate_signals(self, sample_data):
        strategy = RSIStrategy()
        result = strategy.generate_signals(sample_data)
        assert len(result) == len(sample_data)
        assert set(result['signal'].unique()).issubset({-1, 0, 1})
    
    def test_signals_with_params(self, sample_data):
        strategy = RSIStrategy(oversold=25, overbought=75)
        result = strategy.generate_signals(sample_data)
        assert len(result) == len(sample_data)
        assert 'signal' in result.columns
        assert 'rsi' in result.columns
