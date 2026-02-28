"""AKQuant适配器测试"""
import pytest
import pandas as pd
import numpy as np
from pixiu.services.akquant_adapter import AKQuantAdapter
from pixiu.strategies.classic.rsi import RSIStrategy


class TestAKQuantAdapter:
    
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
        adapter = AKQuantAdapter()
        assert adapter is not None
    
    def test_fallback_backtest(self, sample_data):
        adapter = AKQuantAdapter()
        strategy = RSIStrategy()
        config = {'initial_capital': 100000, 'symbol': 'test'}
        result = adapter.run_backtest(sample_data, strategy, config)
        assert result is not None
