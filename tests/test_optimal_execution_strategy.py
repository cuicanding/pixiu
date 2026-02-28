"""最优执行策略测试"""
import pytest
import pandas as pd
import numpy as np
from pixiu.strategies.advanced.optimal_execution import OptimalExecutionStrategy


class TestOptimalExecutionStrategy:
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        close = 10 + np.cumsum(np.random.randn(100) * 0.5 + 0.05)
        return pd.DataFrame({
            'trade_date': dates, 'close': close, 'volume': np.random.randint(1000000, 10000000, 100),
            'high': close + 0.5, 'low': close - 0.5, 'open': close
        })
    
    def test_init(self):
        strategy = OptimalExecutionStrategy()
        assert strategy.name == "最优执行策略"
        assert strategy.regime == "trend"
    
    def test_generate_signals(self, sample_data):
        strategy = OptimalExecutionStrategy()
        signals = strategy.generate_signals(sample_data)
        assert len(signals) == len(sample_data)
