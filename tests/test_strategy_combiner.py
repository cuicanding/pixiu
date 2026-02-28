"""策略组合器测试"""
import pytest
import pandas as pd
import numpy as np
from pixiu.strategies.combiner import StrategyCombiner
from pixiu.strategies.classic.rsi import RSIStrategy
from pixiu.strategies.classic.ma_cross import MACrossStrategy


class TestStrategyCombiner:
    
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
        combiner = StrategyCombiner()
        assert combiner is not None
    
    def test_equal_weight(self, sample_data):
        combiner = StrategyCombiner()
        rsi = RSIStrategy()
        ma = MACrossStrategy()
        
        signals = [
            rsi.generate_signals(sample_data),
            ma.generate_signals(sample_data)
        ]
        
        combined = combiner.equal_weight(signals)
        assert len(combined) == len(sample_data)
        assert set(combined.unique()).issubset({-1, 0, 1})
    
    def test_signal_filter(self, sample_data):
        combiner = StrategyCombiner()
        rsi = RSIStrategy()
        ma = MACrossStrategy()
        
        signals = [
            rsi.generate_signals(sample_data),
            ma.generate_signals(sample_data)
        ]
        
        filtered = combiner.signal_filter(signals, threshold=2)
        assert len(filtered) == len(sample_data)
    
    def test_complementary(self, sample_data):
        combiner = StrategyCombiner()
        rsi = RSIStrategy()
        ma = MACrossStrategy()
        
        combined = combiner.complementary(
            sample_data, "trend",
            [ma], [rsi]
        )
        assert len(combined) == len(sample_data)
