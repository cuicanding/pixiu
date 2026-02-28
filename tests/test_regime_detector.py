"""择势判断模块测试"""
import pytest
import pandas as pd
import numpy as np
from pixiu.analysis.regime_detector import MarketRegimeDetector


class TestMarketRegimeDetector:
    
    @pytest.fixture
    def trend_data(self):
        """生成趋势行情数据"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        close = 10 + np.cumsum(np.random.randn(100) * 0.5 + 0.1)
        return pd.DataFrame({
            'trade_date': dates,
            'open': close + np.random.randn(100) * 0.1,
            'high': close + np.abs(np.random.randn(100) * 0.2),
            'low': close - np.abs(np.random.randn(100) * 0.2),
            'close': close,
            'volume': np.random.randint(1000000, 10000000, 100)
        })
    
    @pytest.fixture
    def range_data(self):
        """生成震荡行情数据"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        close = 10 + np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.randn(100) * 0.1
        return pd.DataFrame({
            'trade_date': dates,
            'open': close + np.random.randn(100) * 0.05,
            'high': close + np.abs(np.random.randn(100) * 0.1),
            'low': close - np.abs(np.random.randn(100) * 0.1),
            'close': close,
            'volume': np.random.randint(1000000, 10000000, 100)
        })
    
    def test_init(self):
        """测试初始化"""
        detector = MarketRegimeDetector()
        assert detector is not None
    
    def test_calc_adx(self, trend_data):
        """测试ADX计算"""
        detector = MarketRegimeDetector()
        adx = detector._calc_adx(trend_data)
        assert isinstance(adx, float)
        assert 0 <= adx <= 100
    
    def test_calc_ma_slope(self, trend_data):
        """测试MA斜率计算"""
        detector = MarketRegimeDetector()
        slope = detector._calc_ma_slope(trend_data)
        assert isinstance(slope, float)
    
    def test_calc_volatility(self, trend_data):
        """测试波动率计算"""
        detector = MarketRegimeDetector()
        vol = detector._calc_volatility(trend_data)
        assert isinstance(vol, float)
        assert vol >= 0
    
    def test_detect_regime_trend(self, trend_data):
        """测试趋势识别"""
        detector = MarketRegimeDetector()
        regime = detector.detect_regime(trend_data)
        assert regime == 'trend'
    
    def test_detect_regime_range(self, range_data):
        """测试震荡识别"""
        detector = MarketRegimeDetector()
        regime = detector.detect_regime(range_data)
        assert regime == 'range'
    
    def test_get_analysis_detail(self, trend_data):
        """测试详细分析"""
        detector = MarketRegimeDetector()
        detail = detector.get_analysis_detail(trend_data)
        assert 'regime' in detail
        assert 'adx' in detail
        assert 'ma_slope' in detail
        assert 'volatility' in detail
    
    def test_missing_columns_raises_error(self):
        """测试缺少必需列时抛出异常"""
        detector = MarketRegimeDetector()
        df = pd.DataFrame({'a': [1, 2, 3]})
        with pytest.raises(ValueError, match="缺少必需的列"):
            detector.detect_regime(df)
    
    def test_insufficient_data_returns_range(self):
        """测试数据不足时返回range"""
        detector = MarketRegimeDetector()
        df = pd.DataFrame({
            'open': [10, 11],
            'high': [11, 12],
            'low': [9, 10],
            'close': [10.5, 11.5]
        })
        regime = detector.detect_regime(df)
        assert regime == 'range'
