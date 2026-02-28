"""测试时间线择势分析"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pixiu.analysis.regime_timeline import RegimeTimelineAnalyzer

def generate_test_data(days: int = 180, regime: str = "trend") -> pd.DataFrame:
    """生成测试数据"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    if regime == "trend":
        close = 100 * (1 + np.linspace(0, 0.3, days))
    else:
        close = 100 + np.sin(np.linspace(0, 10*np.pi, days)) * 10
    
    return pd.DataFrame({
        'trade_date': dates,
        'open': close * 0.99,
        'high': close * 1.02,
        'low': close * 0.98,
        'close': close,
        'volume': np.random.randint(1000000, 10000000, days),
    })

def test_timeline_analyzer_init():
    """测试初始化"""
    analyzer = RegimeTimelineAnalyzer(window=60)
    assert analyzer.window == 60
    assert analyzer.adx_threshold == 25.0

def test_analyze_trend_data():
    """测试分析趋势数据"""
    analyzer = RegimeTimelineAnalyzer(window=30)
    df = generate_test_data(90, "trend")
    
    result = analyzer.analyze_timeline(df)
    
    assert 'segments' in result
    assert 'turning_points' in result
    assert len(result['segments']) > 0

def test_analyze_range_data():
    """测试分析震荡数据"""
    analyzer = RegimeTimelineAnalyzer(window=30)
    df = generate_test_data(90, "range")
    
    result = analyzer.analyze_timeline(df)
    
    assert 'segments' in result
    assert len(result['segments']) > 0

def test_empty_data():
    """测试空数据处理"""
    analyzer = RegimeTimelineAnalyzer(window=30)
    df = pd.DataFrame()
    
    result = analyzer.analyze_timeline(df)
    
    assert result['segments'] == []
    assert result['turning_points'] == []
