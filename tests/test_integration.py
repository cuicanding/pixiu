"""集成测试"""
import pytest
import pandas as pd
import numpy as np
from pixiu.analysis import MarketRegimeDetector
from pixiu.strategies.classic import RSIStrategy, MACrossStrategy, GridTradingStrategy
from pixiu.strategies.advanced import StochasticStrategy
from pixiu.strategies.combiner import StrategyCombiner
from pixiu.services.backtest_service import BacktestEngine, BacktestConfig


class TestIntegration:
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=200, freq='D')
        close = 10 + np.cumsum(np.random.randn(200) * 0.5)
        return pd.DataFrame({
            'trade_date': dates,
            'open': close + np.random.randn(200) * 0.1,
            'high': close + np.abs(np.random.randn(200) * 0.2),
            'low': close - np.abs(np.random.randn(200) * 0.2),
            'close': close,
            'volume': np.random.randint(1000000, 10000000, 200)
        })
    
    def test_full_workflow(self, sample_data):
        """测试完整工作流"""
        # 1. 择势判断
        detector = MarketRegimeDetector()
        regime = detector.detect_regime(sample_data)
        assert regime in ['trend', 'range']
        
        # 2. 选择策略
        if regime == 'trend':
            strategies = [MACrossStrategy()]
        else:
            strategies = [RSIStrategy(), GridTradingStrategy()]
        
        # 3. 生成信号
        signals = [s.generate_signals(sample_data) for s in strategies]
        
        # 4. 组合信号
        combiner = StrategyCombiner()
        combined = combiner.equal_weight(signals)
        
        # 5. 回测
        config = BacktestConfig(initial_capital=100000)
        engine = BacktestEngine(config)
        result = engine.run(sample_data, combined)
        
        # 6. 验证结果
        assert result.total_return is not None
        assert result.max_drawdown is not None
        assert result.sharpe_ratio is not None
    
    def test_strategy_combiner_all_modes(self, sample_data):
        """测试所有组合模式"""
        rsi = RSIStrategy()
        ma = MACrossStrategy()
        signals = [rsi.generate_signals(sample_data), ma.generate_signals(sample_data)]
        
        combiner = StrategyCombiner()
        
        # 等权组合
        eq = combiner.equal_weight(signals)
        assert len(eq) == len(sample_data)
        
        # 信号过滤
        sf = combiner.signal_filter(signals, threshold=1)
        assert len(sf) == len(sample_data)
        
        # 互补策略
        cp = combiner.complementary(sample_data, 'range', [ma], [rsi])
        assert len(cp) == len(sample_data)
    
    def test_regime_analysis_detail(self, sample_data):
        """测试择势分析详情"""
        detector = MarketRegimeDetector()
        detail = detector.get_analysis_detail(sample_data)
        
        assert 'regime' in detail
        assert 'adx' in detail
        assert 'ma_slope' in detail
        assert 'volatility' in detail
        assert detail['regime'] in ['trend', 'range']
