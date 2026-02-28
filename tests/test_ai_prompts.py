"""测试AI三阶段Prompt"""
import pytest
from pixiu.services.ai_service import AIReportService

def test_regime_explanation_prompt():
    """测试择势解释Prompt生成"""
    timeline = {
        'segments': [
            {'start_date': '2025-01-01', 'end_date': '2025-03-15', 'regime': 'trend', 'confidence': 0.8}
        ],
        'turning_points': []
    }
    
    prompt = AIReportService._build_regime_prompt(
        stock_code="000001",
        stock_name="平安银行",
        timeline=timeline
    )
    
    assert "平安银行" in prompt
    assert "000001" in prompt

def test_strategy_recommendation_prompt():
    """测试策略推荐Prompt生成"""
    prompt = AIReportService._build_strategy_prompt(
        regime_summary="大盘趋势，个股震荡",
        strategies=["网格交易策略", "RSI策略"]
    )
    
    assert "网格交易" in prompt
    assert "大盘趋势" in prompt

def test_backtest_evaluation_prompt():
    """测试回测评估Prompt生成"""
    prompt = AIReportService._build_backtest_prompt(
        strategy="网格交易策略",
        results={"total_return": 15.5, "sharpe_ratio": 1.2, "max_drawdown": -8.3, "win_rate": 0.6}
    )
    
    assert "网格交易" in prompt
    assert "15.5" in prompt

def test_three_new_methods_exist():
    """测试三个新方法存在"""
    assert hasattr(AIReportService, '_build_regime_prompt')
    assert hasattr(AIReportService, '_build_strategy_prompt')
    assert hasattr(AIReportService, '_build_backtest_prompt')
    assert hasattr(AIReportService, 'explain_regime_timeline')
    assert hasattr(AIReportService, 'recommend_strategy')
    assert hasattr(AIReportService, 'evaluate_backtest')
