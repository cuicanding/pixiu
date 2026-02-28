import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pixiu.services.chart_service import generate_backtest_chart, generate_regime_chart

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'trade_date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'open': [10.0] * 100,
        'high': [10.5] * 100,
        'low': [9.5] * 100,
        'close': [10.0 + i * 0.01 for i in range(100)],
        'volume': [1000000] * 100,
    })

@pytest.fixture
def sample_trades():
    from pixiu.models.backtest import Trade
    return [
        Trade(trade_date='2024-01-10', signal_type='BUY', price=10.0, shares=1000, amount=10000.0, commission=3.0),
        Trade(trade_date='2024-01-20', signal_type='SELL', price=10.5, shares=1000, amount=10500.0, commission=3.15),
    ]

@patch('plotly.graph_objects.Figure.to_image')
def test_generate_backtest_chart_returns_base64(mock_to_image, sample_df, sample_trades):
    import base64
    fake_image = b'fake_image_bytes_' * 20
    mock_to_image.return_value = fake_image
    expected_base64 = base64.b64encode(fake_image).decode('utf-8')
    
    equity = [100000 + i * 10 for i in range(100)]
    drawdown = [0.0] * 50 + [-0.01 * (i-50) for i in range(50, 100)]
    
    result = generate_backtest_chart(sample_df, sample_trades, equity, drawdown)
    assert isinstance(result, str)
    assert result == expected_base64
    assert len(result) > 100
    mock_to_image.assert_called_once_with(format='png')

@patch('plotly.graph_objects.Figure.to_image')
def test_generate_regime_chart_returns_base64(mock_to_image, sample_df):
    import base64
    fake_image = b'fake_image_bytes_' * 20
    mock_to_image.return_value = fake_image
    expected_base64 = base64.b64encode(fake_image).decode('utf-8')
    
    analysis = {'regime': 'trend', 'adx': 30.0}
    result = generate_regime_chart(sample_df, analysis)
    assert isinstance(result, str)
    assert result == expected_base64
    assert len(result) > 100
    mock_to_image.assert_called_once_with(format='png')
