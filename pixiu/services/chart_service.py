import base64
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List
from pixiu.models.backtest import Trade


def generate_backtest_chart(
    df: pd.DataFrame,
    trades: List[Trade],
    equity_curve: List[float],
    drawdown_curve: List[float]
) -> str:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=('K线与交易信号', '资金曲线', '回撤曲线')
    )
    
    fig.add_trace(
        go.Candlestick(
            x=df['trade_date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K线',
            increasing_line_color='red',
            decreasing_line_color='green'
        ),
        row=1,
        col=1
    )
    
    buy_trades = [t for t in trades if t.signal_type == 'BUY']
    sell_trades = [t for t in trades if t.signal_type == 'SELL']
    
    if buy_trades:
        buy_dates = pd.to_datetime([t.trade_date for t in buy_trades])
        buy_prices = [t.price for t in buy_trades]
        fig.add_trace(
            go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='red'),
                name='买入'
            ),
            row=1,
            col=1
        )
    
    if sell_trades:
        sell_dates = pd.to_datetime([t.trade_date for t in sell_trades])
        sell_prices = [t.price for t in sell_trades]
        fig.add_trace(
            go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode='markers',
                marker=dict(symbol='triangle-down', size=12, color='green'),
                name='卖出'
            ),
            row=1,
            col=1
        )
    
    fig.add_trace(
        go.Scatter(
            x=df['trade_date'],
            y=equity_curve,
            mode='lines',
            name='资金曲线',
            line=dict(color='blue', width=2)
        ),
        row=2,
        col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['trade_date'],
            y=drawdown_curve,
            mode='lines',
            name='回撤',
            fill='tozeroy',
            line=dict(color='red', width=2)
        ),
        row=3,
        col=1
    )
    
    fig.update_layout(
        template='plotly_dark',
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    fig.update_xaxes(title_text='日期', row=3, col=1)
    fig.update_yaxes(title_text='价格', row=1, col=1)
    fig.update_yaxes(title_text='资金', row=2, col=1)
    fig.update_yaxes(title_text='回撤率', row=3, col=1)
    
    img_bytes = fig.to_image(format='png')
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    
    return base64_str


def generate_regime_chart(df: pd.DataFrame, analysis: dict) -> str:
    df = df.copy()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['adx'] = _calculate_adx(df, period=14)
    
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.4],
        subplot_titles=('价格与MA20', 'ADX指标')
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['trade_date'],
            y=df['close'],
            mode='lines',
            name='收盘价',
            line=dict(color='#00D4FF', width=2)
        ),
        row=1,
        col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['trade_date'],
            y=df['ma20'],
            mode='lines',
            name='MA20',
            line=dict(color='orange', width=1.5)
        ),
        row=1,
        col=1
    )
    
    regime = analysis.get('regime', 'unknown')
    adx_value = analysis.get('adx', 0)
    
    regime_color = 'green' if regime == 'trend' else 'orange' if regime == 'range' else 'gray'
    regime_text = f'趋势' if regime == 'trend' else f'震荡' if regime == 'range' else f'未知'
    
    fig.add_annotation(
        x=df['trade_date'].iloc[-1],
        y=df['close'].iloc[-1],
        text=f'{regime_text} (ADX: {adx_value:.1f})',
        showarrow=False,
        font=dict(size=14, color=regime_color),
        xanchor='right',
        yanchor='top',
        row=1,
        col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['trade_date'],
            y=df['adx'],
            mode='lines',
            name='ADX',
            line=dict(color='purple', width=2)
        ),
        row=2,
        col=1
    )
    
    fig.add_hline(
        y=25,
        line_dash='dash',
        line_color='red',
        annotation_text='ADX=25',
        row=2,
        col=1
    )
    
    fig.update_layout(
        template='plotly_dark',
        height=500,
        showlegend=True
    )
    
    fig.update_xaxes(title_text='日期', row=2, col=1)
    fig.update_yaxes(title_text='价格', row=1, col=1)
    fig.update_yaxes(title_text='ADX', row=2, col=1)
    
    img_bytes = fig.to_image(format='png')
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    
    return base64_str


def _calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (abs(minus_dm.rolling(window=period).mean()) / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx
