import base64
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from typing import List
from io import BytesIO
from pixiu.models.backtest import Trade

plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#16213e'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.edgecolor'] = '#444'
plt.rcParams['grid.color'] = '#333'


def generate_backtest_chart(
    df: pd.DataFrame,
    trades: List[Trade],
    equity_curve: List[float],
    drawdown_curve: List[float]
) -> str:
    df = df.copy()
    if 'trade_date' not in df.columns:
        if df.index.name == 'trade_date':
            df = df.reset_index()
        else:
            df = df.reset_index()
            if 'index' in df.columns:
                df = df.rename(columns={'index': 'trade_date'})
    
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    dates = df['trade_date'].tolist()
    
    for i in range(len(df)):
        o = float(df['open'].iloc[i])
        h = float(df['high'].iloc[i])
        l = float(df['low'].iloc[i])
        c = float(df['close'].iloc[i])
        color = '#ef4444' if c >= o else '#22c55e'
        ax1.plot([dates[i], dates[i]], [l, h], color=color, linewidth=1)
        ax1.plot([dates[i], dates[i]], [o, c], color=color, linewidth=3)
    
    buy_trades = [t for t in trades if t.signal_type == 'BUY']
    sell_trades = [t for t in trades if t.signal_type == 'SELL']
    
    if buy_trades:
        buy_dates = [pd.to_datetime(t.trade_date) for t in buy_trades]
        buy_prices = [t.price for t in buy_trades]
        ax1.scatter(buy_dates, buy_prices, marker='^', color='#ef4444', s=100, zorder=5, label='BUY')
    
    if sell_trades:
        sell_dates = [pd.to_datetime(t.trade_date) for t in sell_trades]
        sell_prices = [t.price for t in sell_trades]
        ax1.scatter(sell_dates, sell_prices, marker='v', color='#22c55e', s=100, zorder=5, label='SELL')
    
    ax1.set_title('K-Line & Trade Signals', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    ax2.plot(dates, equity_curve, color='#3b82f6', linewidth=2)
    ax2.fill_between(dates, equity_curve, alpha=0.3, color='#3b82f6')
    ax2.set_title('Equity Curve', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    drawdown_pct = [d * 100 for d in drawdown_curve]
    ax3.plot(dates, drawdown_pct, color='#ef4444', linewidth=1.5)
    ax3.fill_between(dates, drawdown_pct, alpha=0.3, color='#ef4444')
    ax3.set_title('Drawdown (%)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    plt.tight_layout()
    
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode()


def generate_regime_chart(df: pd.DataFrame, analysis: dict) -> str:
    df = df.copy()
    if 'trade_date' not in df.columns:
        if df.index.name == 'trade_date':
            df = df.reset_index()
        else:
            df = df.reset_index()
            if 'index' in df.columns:
                df = df.rename(columns={'index': 'trade_date'})
    
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['adx'] = _calculate_adx(df, period=14)
    
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    dates = df['trade_date'].tolist()
    
    ax1.plot(dates, df['close'], color='#00D4FF', linewidth=2, label='Close')
    ax1.plot(dates, df['ma20'], color='orange', linewidth=1.5, linestyle='--', label='MA20')
    
    regime = analysis.get('regime', 'unknown')
    adx_value = analysis.get('adx', 0)
    
    regime_color = '#22c55e' if regime == 'trend' else '#f59e0b' if regime == 'range' else 'gray'
    regime_text = 'TREND' if regime == 'trend' else 'RANGE' if regime == 'range' else 'UNKNOWN'
    
    ax1.annotate(f'{regime_text} (ADX: {adx_value:.1f})',
                xy=(0.98, 0.95), xycoords='axes fraction',
                fontsize=12, fontweight='bold', color=regime_color,
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor=regime_color, alpha=0.3))
    
    ax1.set_title('Price & MA20', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    ax2.plot(dates, df['adx'], color='#a855f7', linewidth=2, label='ADX')
    ax2.axhline(y=25, color='#ef4444', linestyle='--', linewidth=1, label='Trend Line (25)')
    ax2.set_title('ADX Indicator', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    plt.tight_layout()
    
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode()


def _calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    
    atr_safe = atr.replace(0, float('nan'))
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_safe)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_safe)
    
    di_sum = plus_di + minus_di
    di_sum_safe = di_sum.replace(0, float('nan'))
    dx = 100 * (plus_di - minus_di).abs() / di_sum_safe
    adx = dx.rolling(window=period).mean()
    
    return adx.fillna(0)
