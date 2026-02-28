"""择势判断模块"""
from typing import Dict
import pandas as pd
import numpy as np

EPSILON = 1e-10


class MarketRegimeDetector:
    """大盘/个股择势判断
    
    通过ADX、MA斜率、波动率等指标判断市场状态：
    - trend: 趋势行情，适合跟踪策略
    - range: 震荡行情，适合均值回归策略
    """
    
    def __init__(self, adx_period: int = 14, ma_period: int = 20, vol_period: int = 20):
        self.adx_period = adx_period
        self.ma_period = ma_period
        self.vol_period = vol_period
    
    def detect_regime(self, df: pd.DataFrame) -> str:
        """判断市场状态
        
        Args:
            df: 行情数据，包含open, high, low, close
            
        Returns:
            'trend' 或 'range'
            
        Raises:
            ValueError: 如果缺少必需的列
        """
        required_cols = ['open', 'high', 'low', 'close']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"缺少必需的列: {missing}")
        
        if len(df) < max(self.adx_period, self.ma_period, self.vol_period) + 10:
            return "range"
        
        adx = self._calc_adx(df)
        slope = self._calc_ma_slope(df)
        vol = self._calc_volatility(df)
        vol_ma = self._calc_volatility(df)
        
        trend_score = 0.0
        
        if adx > 25:
            trend_score += 0.4
        
        if abs(slope) > 0.005:
            trend_score += 0.3
        
        if vol > vol_ma * 1.2:
            trend_score += 0.3
        
        return "trend" if trend_score > 0.5 else "range"
    
    def _calc_adx(self, df: pd.DataFrame) -> float:
        """计算ADX指标
        
        ADX > 25 表示趋势行情
        ADX < 25 表示震荡行情
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(self.adx_period).mean()
        
        plus_di = 100 * (plus_dm.rolling(self.adx_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(self.adx_period).mean() / atr)
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPSILON)
        adx = dx.rolling(self.adx_period).mean()
        
        return float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0
    
    def _calc_ma_slope(self, df: pd.DataFrame) -> float:
        """计算MA斜率
        
        |斜率| > 0.005 (0.5%) 表示趋势行情
        """
        ma = df['close'].rolling(self.ma_period).mean()
        
        if len(ma) < 2:
            return 0.0
        
        slope = (ma.iloc[-1] - ma.iloc[-2]) / ma.iloc[-2]
        return float(slope)
    
    def _calc_volatility(self, df: pd.DataFrame) -> float:
        """计算波动率"""
        returns = df['close'].pct_change()
        vol = returns.rolling(self.vol_period).std().iloc[-1]
        return float(vol) if not pd.isna(vol) else 0.0
    
    def get_analysis_detail(self, df: pd.DataFrame) -> Dict:
        """获取详细分析结果
        
        Returns:
            包含regime, adx, ma_slope, volatility的字典
        """
        return {
            "regime": self.detect_regime(df),
            "adx": round(self._calc_adx(df), 2),
            "ma_slope": round(self._calc_ma_slope(df), 4),
            "volatility": round(self._calc_volatility(df), 4),
        }
