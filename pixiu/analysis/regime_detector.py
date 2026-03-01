"""择势判断模块

采用多指标投票机制，综合判断市场状态：
1. ADX 使用 Wilder 平滑判断趋势强度
2. RSI 判断动量状态
3. 布林带判断价格通道突破
4. 成交量确认趋势有效性
5. 状态持续确认机制避免单日噪音
"""
from typing import Dict
import pandas as pd
import numpy as np

EPSILON = 1e-10


class MarketRegimeDetector:
    """大盘/个股择势判断
    
    通过ADX、MA斜率、MACD、布林带等指标判断市场状态：
    - trend: 趋势行情（有方向性）
    - range: 震荡行情（方向不明）
    
    支持大盘/个股差异化参数配置
    """
    
    def __init__(
        self, 
        adx_period: int = 14, 
        ma_period: int = 20, 
        vol_period: int = 20, 
        rsi_period: int = 14, 
        bb_period: int = 20, 
        bb_std: float = 2.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        # 差异化参数：大盘用更低的阈值
        adx_trend_threshold: float = 20.0,  # ADX趋势阈值
        slope_threshold: float = 0.002,  # 斜率阈值
        trend_score_threshold: int = 5,  # 趋势判断门槛分（提高到5分以更好识别震荡）
    ):
        self.adx_period = adx_period
        self.ma_period = ma_period
        self.vol_period = vol_period
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.adx_trend_threshold = adx_trend_threshold
        self.slope_threshold = slope_threshold
        self.trend_score_threshold = trend_score_threshold
        # 状态记忆：用于持续确认机制
        self._regime_history = []  # 存储最近的判断结果
    
    def detect_regime(self, df: pd.DataFrame, enable_state_memory: bool = True) -> str:
        """判断市场状态（多指标投票机制）
        
        整合多个技术指标进行综合判断：
        - ADX: 趋势强度 (0-2分)
        - MA斜率: 方向性 (0-2分)
        - 波动率: 高波动支持趋势 (0-1分)
        - RSI: 动量状态 (0-1分)
        - 布林带: 价格通道突破 (0-3分)
        - 成交量: 量价确认 (0-1分)
        
        Args:
            df: 行情数据，包含open, high, low, close
            enable_state_memory: 是否启用状态记忆机制
            
        Returns:
            'trend' 或 'range'
        """
        detail = self.get_regime_detail(df, enable_state_memory)
        return detail['regime']
    
    def get_regime_detail(self, df: pd.DataFrame, enable_state_memory: bool = True) -> Dict:
        """获取详细的择势判断结果（包含评分详情）
        
        改进：移除震荡/趋势二分法，改为方向+强度评估
        
        Returns:
            {
                'direction': 'up'/'down'/'neutral',  # 方向
                'strength': 0-10,  # 强度分数
                'confidence': 0.0-1.0,  # 置信度
                'total_score': int,  # 总分
                'score_details': {
                    'adx_score': 0-2,      # ADX贡献
                    'slope_score': 0-2,    # 斜率贡献
                    'macd_score': 0-3,     # MACD贡献
                    'bb_score': 0-2,       # 布林带贡献
                    'ma_cross_score': 0-2, # MA交叉贡献
                    'momentum_score': 0-2, # 动量贡献
                    'volume_score': 0-1,   # 成交量贡献
                },
                'key_indicators': [...],  # 关键指标转折理由
                'indicators': {...}  # 所有指标值
            }
        """
        required_cols = ['open', 'high', 'low', 'close']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return {'direction': 'neutral', 'strength': 0, 'regime': 'range',
                    'confidence': 0.5, 'total_score': 0, 
                    'score_details': {}, 'key_indicators': [], 'indicators': {}}
        
        # 数据长度检查：需要足够数据计算各指标
        min_required = max(self.adx_period + 10, self.ma_period + 5, self.macd_slow + self.macd_signal)
        if len(df) < min_required:
            return {'direction': 'neutral', 'strength': 0, 'regime': 'range',
                    'confidence': 0.5, 'total_score': 0,
                    'score_details': {}, 'key_indicators': [], 'indicators': {}}
        
        # ============================================================
        # 第一步：计算所有指标
        # ============================================================
        adx = self._calc_adx(df)
        slope = self._calc_ma_slope(df)
        vol = self._calc_volatility(df)
        vol_ma = self._calc_volatility_ma(df)
        rsi = self._calc_rsi(df)
        macd_val, macd_signal_val, macd_hist = self._calc_macd(df)
        bb_upper, bb_middle, bb_lower, bb_bandwidth = self._calc_bollinger_bands(df)
        
        # 新增：MA5/MA20交叉和动量
        ma5 = df['close'].rolling(5).mean().iloc[-1] if len(df) >= 5 else df['close'].iloc[-1]
        ma20 = df['close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else df['close'].iloc[-1]
        ma5_prev = df['close'].rolling(5).mean().iloc[-2] if len(df) >= 6 else ma5
        ma20_prev = df['close'].rolling(20).mean().iloc[-2] if len(df) >= 21 else ma20
        
        # 动量：10日价格变化率
        momentum = 0
        if len(df) >= 10:
            momentum = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
        
        # ============================================================
        # 第二步：方向判断（信号投票）
        # ============================================================
        direction_votes = {'up': 0, 'down': 0, 'neutral': 0}
        score_details = {}
        key_indicators = []
        
        # 1. MA5/MA20交叉判断（权重最高）
        ma_cross_score = 0
        if ma5 > ma20 and ma5_prev <= ma20_prev:
            # 金叉
            direction_votes['up'] += 3
            ma_cross_score = 2
            key_indicators.append("MA金叉")
        elif ma5 < ma20 and ma5_prev >= ma20_prev:
            # 死叉
            direction_votes['down'] += 3
            ma_cross_score = 2
            key_indicators.append("MA死叉")
        elif ma5 > ma20:
            direction_votes['up'] += 1
            ma_cross_score = 1
        elif ma5 < ma20:
            direction_votes['down'] += 1
            ma_cross_score = 1
        score_details['ma_cross_score'] = ma_cross_score
        
        # 2. MA斜率方向（反映趋势方向）
        slope_score = 0
        if abs(slope) >= self.slope_threshold:
            if slope > 0:
                direction_votes['up'] += 2
                slope_score = 2
                key_indicators.append(f"斜率上涨({abs(slope)*100:.2f}%)")
            else:
                direction_votes['down'] += 2
                slope_score = 2
                key_indicators.append(f"斜率下跌({abs(slope)*100:.2f}%)")
        else:
            direction_votes['neutral'] += 1
        score_details['slope_score'] = slope_score
        
        # 3. MACD方向
        macd_score = 0
        if macd_hist is not None and len(macd_hist) >= 2:
            hist_val = macd_hist.iloc[-1]
            hist_prev = macd_hist.iloc[-2]
            if hist_val > 0 and hist_prev <= 0:
                direction_votes['up'] += 2
                macd_score = 2
                key_indicators.append("MACD金叉")
            elif hist_val < 0 and hist_prev >= 0:
                direction_votes['down'] += 2
                macd_score = 2
                key_indicators.append("MACD死叉")
            elif hist_val > 0:
                direction_votes['up'] += 1
                macd_score = 1
            elif hist_val < 0:
                direction_votes['down'] += 1
                macd_score = 1
        score_details['macd_score'] = macd_score
        
        # 4. 动量方向（10日变化）
        momentum_score = 0
        if abs(momentum) >= 0.03:  # 3%以上变化
            if momentum > 0:
                direction_votes['up'] += 1
                momentum_score = 1
                key_indicators.append(f"动量上涨({momentum*100:.1f}%)")
            else:
                direction_votes['down'] += 1
                momentum_score = 1
                key_indicators.append(f"动量下跌({abs(momentum)*100:.1f}%)")
        score_details['momentum_score'] = momentum_score
        
        # ============================================================
        # 第三步：强度判断（0-10分）
        # ============================================================
        strength = 0
        
        # 1. ADX强度（趋势强度指标）
        adx_score = 0
        if adx >= self.adx_trend_threshold + 5:
            adx_score = 2
            strength += 2
            key_indicators.append(f"ADX强({adx:.1f})")
        elif adx >= self.adx_trend_threshold:
            adx_score = 1
            strength += 1
        score_details['adx_score'] = adx_score
        
        # 2. MACD柱强度（红绿柱越大趋势越强）
        if macd_hist is not None and len(macd_hist) > 0:
            hist_abs = abs(macd_hist.iloc[-1])
            if hist_abs >= 0.01:
                strength += 1
            if hist_abs >= 0.02:
                strength += 1
        
        # 3. 布林带带宽（带宽越大趋势越强）
        bb_score = 0
        if bb_bandwidth >= 0.15:  # 15%以上带宽
            bb_score = 2
            strength += 2
            key_indicators.append(f"布林带宽{bb_bandwidth*100:.1f}%")
        elif bb_bandwidth >= 0.08:
            bb_score = 1
            strength += 1
        # 布林带突破
        current_price = df['close'].iloc[-1]
        if current_price > bb_upper:
            bb_score = max(bb_score, 2)
            strength = max(strength, strength + 1)
            key_indicators.append("突破上轨")
        elif current_price < bb_lower:
            bb_score = max(bb_score, 2)
            strength = max(strength, strength + 1)
            key_indicators.append("突破下轨")
        score_details['bb_score'] = bb_score
        
        # 4. 成交量确认
        vol_score = 0
        if 'volume' in df.columns:
            vol_signal = self._calc_volume_signal(df)
            if vol_signal > 0:
                vol_score = 1
                strength += 1
                key_indicators.append("放量")
        score_details['volume_score'] = vol_score
        
        # 5. RSI极端值（增加强度）
        rsi_score = 0
        if rsi >= 70 or rsi <= 30:
            rsi_score = 1
            strength += 1
            key_indicators.append(f"RSI极端({rsi:.0f})")
        score_details['rsi_score'] = rsi_score
        
        # 总分计算
        total_score = sum(score_details.values())
        
        # ============================================================
        # 第四步：最终判断
        # ============================================================
        # 确定方向
        if direction_votes['up'] > direction_votes['down'] + 1:
            final_direction = 'up'
        elif direction_votes['down'] > direction_votes['up'] + 1:
            final_direction = 'down'
        else:
            final_direction = 'neutral'
        
        # 强度上限为10
        strength = min(strength, 10)
        
        # 置信度 = 强度/10
        confidence = strength / 10.0
        
        # 兼容旧代码的regime字段
        final_regime = "trend" if strength >= 4 else "range"
        
        # 状态持续确认机制
        if enable_state_memory:
            self._regime_history.append(final_regime)
            if len(self._regime_history) > 3:
                self._regime_history.pop(0)
        
        # 构造返回结果
        return {
            'direction': final_direction,
            'strength': strength,
            'regime': final_regime,  # 兼容旧代码
            'confidence': round(confidence, 2),
            'total_score': total_score,
            'score_details': score_details,
            'key_indicators': key_indicators,
            'indicators': {
                'adx': round(adx, 2),
                'ma5': round(ma5, 2),
                'ma20': round(ma20, 2),
                'ma_slope': round(slope, 4),
                'momentum': round(momentum, 4),
                'volatility': round(vol, 4),
                'rsi': round(rsi, 2),
                'macd': round(macd_val, 4) if macd_val else 0,
                'macd_signal': round(macd_signal_val, 4) if macd_signal_val else 0,
                'macd_hist': round(macd_hist.iloc[-1], 4) if macd_hist is not None and len(macd_hist) > 0 else 0,
                'bb_upper': round(bb_upper, 2) if bb_upper > 0 else 0,
                'bb_middle': round(bb_middle, 2) if bb_middle > 0 else 0,
                'bb_lower': round(bb_lower, 2) if bb_lower > 0 else 0,
                'bb_bandwidth': round(bb_bandwidth, 4) if bb_bandwidth > 0 else 0,
                'direction_votes': direction_votes,
            }
        }
    
    def _calc_adx(self, df: pd.DataFrame) -> float:
        """计算ADX指标（使用Wilder平滑）
        
        Wilder平滑是EMA的变体，alpha = 1/period
        ADX >= 25 表示强趋势
        ADX >= 20 表示有趋势
        ADX < 20 表示无趋势/震荡
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        n = len(df)
        
        if n < self.adx_period + 10:
            return 0.0
        
        # 计算 True Range
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
        
        # 计算 +DM 和 -DM
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        for i in range(1, n):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
        
        # Wilder 平滑（EMA with alpha = 1/period）
        alpha = 1.0 / self.adx_period
        
        atr = self._wilder_smooth(tr, alpha)
        smooth_plus_dm = self._wilder_smooth(plus_dm, alpha)
        smooth_minus_dm = self._wilder_smooth(minus_dm, alpha)
        
        # 计算 +DI 和 -DI
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        for i in range(n):
            if atr[i] > EPSILON:
                plus_di[i] = 100 * smooth_plus_dm[i] / atr[i]
                minus_di[i] = 100 * smooth_minus_dm[i] / atr[i]
        
        # 计算 DX
        dx = np.zeros(n)
        for i in range(n):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > EPSILON:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
        
        # ADX 是 DX 的 Wilder 平滑
        adx = self._wilder_smooth(dx, alpha)
        
        return float(adx[-1]) if not np.isnan(adx[-1]) else 0.0
    
    def _wilder_smooth(self, data: np.ndarray, alpha: float) -> np.ndarray:
        """Wilder 平滑（指数移动平均的变体）"""
        n = len(data)
        result = np.zeros(n)
        result[0] = data[0]
        
        for i in range(1, n):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        
        return result
    
    def _calc_ma_slope(self, df: pd.DataFrame) -> float:
        """计算MA斜率（10日变化率）
        
        使用更长的周期（10天）来判断趋势方向，避免短期波动干扰
        返回值含义：0.005 表示10天内MA上涨了0.5%
        """
        ma = df['close'].rolling(self.ma_period).mean()
        
        if len(ma) < 10:
            return 0.0
        
        # 计算10日MA变化率（更稳定）
        ma_current = ma.iloc[-1]
        ma_10d_ago = ma.iloc[-10]
        
        if pd.isna(ma_current) or pd.isna(ma_10d_ago) or ma_10d_ago == 0:
            return 0.0
        
        slope = (ma_current - ma_10d_ago) / ma_10d_ago
        return float(slope)
    
    def _calc_volatility(self, df: pd.DataFrame) -> float:
        """计算短期波动率（20日）"""
        returns = df['close'].pct_change()
        vol = returns.rolling(self.vol_period).std().iloc[-1]
        return float(vol) if not pd.isna(vol) else 0.0
    
    def _calc_volatility_ma(self, df: pd.DataFrame) -> float:
        """计算长期波动率均值（60日），用于对比"""
        returns = df['close'].pct_change()
        vol_series = returns.rolling(self.vol_period).std()
        vol_ma = vol_series.rolling(60).mean().iloc[-1]
        return float(vol_ma) if not pd.isna(vol_ma) else self._calc_volatility(df)
    
    def _calc_macd(self, df: pd.DataFrame) -> tuple:
        """计算MACD指标
        
        MACD = DIF = EMA(12) - EMA(26)
        DEA = EMA(DIF, 9)
        MACD柱 = DIF - DEA
        
        Returns:
            (macd_val, signal_val, hist): MACD值、信号线、柱状图
        """
        if len(df) < self.macd_slow + self.macd_signal:
            return 0.0, 0.0, None
        
        close = df['close']
        
        # 计算EMA
        ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
        
        # DIF (MACD线)
        dif = ema_fast - ema_slow
        
        # DEA (信号线)
        dea = dif.ewm(span=self.macd_signal, adjust=False).mean()
        
        # MACD柱状图
        macd_hist = (dif - dea) * 2  # 乘2是惯例
        
        return float(dif.iloc[-1]), float(dea.iloc[-1]), macd_hist
    
    def _calc_rsi(self, df: pd.DataFrame) -> float:
        """计算RSI指标（相对强弱指数）
        
        RSI = 100 - 100 / (1 + RS)
        RS = 平均上涨幅度 / 平均下跌幅度
        
        Returns:
            RSI值，范围0-100
        """
        if len(df) < self.rsi_period + 1:
            return 50.0  # 默认中性值
        
        close = df['close'].values
        deltas = np.diff(close)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # 使用Wilder平滑计算平均涨跌幅
        alpha = 1.0 / self.rsi_period
        avg_gains = self._wilder_smooth(np.concatenate([[0], gains]), alpha)
        avg_losses = self._wilder_smooth(np.concatenate([[0], losses]), alpha)
        
        # 计算RS和RSI
        rs = avg_gains[-1] / (avg_losses[-1] + EPSILON)
        rsi = 100 - 100 / (1 + rs)
        
        return float(rsi) if not np.isnan(rsi) else 50.0
    
    def _calc_rsi_signal(self, df: pd.DataFrame) -> int:
        """根据RSI计算趋势信号
        
        Returns:
            趋势信号分数 0-1
        """
        rsi = self._calc_rsi(df)
        
        # RSI在极端区域（>60或<40）表示有趋势
        if rsi > 60 or rsi < 40:
            return 1
        # RSI在中性区域（40-60）表示震荡
        return 0
    
    def _calc_bollinger_bands(self, df: pd.DataFrame) -> tuple:
        """计算布林带
        
        Returns:
            (upper, middle, lower, bandwidth) 上轨、中轨、下轨、带宽
        """
        if len(df) < self.bb_period:
            return 0.0, 0.0, 0.0, 0.0
        
        close = df['close'].values
        middle = np.mean(close[-self.bb_period:])
        std = np.std(close[-self.bb_period:])
        
        upper = middle + self.bb_std * std
        lower = middle - self.bb_std * std
        bandwidth = (upper - lower) / middle if middle > EPSILON else 0.0
        
        return upper, middle, lower, bandwidth
    
    def _calc_bb_signal(self, df: pd.DataFrame) -> int:
        """根据布林带计算趋势信号
        
        Returns:
            趋势信号分数 0-3
        """
        if len(df) < self.bb_period + 60:
            return 0
        
        upper, middle, lower, bandwidth = self._calc_bollinger_bands(df)
        current_price = df['close'].iloc[-1]
        
        signal = 0
        
        # 1. 价格突破上轨或下轨：强趋势信号
        if current_price > upper or current_price < lower:
            signal += 2
        # 价格在中轨附近（±0.5倍带宽）：震荡信号
        elif abs(current_price - middle) < 0.5 * (upper - middle):
            return 0  # 震荡信号，不加分
        
        # 2. 布林带宽度扩张：趋势信号
        # 计算60日平均带宽
        bandwidth_history = []
        for i in range(max(60, self.bb_period), len(df)):
            window_df = df.iloc[i-self.bb_period:i]
            _, _, _, bw = self._calc_bollinger_bands(window_df)
            bandwidth_history.append(bw)
        
        if bandwidth_history:
            avg_bandwidth = np.mean(bandwidth_history)
            if bandwidth > avg_bandwidth * 1.0:  # 带宽高于平均值
                signal += 1
        
        return signal
    
    def _calc_volume_signal(self, df: pd.DataFrame) -> int:
        """根据成交量计算趋势信号
        
        Returns:
            趋势信号分数 0-1
        """
        # 如果数据中没有volume字段，返回0
        if 'volume' not in df.columns or len(df) < 20:
            return 0
        
        # 计算5日和20日成交量均线
        vol_ma5 = df['volume'].rolling(5).mean().iloc[-1]
        vol_ma20 = df['volume'].rolling(20).mean().iloc[-1]
        
        # 计算价格变化
        price_change = df['close'].pct_change().iloc[-1]
        
        # 量价齐升：趋势信号
        if vol_ma5 > vol_ma20 * 1.2 and abs(price_change) > 0.01:
            return 1
        
        return 0
    
    def get_analysis_detail(self, df: pd.DataFrame) -> Dict:
        """获取详细分析结果（包括所有指标）
        
        兼容旧API,内部调用get_regime_detail
        """
        detail = self.get_regime_detail(df, enable_state_memory=False)
        return detail
