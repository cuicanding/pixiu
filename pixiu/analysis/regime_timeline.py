"""时间线择势分析"""
from typing import Dict, List, Any, TypedDict
import pandas as pd
from .regime_detector import MarketRegimeDetector


class TimelineSegment(TypedDict):
    start: str
    end: str
    regime: str  # 保留兼容性，但实际只用direction
    direction: str  # "up" 或 "down"
    strength: int  # 趋势强度 0-10
    duration: int
    start_df_idx: int  # 在df中的起始索引
    end_df_idx: int  # 在df中的结束索引


class TurningPoint(TypedDict):
    date: str
    from_regime: str
    to_regime: str
    triggers: Dict[str, Any]


class RegimeTimelineResult(TypedDict, total=False):
    segments: List[TimelineSegment]
    turning_points: List[TurningPoint]
    current: Dict[str, Any]


class RegimeTimelineAnalyzer:
    """时间线择势分析器
    
    通过滚动窗口分析历史行情，识别趋势/震荡阶段及转势点
    支持大盘/个股差异化参数配置
    """
    
    def __init__(
        self,
        window: int = 60,
        adx_threshold: float = 25.0,
        slope_threshold: float = 0.005,
        min_duration: int = 15,  # 最小片段持续天数（大盘用20，个股用10）
        min_change_pct: float = 0.05,  # 最小变化幅度，小于此值的区间将被合并
        detector: 'MarketRegimeDetector' = None
    ):
        """Initialize the timeline analyzer.
        
        Args:
            window: Rolling window size in days for regime detection
            adx_threshold: Threshold for identifying ADX-based turning point triggers
            slope_threshold: Threshold for identifying slope-based turning point triggers
            min_duration: Minimum segment duration in days (大盘20天, 个股10天)
            min_change_pct: Minimum price change percentage (default 5%), segments below this will be merged
            detector: Custom MarketRegimeDetector instance (if None, creates default)
        """
        self.window = window
        self.adx_threshold = adx_threshold
        self.slope_threshold = slope_threshold
        self.min_duration = min_duration
        self.min_change_pct = min_change_pct
        self.detector = detector if detector is not None else MarketRegimeDetector()
    
    def analyze_timeline(self, df: pd.DataFrame) -> RegimeTimelineResult:
        """分析时间线择势
        
        Args:
            df: 行情数据，包含open, high, low, close, trade_date
            
        Returns:
            {
                'segments': [{'start': date, 'end': date, 'regime': 'trend'|'range', 'duration': int}],
                'turning_points': [{'date': date, 'from_regime': regime, 'to_regime': regime, 'triggers': {...}}],
                'current': {...}
            }
        """
        if df.empty or len(df) < self.window:
            return {
                'segments': [],
                'turning_points': [],
                'current': None
            }
        
        regime_history = self._build_regime_history(df)
        segments = self._build_segments(df, regime_history)
        
        # 转折点基于平滑后的 segments 生成，而不是原始 regime_history
        # 这样确保 segments 和 turning_points 一致
        turning_points = self._detect_turning_points_from_segments(segments, df, regime_history)
        
        current_regime = regime_history[-1] if regime_history else None
        current_detail = None
        if current_regime and len(df) >= self.window:
            current_detail = self.detector.get_analysis_detail(df.tail(self.window + 20))
        
        return {
            'segments': segments,
            'turning_points': turning_points,
            'current': current_detail
        }
    
    def _build_regime_history(self, df: pd.DataFrame) -> List[Dict]:
        """构建滚动窗口的regime历史
        
        注意：启用状态记忆机制，实现连续3天确认
        """
        # 清空 detector 的状态历史，确保每次分析从头开始
        self.detector._regime_history = []
        
        regime_history = []
        
        for i in range(self.window, len(df) + 1):
            window_df = df.iloc[i - self.window:i]
            # 启用状态记忆，实现连续3天确认
            detail = self.detector.get_regime_detail(window_df, enable_state_memory=True)
            regime = detail['regime']
            
            regime_history.append({
                'idx': i - 1,
                'date': df.iloc[i - 1].get('trade_date', i - 1),
                'regime': regime,
                'detail': detail
            })
        
        return regime_history
    
    def _build_segments(self, df: pd.DataFrame, regime_history: List[Dict]) -> List[Dict]:
        """将连续相同regime的点聚合成segment，并判断趋势方向
        
        包含平滑处理：
        1. 添加前置段覆盖从数据起始到第一个窗口
        2. 过滤持续时间太短的片段（少于10天的视为噪声）
        3. 确保段与段之间连续无间隙
        """
        if not regime_history:
            return []
        
        first_data_date = str(df.iloc[0].get('trade_date', 0))
        last_data_date = str(df.iloc[-1].get('trade_date', 0))
        
        # 第一步：构建原始 segments
        raw_segments = []
        current_regime = regime_history[0]['regime']
        start_idx = 0
        
        for i, item in enumerate(regime_history):
            if item['regime'] != current_regime:
                # 计算该段的趋势方向，使用 df 的真实索引
                start_df_idx = regime_history[start_idx]['idx']
                end_df_idx = regime_history[i - 1]['idx']
                direction = self._calculate_direction_by_idx(df, start_df_idx, end_df_idx)
                
                raw_segments.append({
                    'start_df_idx': start_df_idx,
                    'end_df_idx': end_df_idx,
                    'start_date': str(regime_history[start_idx]['date']),
                    'end_date': str(regime_history[i - 1]['date']),
                    'regime': current_regime,
                    'direction': direction,
                    'duration': i - start_idx
                })
                current_regime = item['regime']
                start_idx = i
        
        # 最后一段
        start_df_idx = regime_history[start_idx]['idx']
        end_df_idx = regime_history[-1]['idx']
        direction = self._calculate_direction_by_idx(df, start_df_idx, end_df_idx)
        raw_segments.append({
            'start_df_idx': start_df_idx,
            'end_df_idx': end_df_idx,
            'start_date': str(regime_history[start_idx]['date']),
            'end_date': last_data_date,
            'regime': current_regime,
            'direction': direction,
            'duration': len(regime_history) - start_idx
        })
        
        # 第二步：添加前置段（从数据起始到第一个 regime 日期前一天）
        if raw_segments and self.window > 0:
            first_regime_date = raw_segments[0]['start_date']
            if first_data_date != first_regime_date:
                # 使用第一个 segment 的 regime 作为前置段的 regime
                pre_regime = raw_segments[0]['regime']
                # 前置段的 end_df_idx 应该是 window-1（第一个 regime 日期的前一天）
                pre_end_idx = self.window - 1
                pre_direction = self._calculate_direction_by_idx(df, 0, pre_end_idx)
                
                # 获取前置段的结束日期（第一个 regime 日期的前一天）
                pre_end_date = str(df.iloc[pre_end_idx].get('trade_date', 0))
                
                raw_segments.insert(0, {
                    'start_df_idx': 0,
                    'end_df_idx': pre_end_idx,
                    'start_date': first_data_date,
                    'end_date': pre_end_date,  # 使用正确的结束日期，避免重叠
                    'regime': pre_regime,
                    'direction': pre_direction,
                    'duration': self.window
                })
        
        # 第三步：基于短期方向变化进一步分割segment
        # 改进：降低阈值使检测更敏感
        DIRECTION_LOOKBACK = 5  # 用5天计算短期方向（更敏感）
        DIRECTION_THRESHOLD = 0.02  # 2%以上变化视为显著（降低阈值）
        
        direction_split_segments = []
        for seg in raw_segments:
            # 检查segment内是否有短期方向变化
            seg_start = seg['start_df_idx']
            seg_end = seg['end_df_idx']
            
            if seg_end - seg_start < DIRECTION_LOOKBACK * 2:
                # segment太短，不需要分割
                direction_split_segments.append(seg.copy())
                continue
            
            # 遍历segment内部，找方向变化点
            current_start = seg_start
            current_direction = None
            
            for j in range(seg_start + DIRECTION_LOOKBACK, seg_end + 1):
                # 计算短期方向
                lookback_start = j - DIRECTION_LOOKBACK
                start_price = float(df.iloc[lookback_start]['close'])
                end_price = float(df.iloc[j]['close'])
                change_pct = (end_price - start_price) / start_price
                
                if abs(change_pct) < DIRECTION_THRESHOLD:
                    continue  # 变化不显著，跳过
                
                new_direction = "up" if change_pct >= 0 else "down"
                
                if current_direction is None:
                    current_direction = new_direction
                elif new_direction != current_direction:
                    # 方向变化了，在此点分割
                    direction_split_segments.append({
                        'start_df_idx': current_start,
                        'end_df_idx': j - 1,
                        'start_date': str(df.iloc[current_start].get('trade_date', '')),
                        'end_date': str(df.iloc[j - 1].get('trade_date', '')),
                        'regime': seg['regime'],
                        'direction': current_direction,
                        'duration': j - 1 - current_start + 1
                    })
                    current_start = j
                    current_direction = new_direction
            
            # 添加剩余部分
            if current_start < seg_end:
                direction_split_segments.append({
                    'start_df_idx': current_start,
                    'end_df_idx': seg_end,
                    'start_date': str(df.iloc[current_start].get('trade_date', '')),
                    'end_date': seg['end_date'],
                    'regime': seg['regime'],
                    'direction': current_direction if current_direction else seg['direction'],
                    'duration': seg_end - current_start + 1
                })
        
        raw_segments = direction_split_segments
        
        # 第四步：多轮平滑处理 - 合并持续时间太短的片段
        # 关键：短片段视为噪音，合并到前一个片段
        # 使用可配置的最小持续天数
        MIN_DURATION = self.min_duration
        
        # 进行多轮平滑，直到没有短片段
        for _ in range(10):  # 最多10轮
            smoothed_segments = []
            has_short = False
            
            for seg in raw_segments:
                if seg['duration'] >= MIN_DURATION:
                    # 长片段直接保留
                    smoothed_segments.append(seg.copy())
                else:
                    # 短片段需要处理
                    has_short = True
                    if smoothed_segments:
                        # 合并到前一个片段（视为噪音）
                        prev_seg = smoothed_segments[-1]
                        prev_seg['end_df_idx'] = seg['end_df_idx']
                        prev_seg['end_date'] = seg['end_date']
                        prev_seg['duration'] += seg['duration']
                        # 保持前一个片段的 regime 和 direction
                    # 如果没有前一个片段，跳过（第一个片段通常是前置段）
            
            raw_segments = smoothed_segments
            if not has_short:
                break
        
        # 第五步：不再合并同方向片段
        # 关键改进：保留regime变化的转折点，即使方向相同
        # 因为regime变化（trend<->range）本身就是重要的市场状态变化
        smoothed_segments = raw_segments
        
        # 第六步：合并变化幅度太小的区间（<5%）
        # 如果一个区间的价格变化幅度很小，应该被合并到相邻区间
        MIN_CHANGE = self.min_change_pct  # 最小变化幅度5%
        
        for _ in range(10):  # 最多10轮
            new_segments = []
            has_small_change = False
            
            for i, seg in enumerate(smoothed_segments):
                # 计算该区间的价格变化幅度
                try:
                    start_price = float(df.iloc[seg['start_df_idx']]['close'])
                    end_price = float(df.iloc[seg['end_df_idx']]['close'])
                    change_pct = abs((end_price - start_price) / start_price)
                except:
                    change_pct = 1.0  # 如果计算失败，不合并
                
                if change_pct >= MIN_CHANGE:
                    # 变化幅度足够大，保留
                    new_segments.append(seg.copy())
                else:
                    # 变化幅度太小，合并到前一个区间
                    has_small_change = True
                    if new_segments:
                        prev_seg = new_segments[-1]
                        prev_seg['end_df_idx'] = seg['end_df_idx']
                        prev_seg['end_date'] = seg['end_date']
                        prev_seg['duration'] += seg['duration']
                        # 重新计算方向
                        prev_seg['direction'] = self._calculate_direction_by_idx(
                            df, prev_seg['start_df_idx'], prev_seg['end_df_idx']
                        )
                    elif i < len(smoothed_segments) - 1:
                        # 如果是第一个区间且变化小，合并到下一个
                        next_seg = smoothed_segments[i + 1]
                        next_seg['start_df_idx'] = seg['start_df_idx']
                        next_seg['start_date'] = seg['start_date']
                        next_seg['duration'] += seg['duration']
            
            smoothed_segments = new_segments if new_segments else smoothed_segments
            if not has_small_change:
                break
        
        # 第七步：构建最终 segments，确保连续无间隙
        final_segments = []
        for i, seg in enumerate(smoothed_segments):
            # 确保连续：当前段的 end = 下一段的 start
            if i < len(smoothed_segments) - 1:
                end_date = smoothed_segments[i + 1]['start_date']
            else:
                end_date = last_data_date
            
            # 关键修复：重新计算方向，基于合并后的实际价格变化
            actual_direction = self._calculate_direction_by_idx(
                df, seg['start_df_idx'], seg['end_df_idx']
            )
            
            final_segments.append({
                'index': i + 1,  # 添加序号
                'start': seg['start_date'],
                'end': end_date,
                'start_df_idx': seg['start_df_idx'],  # 添加索引用于计算价格
                'end_df_idx': seg['end_df_idx'],
                'regime': seg['regime'],
                'direction': actual_direction,  # 使用重新计算的方向
                'duration': seg['duration']
            })
        
        return final_segments
    
    def _calculate_direction_by_idx(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> str:
        """根据 df 索引计算趋势方向
        
        改进：综合考虑整个区间和短期趋势
        - 区间整体变化为主要判断依据
        - 短期趋势作为辅助判断
        """
        try:
            # 方法1：区间整体变化（主要判断）
            start_price = float(df.iloc[start_idx]['close'])
            end_price = float(df.iloc[end_idx]['close'])
            overall_change = (end_price - start_price) / start_price
            
            # 方法2：短期趋势（最近5天，更敏感）
            lookback = min(5, end_idx - start_idx)
            if lookback > 0:
                recent_start = end_idx - lookback
                recent_start_price = float(df.iloc[recent_start]['close'])
                short_term_change = (end_price - recent_start_price) / recent_start_price
            else:
                short_term_change = overall_change
            
            # 综合判断：优先使用整体变化方向
            # 如果短期变化更显著且幅度较大，则使用短期方向
            
            overall_direction = "up" if overall_change >= 0 else "down"
            short_term_direction = "up" if short_term_change >= 0 else "down"
            
            # 如果两者方向一致，直接返回
            if overall_direction == short_term_direction:
                return overall_direction
            
            # 如果不一致，看哪个变化更显著
            if abs(overall_change) >= abs(short_term_change):
                return overall_direction
            else:
                # 短期变化更显著时，需要看是否真的代表趋势反转
                # 降低阈值到2%以更敏感
                if abs(short_term_change) >= 0.02:
                    return short_term_direction
                else:
                    return overall_direction
                    
        except Exception:
            return "up"
    
    def _detect_turning_points_from_segments(
        self, segments: List[Dict], df: pd.DataFrame, regime_history: List[Dict]
    ) -> List[Dict]:
        """基于平滑后的 segments 检测转折点
        
        新逻辑：只要方向变化（up↔down）就算转折点
        同时记录强度变化
        """
        turning_points = []
        tp_index = 0  # 转折点序号计数器
        
        # 构建日期到 regime_history 索引的映射，用于获取指标详情
        date_to_detail = {}
        for item in regime_history:
            date_to_detail[str(item['date'])] = item.get('detail', {})
        
        for i in range(1, len(segments)):
            prev_seg = segments[i - 1]
            curr_seg = segments[i]
            
            # 方向变化检测：up↔down
            prev_dir = prev_seg.get('direction', 'neutral')
            curr_dir = curr_seg.get('direction', 'neutral')
            
            # 只有方向真正变化时才生成转折点（up↔down）
            direction_changed = (
                prev_dir != curr_dir and 
                prev_dir in ['up', 'down'] and 
                curr_dir in ['up', 'down']
            )
            
            if not direction_changed:
                continue  # 方向没变化，跳过
            
            tp_index += 1  # 递增序号
            # 转折点日期是当前 segment 的起始日期
            tp_date = curr_seg['start']
            
            # 尝试获取前后的指标详情
            prev_detail = date_to_detail.get(prev_seg['end'], {})
            curr_detail = date_to_detail.get(tp_date, {})
            
            # 获取方向和强度
            prev_strength = prev_detail.get('strength', 5)
            curr_strength = curr_detail.get('strength', 5)
            
            # 获取置信度、评分和关键指标
            confidence = curr_detail.get('confidence', 0.5)
            total_score = curr_detail.get('total_score', 0)
            key_indicators = curr_detail.get('key_indicators', [])
            score_details = curr_detail.get('score_details', {})
            indicators = curr_detail.get('indicators', {})
            
            # 生成转折理由
            reason_parts = []
            # 方向变化
            dir_desc = "上涨→下跌" if prev_dir == 'up' else "下跌→上涨"
            reason_parts.append(f"{dir_desc}")
            # 强度变化
            if abs(curr_strength - prev_strength) >= 2:
                reason_parts.append(f"强度{prev_strength}→{curr_strength}")
            # 关键指标
            if key_indicators:
                reason_parts.extend(key_indicators[:2])  # 最多2个指标
            reason = "; ".join(reason_parts) if reason_parts else "方向转变"
            
            turning_points.append({
                'index': tp_index,
                'date': tp_date,
                'from_direction': prev_dir,
                'to_direction': curr_dir,
                'from_strength': prev_strength,  # 前一阶段强度
                'to_strength': curr_strength,    # 当前阶段强度
                'prev_duration': prev_seg.get('duration', 0),  # 前一阶段持续天数
                'confidence': confidence,
                'confidence_pct': round(confidence * 100, 0),
                'total_score': total_score,
                'key_indicators': key_indicators,
                'reason': reason,
                'indicators': indicators,
                # 兼容旧字段
                'from_regime': prev_seg.get('regime', 'range'),
                'to_regime': curr_seg.get('regime', 'range'),
            })
        
        return turning_points

    def _detect_turning_points(self, df: pd.DataFrame, regime_history: List[Dict]) -> List[Dict]:
        """检测转势点"""
        turning_points = []
        
        for i in range(1, len(regime_history)):
            prev = regime_history[i - 1]
            curr = regime_history[i]
            
            if prev['regime'] != curr['regime']:
                triggers = self._identify_triggers(prev['detail'], curr['detail'])
                turning_points.append({
                    'date': curr['date'],
                    'from_regime': prev['regime'],
                    'to_regime': curr['regime'],
                    'triggers': triggers
                })
        
        return turning_points
    
    def _identify_triggers(self, prev_detail: Dict, curr_detail: Dict) -> Dict:
        """识别导致转势的触发因素"""
        triggers = {
            'adx_cross_up': False,
            'adx_cross_down': False,
            'slope_increase': False,
            'slope_decrease': False
        }
        
        if prev_detail.get('adx', 0) < self.adx_threshold <= curr_detail.get('adx', 0):
            triggers['adx_cross_up'] = True
        elif prev_detail.get('adx', 0) >= self.adx_threshold > curr_detail.get('adx', 0):
            triggers['adx_cross_down'] = True
        
        prev_slope = abs(prev_detail.get('ma_slope', 0))
        curr_slope = abs(curr_detail.get('ma_slope', 0))
        
        if curr_slope > prev_slope * 1.5:
            triggers['slope_increase'] = True
        elif curr_slope < prev_slope * 0.5:
            triggers['slope_decrease'] = True
        
        return triggers
    
    def _calculate_confidence(self, detail: Dict) -> float:
        """计算转折点置信度
        
        基于多个指标的强度计算置信度(0-1)
        """
        if not detail:
            return 0.5
        
        score = 0
        max_score = 0
        
        # 1. ADX强度 (权重30%)
        adx = detail.get('adx', 0)
        max_score += 30
        if adx >= 30:
            score += 30
        elif adx >= 25:
            score += 20
        elif adx >= 20:
            score += 10
        
        # 2. MA斜率强度 (权重25%)
        slope = abs(detail.get('ma_slope', 0))
        max_score += 25
        if slope >= 0.008:
            score += 25
        elif slope >= 0.005:
            score += 20
        elif slope >= 0.003:
            score += 15
        
        # 3. RSI极端程度 (权重20%)
        rsi = detail.get('rsi', 50)
        max_score += 20
        if rsi > 70 or rsi < 30:
            score += 20  # 极端区域
        elif rsi > 60 or rsi < 40:
            score += 15  # 接近极端
        else:
            score += 5  # 中性
        
        # 4. 布林带带宽 (权重15%)
        bb_width = detail.get('bb_bandwidth', 0)
        max_score += 15
        if bb_width > 0.15:
            score += 15  # 宽带宽,趋势明显
        elif bb_width > 0.10:
            score += 10
        else:
            score += 5
        
        # 5. 波动率 (权重10%)
        vol = detail.get('volatility', 0)
        max_score += 10
        if vol > 0.02:
            score += 10
        elif vol > 0.015:
            score += 7
        else:
            score += 3
        
        return round(score / max_score, 2) if max_score > 0 else 0.5
    
    def _check_resonance(self, detail: Dict) -> Dict[str, bool]:
        """检查指标共振
        
        Returns:
            {
                'trend_resonance': True/False,  # 趋势共振
                'momentum_resonance': True/False,  # 动量共振
                'volatility_resonance': True/False,  # 波动率共振
                'overall': True/False  # 整体共振(至少2个)
            }
        """
        if not detail:
            return {
                'trend_resonance': False,
                'momentum_resonance': False,
                'volatility_resonance': False,
                'overall': False
            }
        
        # 趋势共振: ADX>25 且 斜率>0.3%
        trend_resonance = (
            detail.get('adx', 0) > 25 and 
            abs(detail.get('ma_slope', 0)) > 0.003
        )
        
        # 动量共振: RSI在极端区域 且 布林带突破
        rsi = detail.get('rsi', 50)
        bb_width = detail.get('bb_bandwidth', 0)
        momentum_resonance = (
            (rsi > 60 or rsi < 40) and bb_width > 0.10
        )
        
        # 波动率共振: 波动率扩张 且 布林带宽度增加
        vol = detail.get('volatility', 0)
        volatility_resonance = vol > 0.018 and bb_width > 0.10
        
        # 整体共振: 至少2个维度共振
        resonance_count = sum([trend_resonance, momentum_resonance, volatility_resonance])
        overall = resonance_count >= 2
        
        return {
            'trend_resonance': trend_resonance,
            'momentum_resonance': momentum_resonance,
            'volatility_resonance': volatility_resonance,
            'overall': overall
        }
