"""时间线择势分析"""
from typing import Dict, List, Any, TypedDict
import pandas as pd
from .regime_detector import MarketRegimeDetector


class TimelineSegment(TypedDict):
    start: str
    end: str
    regime: str
    duration: int


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
    """
    
    def __init__(
        self,
        window: int = 60,
        adx_threshold: float = 25.0,
        slope_threshold: float = 0.005
    ):
        """Initialize the timeline analyzer.
        
        Args:
            window: Rolling window size in days for regime detection
            adx_threshold: Threshold for identifying ADX-based turning point triggers
                (used in trigger identification, not regime detection)
            slope_threshold: Threshold for identifying slope-based turning point triggers
                (used in trigger identification, not regime detection)
        """
        self.window = window
        self.adx_threshold = adx_threshold
        self.slope_threshold = slope_threshold
        self.detector = MarketRegimeDetector()
    
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
        turning_points = self._detect_turning_points(df, regime_history)
        
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
        """构建滚动窗口的regime历史"""
        regime_history = []
        
        for i in range(self.window, len(df) + 1):
            window_df = df.iloc[i - self.window:i]
            regime = self.detector.detect_regime(window_df)
            detail = self.detector.get_analysis_detail(window_df)
            
            regime_history.append({
                'idx': i - 1,
                'date': df.iloc[i - 1].get('trade_date', i - 1),
                'regime': regime,
                'detail': detail
            })
        
        return regime_history
    
    def _build_segments(self, df: pd.DataFrame, regime_history: List[Dict]) -> List[Dict]:
        """将连续相同regime的点聚合成segment"""
        if not regime_history:
            return []
        
        segments = []
        current_regime = regime_history[0]['regime']
        start_idx = 0
        
        for i, item in enumerate(regime_history):
            if item['regime'] != current_regime:
                segments.append({
                    'start': regime_history[start_idx]['date'],
                    'end': regime_history[i - 1]['date'],
                    'regime': current_regime,
                    'duration': i - start_idx
                })
                current_regime = item['regime']
                start_idx = i
        
        segments.append({
            'start': regime_history[start_idx]['date'],
            'end': regime_history[-1]['date'],
            'regime': current_regime,
            'duration': len(regime_history) - start_idx
        })
        
        return segments
    
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
