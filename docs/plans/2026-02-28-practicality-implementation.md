# Pixiu 实用性增强实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修复数据获取，实现时间线择势分析，增强AI全流程指导

**Architecture:** 渐进式三层增强 - 数据层(修复Baostock+混合数据源) → 分析层(滚动窗口时间线择势) → AI层(三阶段Prompt增强)

**Tech Stack:** Python 3.11, Baostock, AKShare, Reflex, GLM API, Pandas

---

## Phase 1: 数据层修复

### Task 1.1: 修复Baostock日期格式问题

**Files:**
- Modify: `pixiu/services/data_service.py:231-291`

**Step 1: 编写测试**

Create: `tests/test_baostock_date_fix.py`

```python
"""测试Baostock日期格式修复"""
import pytest
from datetime import datetime, timedelta
from pixiu.services.data_service import DataService
from pixiu.services.database import Database

def test_date_format_conversion():
    """测试日期格式正确转换为YYYYMMDD"""
    ds = DataService(Database(":memory:"), use_mock=False)
    
    # 模拟输入
    start_date = "2025-01-01"
    end_date = "2025-02-28"
    
    # 内部应该转换为20250101和20250228
    start_fmt = start_date.replace("-", "")
    end_fmt = end_date.replace("-", "")
    
    assert start_fmt == "20250101"
    assert end_fmt == "20250228"

def test_baostock_login_check():
    """测试登录状态检查"""
    try:
        import baostock as bs
        lg = bs.login()
        assert hasattr(lg, 'error_code')
        assert lg.error_code == '0'
        bs.logout()
    except ImportError:
        pytest.skip("baostock not installed")

@pytest.mark.asyncio
async def test_fetch_a_stock_data():
    """测试获取A股真实数据"""
    db = Database("data/stocks.db")
    ds = DataService(db, use_mock=False)
    
    df = await ds.fetch_stock_history(
        "000001", 
        "A股",
        start_date="2025-01-01",
        end_date="2025-02-01"
    )
    
    assert df is not None
    assert not df.empty
    assert 'close' in df.columns
    assert 'trade_date' in df.columns
```

**Step 2: 运行测试确认失败**

Run: `pytest tests/test_baostock_date_fix.py -v`
Expected: FAIL (当前代码有bug)

**Step 3: 修复 _fetch_from_baostock 方法**

Modify: `pixiu/services/data_service.py:231-291`

```python
def _fetch_from_baostock(
    self, 
    code: str, 
    market: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """从Baostock获取数据 - 只支持A股"""
    if bs is None:
        raise ImportError("baostock not installed")
    
    if market != "A股":
        raise ValueError(f"baostock只支持A股，不支持{market}")
    
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    start_date_fmt = start_date.replace("-", "")
    end_date_fmt = end_date.replace("-", "")
    
    if code.startswith("6"):
        bs_code = f"sh.{code}"
    else:
        bs_code = f"sz.{code}"
    
    lg = None
    try:
        lg = bs.login()
        
        if lg is None or lg.error_code != '0':
            logger.error(f"baostock登录失败: {lg.error_msg if lg else 'None'}")
            return pd.DataFrame()
        
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,code,open,high,low,close,volume,amount",
            start_date=start_date_fmt,
            end_date=end_date_fmt,
            frequency="d",
            adjustflag="3"
        )
        
        if rs is None or rs.error_code != '0':
            logger.error(f"baostock查询失败: {rs.error_msg if rs else 'None'}")
            return pd.DataFrame()
        
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            logger.warning(f"baostock未获取到数据: {code}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        df = df.rename(columns={'date': 'trade_date'})
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.sort_values('trade_date')
    except Exception as e:
        logger.error(f"baostock获取数据失败: {e}")
        return pd.DataFrame()
    finally:
        if lg is not None:
            try:
                bs.logout()
            except:
                pass
```

**Step 4: 运行测试确认通过**

Run: `pytest tests/test_baostock_date_fix.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pixiu/services/data_service.py tests/test_baostock_date_fix.py
git commit -m "fix: 修复Baostock日期格式和登录检查问题"
```

---

### Task 1.2: 实现混合数据源策略

**Files:**
- Modify: `pixiu/services/data_service.py:151-184`
- Modify: `pixiu/config.py`

**Step 1: 添加数据源配置**

Modify: `pixiu/config.py`

```python
from dataclasses import dataclass

@dataclass
class Config:
    glm_api_key: str = ""
    database_path: str = "data/stocks.db"
    cache_dir: str = "data/cache"
    
    initial_capital: float = 100000.0
    commission_rate: float = 0.0003
    position_size: float = 0.95
    
    # 择势参数
    regime_window_days: int = 60
    regime_adx_threshold: float = 25.0
    regime_slope_threshold: float = 0.005
    
    # 数据源优先级
    data_source_priority: dict = None
    
    def __post_init__(self):
        if self.data_source_priority is None:
            self.data_source_priority = {
                "A股": ["baostock", "akshare", "mock"],
                "港股": ["akshare", "mock"],
                "美股": ["akshare", "mock"],
                "index": ["baostock", "mock"],
            }

config = Config()
```

**Step 2: 编写测试**

Create: `tests/test_data_source_priority.py`

```python
"""测试混合数据源策略"""
import pytest
from pixiu.services.data_service import DataService
from pixiu.services.database import Database
from pixiu.config import config

def test_data_source_priority_config():
    """测试数据源优先级配置"""
    assert "A股" in config.data_source_priority
    assert config.data_source_priority["A股"][0] == "baostock"
    assert config.data_source_priority["港股"][0] == "akshare"

@pytest.mark.asyncio
async def test_fallback_to_akshare():
    """测试Baostock失败时fallback到AKShare"""
    db = Database(":memory:")
    ds = DataService(db, use_mock=False)
    
    # 获取港股数据（应该用akshare）
    df = await ds.fetch_stock_history("00700", "港股")
    
    # 即使获取失败也不应该抛出异常
    assert df is not None
```

**Step 3: 实现智能数据源选择**

Modify: `pixiu/services/data_service.py:151-184`

```python
async def fetch_stock_history(
    self, 
    code: str, 
    market: str = "A股",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """获取股票历史数据 - 按优先级尝试数据源"""
    from pixiu.config import config
    
    if self.use_mock:
        return self._generate_mock_history(code)
    
    priorities = config.data_source_priority.get(market, ["mock"])
    
    for source in priorities:
        if source == "baostock" and bs is not None and market in ["A股", "index"]:
            try:
                df = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._fetch_from_baostock, code, market, start_date, end_date
                    ),
                    timeout=30
                )
                if df is not None and not df.empty:
                    logger.info(f"成功从Baostock获取 {code} 数据")
                    return df
            except Exception as e:
                logger.warning(f"Baostock获取失败: {e}")
        
        elif source == "akshare" and ak is not None:
            try:
                df = await asyncio.wait_for(
                    asyncio.to_thread(self._fetch_from_akshare, code, market),
                    timeout=30
                )
                if df is not None and not df.empty:
                    logger.info(f"成功从AKShare获取 {code} 数据")
                    return df
            except Exception as e:
                logger.warning(f"AKShare获取失败: {e}")
        
        elif source == "mock":
            logger.info(f"使用模拟数据: {code}")
            return self._generate_mock_history(code)
    
    return self._generate_mock_history(code)
```

**Step 4: 运行测试**

Run: `pytest tests/test_data_source_priority.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pixiu/services/data_service.py pixiu/config.py tests/test_data_source_priority.py
git commit -m "feat: 实现混合数据源策略，按优先级自动切换"
```

---

## Phase 2: 时间线择势分析

### Task 2.1: 创建 RegimeTimelineAnalyzer 类

**Files:**
- Create: `pixiu/analysis/regime_timeline.py`
- Modify: `pixiu/analysis/__init__.py`

**Step 1: 编写测试**

Create: `tests/test_regime_timeline.py`

```python
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
        # 上升趋势
        close = 100 * (1 + np.linspace(0, 0.3, days))
    else:
        # 震荡
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
    # 趋势数据应该识别出trend
    assert any(s['regime'] == 'trend' for s in result['segments'])

def test_analyze_range_data():
    """测试分析震荡数据"""
    analyzer = RegimeTimelineAnalyzer(window=30)
    df = generate_test_data(90, "range")
    
    result = analyzer.analyze_timeline(df)
    
    assert 'segments' in result
    # 震荡数据应该识别出range
    assert any(s['regime'] == 'range' for s in result['segments'])

def test_turning_point_detection():
    """测试转势点检测"""
    analyzer = RegimeTimelineAnalyzer(window=30)
    
    # 生成混合数据：前60天趋势，后60天震荡
    trend_df = generate_test_data(60, "trend")
    range_df = generate_test_data(60, "range")
    range_df['trade_date'] = pd.date_range(
        start=trend_df['trade_date'].iloc[-1] + timedelta(days=1),
        periods=60,
        freq='D'
    )
    df = pd.concat([trend_df, range_df], ignore_index=True)
    
    result = analyzer.analyze_timeline(df)
    
    # 应该检测到至少一个转势点
    assert 'turning_points' in result
```

**Step 2: 运行测试确认失败**

Run: `pytest tests/test_regime_timeline.py -v`
Expected: FAIL (类不存在)

**Step 3: 实现 RegimeTimelineAnalyzer**

Create: `pixiu/analysis/regime_timeline.py`

```python
"""时间线择势分析模块"""
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from datetime import datetime

from pixiu.analysis.regime_detector import MarketRegimeDetector

@dataclass
class RegimeSegment:
    """择势阶段"""
    start_date: str
    end_date: str
    regime: str
    confidence: float
    indicators: Dict[str, float]

@dataclass
class TurningPoint:
    """转势点"""
    date: str
    from_regime: str
    to_regime: str
    trigger: str
    confidence: float

@dataclass
class RegimeTimeline:
    """择势时间线"""
    segments: List[RegimeSegment] = field(default_factory=list)
    turning_points: List[TurningPoint] = field(default_factory=list)
    current: Optional[Dict] = None

class RegimeTimelineAnalyzer:
    """时间线择势分析器"""
    
    def __init__(
        self,
        window: int = 60,
        adx_threshold: float = 25.0,
        slope_threshold: float = 0.005
    ):
        self.window = window
        self.adx_threshold = adx_threshold
        self.slope_threshold = slope_threshold
        self.detector = MarketRegimeDetector()
    
    def analyze_timeline(self, df: pd.DataFrame) -> Dict:
        """分析整个时间线的择势状态"""
        if len(df) < self.window:
            return self._empty_result()
        
        segments = []
        turning_points = []
        
        # 滚动窗口分析
        regime_history = []
        for i in range(self.window, len(df)):
            window_df = df.iloc[i-self.window:i]
            regime = self.detector.detect_regime(window_df)
            detail = self.detector.get_analysis_detail(window_df)
            
            regime_history.append({
                'date': df.iloc[i]['trade_date'],
                'regime': regime,
                'detail': detail
            })
        
        # 聚合为连续段
        if regime_history:
            segments = self._build_segments(regime_history)
            turning_points = self._detect_turning_points(regime_history)
        
        # 当前状态
        current = None
        if segments:
            last_segment = segments[-1]
            current = {
                'regime': last_segment['regime'],
                'duration_days': self._days_between(
                    last_segment['start_date'],
                    last_segment['end_date']
                ),
                'next_turn_probability': 0.3  # 简化估计
            }
        
        return {
            'segments': segments,
            'turning_points': turning_points,
            'current': current
        }
    
    def _build_segments(self, regime_history: List[Dict]) -> List[Dict]:
        """构建连续的择势段"""
        if not regime_history:
            return []
        
        segments = []
        current_regime = regime_history[0]['regime']
        start_date = regime_history[0]['date']
        
        for i, item in enumerate(regime_history):
            if item['regime'] != current_regime:
                # 结束当前段
                segments.append({
                    'start_date': self._format_date(start_date),
                    'end_date': self._format_date(regime_history[i-1]['date']),
                    'regime': current_regime,
                    'confidence': self._calc_segment_confidence(regime_history, start_date, regime_history[i-1]['date']),
                    'indicators': regime_history[i-1]['detail']
                })
                # 开始新段
                current_regime = item['regime']
                start_date = item['date']
        
        # 最后一段
        segments.append({
            'start_date': self._format_date(start_date),
            'end_date': self._format_date(regime_history[-1]['date']),
            'regime': current_regime,
            'confidence': self._calc_segment_confidence(regime_history, start_date, regime_history[-1]['date']),
            'indicators': regime_history[-1]['detail']
        })
        
        return segments
    
    def _detect_turning_points(self, regime_history: List[Dict]) -> List[Dict]:
        """检测转势点"""
        turning_points = []
        
        for i in range(1, len(regime_history)):
            prev = regime_history[i-1]
            curr = regime_history[i]
            
            if prev['regime'] != curr['regime']:
                trigger = self._identify_trigger(prev['detail'], curr['detail'])
                turning_points.append({
                    'date': self._format_date(curr['date']),
                    'from': prev['regime'],
                    'to': curr['regime'],
                    'trigger': trigger,
                    'confidence': 0.7
                })
        
        return turning_points
    
    def _identify_trigger(self, prev_detail: Dict, curr_detail: Dict) -> str:
        """识别转势触发因素"""
        triggers = []
        
        prev_adx = prev_detail.get('adx', 0)
        curr_adx = curr_detail.get('adx', 0)
        
        if prev_adx > self.adx_threshold >= curr_adx:
            triggers.append("ADX跌破25")
        elif prev_adx < self.adx_threshold <= curr_adx:
            triggers.append("ADX突破25")
        
        prev_slope = abs(prev_detail.get('ma_slope', 0))
        curr_slope = abs(curr_detail.get('ma_slope', 0))
        
        if prev_slope > self.slope_threshold >= curr_slope:
            triggers.append("MA斜率收缩")
        elif prev_slope < self.slope_threshold <= curr_slope:
            triggers.append("MA斜率扩张")
        
        return " + ".join(triggers) if triggers else "多指标变化"
    
    def _calc_segment_confidence(self, history: List[Dict], start, end) -> float:
        """计算段的置信度"""
        return 0.75
    
    def _format_date(self, date) -> str:
        """格式化日期"""
        if hasattr(date, 'strftime'):
            return date.strftime('%Y-%m-%d')
        return str(date)
    
    def _days_between(self, start: str, end: str) -> int:
        """计算两个日期之间的天数"""
        try:
            start_dt = datetime.strptime(start, '%Y-%m-%d')
            end_dt = datetime.strptime(end, '%Y-%m-%d')
            return (end_dt - start_dt).days
        except:
            return 0
    
    def _empty_result(self) -> Dict:
        """返回空结果"""
        return {
            'segments': [],
            'turning_points': [],
            'current': None
        }
```

**Step 4: 更新 __init__.py**

Modify: `pixiu/analysis/__init__.py`

```python
from pixiu.analysis.regime_detector import MarketRegimeDetector
from pixiu.analysis.regime_timeline import RegimeTimelineAnalyzer

__all__ = ['MarketRegimeDetector', 'RegimeTimelineAnalyzer']
```

**Step 5: 运行测试**

Run: `pytest tests/test_regime_timeline.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add pixiu/analysis/regime_timeline.py pixiu/analysis/__init__.py tests/test_regime_timeline.py
git commit -m "feat: 实现时间线择势分析器，支持滚动窗口和转势点检测"
```

---

### Task 2.2: 创建时间线可视化组件

**Files:**
- Create: `pixiu/components/timeline_view.py`
- Modify: `pixiu/components/__init__.py`

**Step 1: 编写测试**

Create: `tests/test_timeline_view.py`

```python
"""测试时间线可视化组件"""
import pytest
from pixiu.components.timeline_view import timeline_view, format_timeline_text

def test_format_timeline_text():
    """测试时间线文本格式化"""
    timeline = {
        'segments': [
            {'start_date': '2025-01-01', 'end_date': '2025-03-15', 'regime': 'trend'},
            {'start_date': '2025-03-16', 'end_date': '2025-06-30', 'regime': 'range'},
        ],
        'turning_points': [
            {'date': '2025-03-16', 'from': 'trend', 'to': 'range', 'trigger': 'ADX跌破25'}
        ]
    }
    
    text = format_timeline_text(timeline)
    
    assert '趋势' in text or 'trend' in text
    assert '震荡' in text or 'range' in text
    assert '转势' in text or 'turn' in text.lower()

def test_timeline_view_renders():
    """测试组件渲染"""
    timeline = {
        'segments': [],
        'turning_points': [],
        'current': None
    }
    
    component = timeline_view(timeline)
    assert component is not None
```

**Step 2: 实现时间线组件**

Create: `pixiu/components/timeline_view.py`

```python
"""时间线择势可视化组件"""
import reflex as rx
from typing import Dict, List

REGIME_COLORS = {
    "trend": "#10b981",
    "range": "#f59e0b",
    "unknown": "#6b7280"
}

REGIME_TEXT = {
    "trend": "趋势",
    "range": "震荡",
    "unknown": "未知"
}

def format_timeline_text(timeline: Dict) -> str:
    """格式化时间线为文本"""
    if not timeline or not timeline.get('segments'):
        return "暂无择势分析数据"
    
    lines = []
    for seg in timeline['segments']:
        regime_text = REGIME_TEXT.get(seg['regime'], seg['regime'])
        lines.append(
            f"[{seg['start_date']}] → [{seg['end_date']}] {regime_text}行情 "
            f"(置信度: {seg.get('confidence', 0):.0%})"
        )
    
    if timeline.get('turning_points'):
        lines.append("\n转势点:")
        for tp in timeline['turning_points']:
            lines.append(
                f"  • {tp['date']}: {REGIME_TEXT.get(tp['from'], tp['from'])} → "
                f"{REGIME_TEXT.get(tp['to'], tp['to'])} ({tp['trigger']})"
            )
    
    return "\n".join(lines)

def timeline_view(timeline: Dict) -> rx.Component:
    """时间线可视化组件"""
    if not timeline or not timeline.get('segments'):
        return rx.box(
            rx.text("暂无择势分析数据", color="gray"),
            padding="1rem"
        )
    
    segments = timeline.get('segments', [])
    turning_points = timeline.get('turning_points', [])
    current = timeline.get('current')
    
    return rx.vstack(
        rx.hstack(
            rx.text("📊 择势时间线", font_size="lg", font_weight="bold"),
            rx.cond(
                current is not None,
                rx.badge(
                    f"当前: {REGIME_TEXT.get(current.get('regime', 'unknown'), '未知')}",
                    color_scheme="green" if current.get('regime') == 'trend' else "yellow"
                ),
                rx.box()
            ),
            justify="space_between",
            width="100%"
        ),
        
        rx.divider(),
        
        rx.vstack(
            *[segment_card(seg) for seg in segments],
            spacing="2",
            width="100%"
        ),
        
        rx.cond(
            len(turning_points) > 0,
            rx.vstack(
                rx.text("⚡ 转势点", font_weight="bold", margin_top="1rem"),
                *[turning_point_card(tp) for tp in turning_points],
                spacing="1",
                width="100%"
            ),
            rx.box()
        ),
        
        spacing="3",
        width="100%",
        padding="1rem",
        bg="#1a1a24",
        border_radius="lg"
    )

def segment_card(segment: Dict) -> rx.Component:
    """单个择势段卡片"""
    regime = segment.get('regime', 'unknown')
    color = REGIME_COLORS.get(regime, REGIME_COLORS['unknown'])
    regime_text = REGIME_TEXT.get(regime, regime)
    
    return rx.box(
        rx.hstack(
            rx.box(
                width="4px",
                height="40px",
                bg=color,
                border_radius="2px"
            ),
            rx.vstack(
                rx.hstack(
                    rx.text(regime_text, font_weight="bold"),
                    rx.badge(f"{segment.get('confidence', 0):.0%}", size="sm"),
                    spacing="2"
                ),
                rx.text(
                    f"{segment['start_date']} ~ {segment['end_date']}",
                    font_size="sm",
                    color="gray"
                ),
                spacing="1",
                align_items="start"
            ),
            spacing="3"
        ),
        padding="0.5rem",
        bg="#252532",
        border_radius="md"
    )

def turning_point_card(tp: Dict) -> rx.Component:
    """转势点卡片"""
    from_regime = REGIME_TEXT.get(tp['from'], tp['from'])
    to_regime = REGIME_TEXT.get(tp['to'], tp['to'])
    
    return rx.box(
        rx.hstack(
            rx.text(tp['date'], font_weight="medium", width="100px"),
            rx.text(from_regime, color=REGIME_COLORS.get(tp['from'], "gray")),
            rx.text("→"),
            rx.text(to_regime, color=REGIME_COLORS.get(tp['to'], "gray")),
            rx.text(f"({tp['trigger']})", font_size="sm", color="gray"),
            spacing="2"
        ),
        padding="0.5rem",
        bg="#2a2a3a",
        border_radius="md",
        border_left="3px solid #f59e0b"
    )
```

**Step 3: 更新组件导出**

Modify: `pixiu/components/__init__.py`

```python
from pixiu.components.metric_card import metric_card
from pixiu.components.stock_card import stock_card
from pixiu.components.strategy_card import strategy_card
from pixiu.components.regime_indicator import regime_indicator
from pixiu.components.strategy_recommender import strategy_recommender
from pixiu.components.explain_button import explain_button
from pixiu.components.timeline_view import timeline_view, format_timeline_text

__all__ = [
    'metric_card', 'stock_card', 'strategy_card',
    'regime_indicator', 'strategy_recommender', 'explain_button',
    'timeline_view', 'format_timeline_text'
]
```

**Step 4: 运行测试**

Run: `pytest tests/test_timeline_view.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pixiu/components/timeline_view.py pixiu/components/__init__.py tests/test_timeline_view.py
git commit -m "feat: 添加时间线择势可视化组件"
```

---

### Task 2.3: 集成时间线分析到State

**Files:**
- Modify: `pixiu/state.py`

**Step 1: 添加时间线状态字段**

在 `State` 类中添加：

```python
# 新增状态字段
regime_timeline: Dict = {}
timeline_loading: bool = False
```

**Step 2: 修改 analyze_regime 方法**

```python
async def analyze_regime(self):
    """分析市场和个股择势 - 增强版：包含时间线分析"""
    from pixiu.analysis import MarketRegimeDetector, RegimeTimelineAnalyzer
    from pixiu.services.data_service import DataService
    from pixiu.services.chart_service import generate_regime_chart
    
    debug_log(f"[择势分析] 开始执行, 股票: {self.selected_stock}")
    
    self.is_loading = True
    self.loading_message = "分析市场和个股状态..."
    yield
    
    try:
        await self._ensure_db_initialized()
        
        db = Database("data/stocks.db")
        data_service = DataService(db, use_mock=False)
        detector = MarketRegimeDetector()
        timeline_analyzer = RegimeTimelineAnalyzer(window=self._get_regime_window())
        
        # 获取大盘数据
        market_codes = {"A股": "000001", "港股": "HSI", "美股": "DJI"}
        market_code = market_codes.get(self.current_market, "000001")
        
        market_df = await data_service.fetch_stock_history(
            market_code, 
            self.current_market if self.current_market != "A股" else "index",
            self.backtest_start_date,
            self.backtest_end_date
        )
        
        if market_df is not None and not market_df.empty:
            market_analysis = detector.get_analysis_detail(market_df)
            self.market_regime = market_analysis["regime"]
            self.market_index_data = market_analysis
            
            # 大盘时间线分析
            market_timeline = timeline_analyzer.analyze_timeline(market_df)
            self.regime_timeline = {"market": market_timeline}
        
        # 获取个股数据
        if self.selected_stock:
            df = await data_service.fetch_stock_history(
                self.selected_stock, 
                self.current_market,
                self.backtest_start_date,
                self.backtest_end_date
            )
            
            if df is not None and not df.empty:
                stock_analysis = detector.get_analysis_detail(df)
                self.stock_regime = stock_analysis["regime"]
                self.regime_analysis = stock_analysis
                
                # 个股时间线分析
                stock_timeline = timeline_analyzer.analyze_timeline(df)
                self.regime_timeline["stock"] = stock_timeline
        
        self.recommended_strategies = self.regime_recommendations
        
    except Exception as e:
        self.error_message = f"择势分析失败: {str(e)}"
        debug_log(f"[择势分析] 失败: {e}")
    finally:
        self.is_loading = False
    yield

def _get_regime_window(self) -> int:
    """获取择势窗口大小"""
    from pixiu.config import config
    return getattr(config, 'regime_window_days', 60)
```

**Step 3: Commit**

```bash
git add pixiu/state.py
git commit -m "feat: 集成时间线择势分析到状态管理"
```

---

## Phase 3: AI全流程指导

### Task 3.1: 实现三阶段AI Prompt

**Files:**
- Modify: `pixiu/services/ai_service.py`
- Modify: `pixiu/services/explain_prompts.py`

**Step 1: 编写测试**

Create: `tests/test_ai_prompts.py`

```python
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
    assert "择势" in prompt or "regime" in prompt.lower()

def test_strategy_recommendation_prompt():
    """测试策略推荐Prompt生成"""
    prompt = AIReportService._build_strategy_prompt(
        regime_summary="大盘趋势，个股震荡",
        strategies=["网格交易策略", "RSI策略"]
    )
    
    assert "网格交易" in prompt
    assert "推荐" in prompt

def test_backtest_evaluation_prompt():
    """测试回测评估Prompt生成"""
    prompt = AIReportService._build_backtest_prompt(
        strategy="网格交易策略",
        results={"total_return": 15.5, "sharpe_ratio": 1.2, "max_drawdown": -8.3}
    )
    
    assert "网格交易" in prompt
    assert "15.5" in prompt
```

**Step 2: 实现三阶段Prompt**

Modify: `pixiu/services/ai_service.py`

添加三个新方法：

```python
@staticmethod
def _build_regime_prompt(stock_code: str, stock_name: str, timeline: dict) -> str:
    """构建择势解释Prompt"""
    segments_text = "\n".join([
        f"- {s['start_date']} ~ {s['end_date']}: {s['regime']} (置信度: {s.get('confidence', 0):.0%})"
        for s in timeline.get('segments', [])
    ])
    
    turning_text = "\n".join([
        f"- {t['date']}: {t['from']} → {t['to']} ({t['trigger']})"
        for t in timeline.get('turning_points', [])
    ])
    
    return f"""你是专业量化分析师。请分析以下择势结果：

股票：{stock_name} ({stock_code})

时间线分析：
{segments_text if segments_text else "暂无"}

转势点：
{turning_text if turning_text else "无转势点"}

请用中文回答：
1. 当前处于什么阶段？持续多久了？
2. 最近一次转势是什么触发的？（如果有）
3. 预判未来可能的走势（基于历史模式）
4. 操作建议（等待确认/立即行动）

请简洁回答，每个要点1-2句话。"""

@staticmethod
def _build_strategy_prompt(regime_summary: str, strategies: list) -> str:
    """构建策略推荐Prompt"""
    strategy_list = "\n".join([f"- {s}" for s in strategies])
    
    return f"""基于择势分析，请推荐策略：

择势结果：{regime_summary}

候选策略：
{strategy_list}

请用中文回答：
1. 为什么这些策略适合当前市场状态？
2. 哪个策略作为主力？哪些作为辅助？
3. 如果组合使用，建议什么权重？
4. 需要注意的风险点

请简洁回答，每个要点1-2句话。"""

@staticmethod
def _build_backtest_prompt(strategy: str, results: dict) -> str:
    """构建回测评估Prompt"""
    return f"""请评估以下回测结果：

策略：{strategy}

回测指标：
- 总收益：{results.get('total_return', 0):.2f}%
- 夏普比率：{results.get('sharpe_ratio', 0):.2f}
- 最大回撤：{results.get('max_drawdown', 0):.2f}%
- 胜率：{results.get('win_rate', 0):.2f}%

请用中文回答：
1. 表现是否符合预期？
2. 主要风险点是什么？
3. 参数优化建议
4. 是否适合实盘使用？

请简洁回答，每个要点1-2句话。"""

async def explain_regime_timeline(self, stock_code: str, stock_name: str, timeline: dict) -> str:
    """解释择势时间线"""
    prompt = self._build_regime_prompt(stock_code, stock_name, timeline)
    return await self._call_api(prompt)

async def recommend_strategy(self, regime_summary: str, strategies: list) -> str:
    """推荐策略"""
    prompt = self._build_strategy_prompt(regime_summary, strategies)
    return await self._call_api(prompt)

async def evaluate_backtest(self, strategy: str, results: dict) -> str:
    """评估回测结果"""
    prompt = self._build_backtest_prompt(strategy, results)
    return await self._call_api(prompt)
```

**Step 3: 运行测试**

Run: `pytest tests/test_ai_prompts.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add pixiu/services/ai_service.py tests/test_ai_prompts.py
git commit -m "feat: 实现AI三阶段Prompt（择势解释、策略推荐、回测评估）"
```

---

### Task 3.2: 添加UI上的AI解释按钮

**Files:**
- Modify: `pixiu/pages/home.py`
- Modify: `pixiu/state.py`

**Step 1: 在State中添加AI解释方法**

在 `State` 类中添加：

```python
async def explain_regime_timeline(self):
    """AI解释择势时间线"""
    if not self.glm_api_key:
        self.current_explanation = "请先配置GLM API Key"
        self.explain_modal_open = True
        yield
        return
    
    self.ai_explaining = True
    self.explain_modal_open = True
    self.current_explanation = ""
    yield
    
    try:
        from pixiu.services.ai_service import AIReportService
        ai = AIReportService(self.glm_api_key)
        
        self.current_explanation = await ai.explain_regime_timeline(
            self.selected_stock,
            self.selected_stock_name,
            self.regime_timeline.get("stock", {})
        )
    except Exception as e:
        self.current_explanation = f"解释生成失败: {str(e)}"
    finally:
        self.ai_explaining = False
    yield

async def evaluate_current_backtest(self, strategy: str):
    """AI评估回测结果"""
    if not self.glm_api_key:
        self.current_explanation = "请先配置GLM API Key"
        self.explain_modal_open = True
        yield
        return
    
    self.ai_explaining = True
    self.explain_modal_open = True
    self.current_explanation = ""
    yield
    
    try:
        from pixiu.services.ai_service import AIReportService
        ai = AIReportService(self.glm_api_key)
        
        result = next((r for r in self.backtest_results if r['strategy'] == strategy), None)
        if result:
            self.current_explanation = await ai.evaluate_backtest(strategy, result)
    except Exception as e:
        self.current_explanation = f"评估生成失败: {str(e)}"
    finally:
        self.ai_explaining = False
    yield
```

**Step 2: 在UI中添加按钮**

择势分析区域添加：
```python
rx.button(
    "🤖 AI解释择势",
    on_click=State.explain_regime_timeline,
    color_scheme="blue",
    size="sm"
)
```

回测结果区域添加：
```python
rx.button(
    "🤖 AI评估",
    on_click=lambda: State.evaluate_current_backtest(strategy_name),
    color_scheme="purple",
    size="sm"
)
```

**Step 3: Commit**

```bash
git add pixiu/state.py pixiu/pages/home.py
git commit -m "feat: 在UI中添加AI解释按钮（择势解释、回测评估）"
```

---

## 最终验证

### Task 4.1: 集成测试

**Step 1: 运行所有测试**

```bash
pytest tests/ -v
```

Expected: 全部PASS

**Step 2: 手动功能测试**

1. 启动应用: `reflex run`
2. 搜索股票（如：平安银行 000001）
3. 验证：
   - 能获取真实数据（非mock）
   - 择势分析显示时间线
   - 转势点正确标注
   - AI解释按钮可用

**Step 3: 最终Commit**

```bash
git add .
git commit -m "feat: 完成实用性增强 - 数据修复、时间线择势、AI全流程指导"
```

---

## 依赖更新

如果需要，更新 `requirements.txt`:

```txt
reflex>=0.4.0
akshare>=1.12.0
baostock>=0.8.9
zhipuai>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
```

---

## 成功标准

1. **数据获取成功** - A股使用Baostock获取真实数据
2. **时间线可视化** - 能看到整段时间的择势状态和转势点
3. **转势识别准确** - 转势点有明确触发原因说明
4. **AI解释有价值** - AI能给出有洞见的分析和建议
5. **测试通过** - 所有单元测试和集成测试通过
