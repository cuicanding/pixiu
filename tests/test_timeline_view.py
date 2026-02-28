"""测试时间线可视化组件"""
import pytest
from pixiu.components.timeline_view import timeline_view, format_timeline_text


def test_format_timeline_text():
    """测试时间线文本格式化"""
    timeline = {
        'segments': [
            {'start_date': '2025-01-01', 'end_date': '2025-03-15', 'regime': 'trend', 'confidence': 0.8},
            {'start_date': '2025-03-16', 'end_date': '2025-06-30', 'regime': 'range', 'confidence': 0.7},
        ],
        'turning_points': [
            {'date': '2025-03-16', 'from': 'trend', 'to': 'range', 'trigger': 'ADX跌破25'}
        ]
    }
    
    text = format_timeline_text(timeline)
    
    assert '趋势' in text or 'trend' in text.lower()
    assert '震荡' in text or 'range' in text.lower()


def test_format_timeline_text_empty():
    """测试空时间线"""
    timeline = {'segments': [], 'turning_points': []}
    text = format_timeline_text(timeline)
    assert '暂无' in text or '无' in text


def test_timeline_view_renders():
    """测试组件渲染"""
    timeline = {
        'segments': [],
        'turning_points': [],
        'current': None
    }
    
    component = timeline_view(timeline)
    assert component is not None
