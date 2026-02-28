"""时间线择势可视化组件"""
import reflex as rx
from typing import Dict, List, Any, Optional, TypedDict


REGIME_COLORS = {
    "trend": "#10b981",
    "range": "#f59e0b",
    "unknown": "#6b7280",
}

REGIME_TEXT = {
    "trend": "趋势",
    "range": "震荡",
    "unknown": "未知",
}


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


class RegimeTimeline(TypedDict, total=False):
    segments: List[TimelineSegment]
    turning_points: List[TurningPoint]
    current: Dict[str, Any]


def format_timeline_text(timeline: Dict[str, Any]) -> str:
    """格式化时间线为可读文本
    
    Args:
        timeline: 包含 segments 和 turning_points 的字典
        
    Returns:
        格式化的文本字符串
    """
    segments = timeline.get('segments', [])
    turning_points = timeline.get('turning_points', [])
    
    if not segments and not turning_points:
        return "暂无时间线数据"
    
    lines = []
    
    if segments:
        lines.append("=== 市场阶段 ===")
        for seg in segments:
            regime = seg.get('regime', 'unknown')
            regime_text = REGIME_TEXT.get(regime, '未知')
            start = seg.get('start_date', '?')
            end = seg.get('end_date', '?')
            conf = seg.get('confidence', 0)
            lines.append(f"{start} ~ {end}: {regime_text}行情 (置信度: {conf:.0%})")
    
    if turning_points:
        lines.append("\n=== 转折点 ===")
        for tp in turning_points:
            date = tp.get('date', '?')
            from_regime = REGIME_TEXT.get(tp.get('from', 'unknown'), '未知')
            to_regime = REGIME_TEXT.get(tp.get('to', 'unknown'), '未知')
            trigger = tp.get('trigger', '未知原因')
            lines.append(f"{date}: {from_regime} → {to_regime} ({trigger})")
    
    return '\n'.join(lines)


def segment_card(segment: Dict) -> rx.Component:
    """渲染单个市场阶段卡片
    
    Args:
        segment: 包含 regime, start, end, duration 的字典
        
    Returns:
        Reflex 组件
    """
    regime = segment['regime']
    
    return rx.box(
        rx.hstack(
            rx.text(
                rx.cond(regime == "trend", "📈", 
                    rx.cond(regime == "range", "📊", "❓")
                ),
                font_size="1.5rem",
                padding_x="0.5rem"
            ),
            rx.vstack(
                rx.hstack(
                    rx.text(
                        rx.cond(regime == "trend", "趋势",
                            rx.cond(regime == "range", "震荡", "未知")
                        ),
                        font_weight="bold",
                        font_size="1rem",
                        color=rx.cond(regime == "trend", "#10b981",
                            rx.cond(regime == "range", "#f59e0b", "#6b7280")
                        )
                    ),
                    rx.text(f"{segment['duration']}天", font_size="0.75rem", color="#6b7280"),
                    spacing="2",
                    align="center",
                ),
                rx.text(f"{segment['start']} ~ {segment['end']}", font_size="0.75rem", color="#a0a0b0"),
                spacing="1",
                align="start",
            ),
            spacing="2",
            align="center",
            width="100%",
        ),
        padding="0.75rem",
        border_radius="0.5rem",
        bg="#1a1a24",
        border_left=rx.cond(regime == "trend", "4px solid #10b981",
            rx.cond(regime == "range", "4px solid #f59e0b", "4px solid #6b7280")
        ),
        width="100%",
    )


def turning_point_card(tp: Dict) -> rx.Component:
    """渲染单个转折点卡片
    
    Args:
        tp: 包含 date, from_regime, to_regime, triggers 的字典
        
    Returns:
        Reflex 组件
    """
    to_regime = tp['to_regime']
    triggers = tp['triggers']
    
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text("⚡", font_size="1rem"),
                rx.text(f"{tp['date']}", font_weight="bold", font_size="0.875rem"),
                spacing="1",
                align="center",
            ),
            rx.hstack(
                rx.text(
                    rx.cond(tp['from_regime'] == "trend", "趋势",
                        rx.cond(tp['from_regime'] == "range", "震荡", "未知")
                    ),
                    color="#6b7280",
                    font_size="0.75rem"
                ),
                rx.text("→", color="#6b7280", font_size="0.75rem"),
                rx.text(
                    rx.cond(to_regime == "trend", "趋势",
                        rx.cond(to_regime == "range", "震荡", "未知")
                    ),
                    color=rx.cond(to_regime == "trend", "#10b981",
                        rx.cond(to_regime == "range", "#f59e0b", "#6b7280")
                    ),
                    font_weight="bold",
                    font_size="0.75rem"
                ),
                spacing="1",
                align="center",
            ),
            rx.hstack(
                rx.text("触发: ", font_size="0.7rem", color="#6b7280"),
                rx.text(
                    rx.cond(triggers['adx_cross_up'], "ADX突破25",
                        rx.cond(triggers['adx_cross_down'], "ADX跌破25",
                            rx.cond(triggers['slope_increase'], "斜率增大",
                                rx.cond(triggers['slope_decrease'], "斜率减小", "市场结构变化")
                            )
                        )
                    ),
                    font_size="0.7rem",
                    color="#6b7280"
                ),
                spacing="0",
            ),
            spacing="1",
            align="start",
        ),
        padding="0.75rem",
        border_radius="0.5rem",
        bg="#1f1f2e",
        border="1px solid #2a2a3a",
        width="100%",
    )


def timeline_view(timeline: Dict[str, Any]) -> rx.Component:
    """时间线择势可视化主组件
    
    Args:
        timeline: 包含以下字段的字典:
            - segments: 市场阶段列表
            - turning_points: 转折点列表
            - current: 当前状态 (可选)
            
    Returns:
        Reflex 组件
    """
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text("📅 市场择势时间线", font_size="1.25rem", font_weight="bold"),
                rx.spacer(),
            ),
            rx.divider(),
            
            # 市场阶段
            rx.vstack(
                rx.text("市场阶段", font_size="0.875rem", color="#a0a0b0", font_weight="bold"),
                rx.foreach(
                    timeline["segments"],
                    segment_card,
                ),
                spacing="2",
                width="100%",
            ),
            
            # 转折点
            rx.vstack(
                rx.text("转折点", font_size="0.875rem", color="#a0a0b0", font_weight="bold"),
                rx.foreach(
                    timeline["turning_points"],
                    turning_point_card,
                ),
                spacing="2",
                width="100%",
                margin_top="1rem",
            ),
            
            spacing="4",
            width="100%",
        ),
        padding="1rem",
        border_radius="0.75rem",
        bg="#12121a",
        border="1px solid #2a2a3a",
        width="100%",
    )
