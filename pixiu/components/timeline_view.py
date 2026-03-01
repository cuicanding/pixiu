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

DIRECTION_COLORS = {
    "up": "#22c55e",
    "down": "#ef4444",
    "neutral": "#6b7280",
}


class TimelineSegment(TypedDict):
    index: int
    start: str
    end: str
    regime: str
    direction: str
    duration: int


class TurningPoint(TypedDict):
    index: int
    date: str
    from_regime: str
    to_regime: str
    to_direction: str
    triggers: Dict[str, Any]
    confidence: float
    total_score: int
    score_details: Dict[str, int]
    key_indicators: List[str]
    indicators: Dict[str, Any]
    resonance: Dict[str, bool]


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


def segment_card(segment: Dict, index: int) -> rx.Component:
    """渲染单个市场阶段卡片
    
    Args:
        segment: 包含 regime, start, end, duration 的字典
        index: 段序号（从1开始）
        
    Returns:
        Reflex 组件
    """
    regime = segment['regime']
    direction = segment.get('direction', 'neutral')
    
    # 方向箭头和颜色
    direction_icon = rx.cond(
        direction == "up", "↑",
        rx.cond(direction == "down", "↓", "→")
    )
    direction_color = rx.cond(
        direction == "up", "#22c55e",
        rx.cond(direction == "down", "#ef4444", "#6b7280")
    )
    
    return rx.box(
        rx.hstack(
            rx.vstack(
                rx.badge(
                    f"区间{index}",
                    color_scheme=rx.cond(regime == "trend", "green", "yellow"),
                    variant="outline",
                    font_size="0.7rem",
                ),
                rx.text(
                    direction_icon,
                    font_size="1.2rem",
                    color=direction_color,
                    font_weight="bold",
                ),
                spacing="1",
                align="center",
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
            spacing="3",
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


def turning_point_card(tp: Dict, index: int) -> rx.Component:
    """渲染单个转折点卡片（带序号）
    
    Args:
        tp: 包含 date, from_direction, to_direction, strength 等的字典
        index: 转折点序号（从1开始）
        
    Returns:
        Reflex 组件
    """
    to_direction = tp.get('to_direction', 'neutral')
    from_direction = tp.get('from_direction', 'neutral')
    to_strength = tp.get('to_strength', 5)
    prev_duration = tp.get('prev_duration', 0)
    reason = tp.get('reason', '')
    
    return rx.box(
        rx.vstack(
            # 第一行：序号 + 日期 + 方向变化
            rx.hstack(
                rx.badge(
                    f"T{index}",
                    color_scheme="cyan",
                    variant="solid",
                    font_size="0.875rem",
                    font_weight="bold",
                ),
                rx.text(f"{tp.get('date', '')}", font_weight="bold", font_size="0.875rem"),
                # 方向变化
                rx.text(
                    rx.cond(from_direction == "up", "上涨", "下跌"),
                    color=rx.cond(from_direction == "up", "#ef4444", "#22c55e"),
                    font_weight="bold", 
                    font_size="0.875rem"
                ),
                rx.text("→", color="#6b7280", font_size="0.875rem"),
                rx.text(
                    rx.cond(to_direction == "up", "上涨", "下跌"),
                    color=rx.cond(to_direction == "up", "#ef4444", "#22c55e"),
                    font_weight="bold", 
                    font_size="0.875rem"
                ),
                rx.spacer(),
                rx.hstack(
                    rx.text("强度:", font_size="0.7rem", color="#6b7280"),
                    rx.text(f"{to_strength}/10", font_size="0.75rem", font_weight="bold"),
                    spacing="1",
                ),
                rx.hstack(
                    rx.text("转折前:", font_size="0.7rem", color="#6b7280"),
                    rx.text(f"{prev_duration}天", font_size="0.75rem", color="#a0a0b0"),
                    spacing="1",
                ),
                spacing="2",
                align="center",
                width="100%",
            ),
            # 第二行：转折理由
            rx.hstack(
                rx.text("信号:", font_size="0.7rem", color="#6b7280"),
                rx.text(reason, font_size="0.75rem", color="#a0a0b0"),
                spacing="1",
            ),
            spacing="2",
            align="start",
        ),
        padding="0.75rem",
        border_radius="0.5rem",
        bg="#1f1f2e",
        border=rx.cond(to_direction == "up", "1px solid #ef4444", "1px solid #22c55e"),
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
    segments = timeline.get('segments', [])
    turning_points = timeline.get('turning_points', [])
    
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text("📅 择势转折点详情", font_size="1.25rem", font_weight="bold"),
                rx.spacer(),
                # 转折点数量badge
                rx.cond(
                    turning_points,
                    rx.badge(
                        rx.text("个转折点", font_size="xs"),
                        color_scheme="cyan",
                        variant="outline",
                    ),
                    rx.badge("暂无转折点", color_scheme="gray", variant="outline"),
                ),
            ),
            rx.divider(),
            
            # 第一个阶段信息（简化版，不用Python if判断）
            # 核心概念：趋势有方向，震荡无方向
            rx.cond(
                segments,
                rx.box(
                    rx.hstack(
                        rx.text("初始阶段:", font_size="0.75rem", color="#6b7280"),
                        rx.text(
                            rx.cond(
                                segments[0].get('regime') == "trend",
                                rx.cond(segments[0].get('direction') == "up", "上涨趋势",
                                       rx.cond(segments[0].get('direction') == "down", "下跌趋势", "趋势")),
                                rx.cond(segments[0].get('regime') == "range", "震荡", "未知")
                            ),
                            font_weight="bold",
                            font_size="0.875rem",
                        ),
                        rx.text("持续", font_size="0.75rem", color="#a0a0b0"),
                        rx.text(segments[0].get('duration', 0), font_size="0.75rem", color="#a0a0b0"),
                        rx.text("天", font_size="0.75rem", color="#a0a0b0"),
                        spacing="2",
                        align="center",
                    ),
                    padding="0.5rem 0.75rem",
                    bg="#1a1a24",
                    border_radius="0.5rem",
                    margin_bottom="0.5rem",
                ),
                rx.box(),
            ),
            
            # 转折点表格
            rx.cond(
                turning_points,
                rx.vstack(
                    rx.foreach(
                        turning_points,
                        lambda tp: turning_point_card(tp, tp.get('index', 1)),
                    ),
                    spacing="2",
                    width="100%",
                ),
                rx.text("暂无转折点数据", font_size="0.875rem", color="#6b7280"),
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
