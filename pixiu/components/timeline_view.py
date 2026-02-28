"""时间线择势可视化组件"""
import reflex as rx
from typing import Dict, List, Any, Optional


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
    current = timeline.get('current')
    
    current_regime = current.get('regime', 'unknown') if current else None
    current_badge = (
        rx.badge(
            REGIME_TEXT.get(current_regime, '未知'),
            color_scheme="green" if current_regime == "trend" else "yellow",
        ) if current else rx.box()
    )
    
    segment_items = []
    for seg in segments:
        regime = seg.get('regime', 'unknown')
        regime_text = REGIME_TEXT.get(regime, '未知')
        regime_color = REGIME_COLORS.get(regime, '#6b7280')
        regime_icon = "📈" if regime == "trend" else "📊" if regime == "range" else "❓"
        start_date = seg.get('start_date', '?')
        end_date = seg.get('end_date', '?')
        confidence = seg.get('confidence', 0)
        
        segment_items.append(
            rx.box(
                rx.hstack(
                    rx.box(regime_icon, font_size="1.5rem", padding_x="0.5rem"),
                    rx.vstack(
                        rx.hstack(
                            rx.text(regime_text, font_weight="bold", font_size="1rem", color=regime_color),
                            rx.text(f"{confidence:.0%}", font_size="0.75rem", color="#6b7280"),
                            spacing="2",
                            align="center",
                        ),
                        rx.text(f"{start_date} ~ {end_date}", font_size="0.75rem", color="#a0a0b0"),
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
                border_left=f"4px solid {regime_color}",
                width="100%",
            )
        )
    
    tp_items = []
    for tp in turning_points:
        date = tp.get('date', '?')
        from_regime = tp.get('from', 'unknown')
        to_regime = tp.get('to', 'unknown')
        trigger = tp.get('trigger', '未知原因')
        
        from_text = REGIME_TEXT.get(from_regime, '未知')
        to_text = REGIME_TEXT.get(to_regime, '未知')
        to_color = REGIME_COLORS.get(to_regime, '#6b7280')
        
        tp_items.append(
            rx.box(
                rx.vstack(
                    rx.hstack(
                        rx.text("⚡", font_size="1rem"),
                        rx.text(date, font_weight="bold", font_size="0.875rem"),
                        spacing="1",
                        align="center",
                    ),
                    rx.hstack(
                        rx.text(from_text, color="#6b7280", font_size="0.75rem"),
                        rx.text("→", color="#6b7280", font_size="0.75rem"),
                        rx.text(to_text, color=to_color, font_weight="bold", font_size="0.75rem"),
                        spacing="1",
                        align="center",
                    ),
                    rx.text(f"触发: {trigger}", font_size="0.7rem", color="#6b7280"),
                    spacing="1",
                    align="start",
                ),
                padding="0.75rem",
                border_radius="0.5rem",
                bg="#1f1f2e",
                border="1px solid #2a2a3a",
                width="100%",
            )
        )
    
    has_data = len(segments) > 0 or len(turning_points) > 0
    
    content = []
    if segment_items:
        content.append(
            rx.vstack(
                rx.text("市场阶段", font_size="0.875rem", color="#a0a0b0", font_weight="bold"),
                *segment_items,
                spacing="2",
                width="100%",
            )
        )
    
    if tp_items:
        content.append(
            rx.vstack(
                rx.text("转折点", font_size="0.875rem", color="#a0a0b0", font_weight="bold"),
                *tp_items,
                spacing="2",
                width="100%",
            )
        )
    
    if not has_data:
        content = [
            rx.box(
                rx.text("暂无时间线数据", color="#6b7280", font_size="0.875rem"),
                padding="2rem",
                text_align="center",
            )
        ]
    
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text("📅 市场择势时间线", font_size="1.25rem", font_weight="bold"),
                current_badge,
                justify="between",
                width="100%",
            ),
            rx.divider(),
            rx.vstack(
                *content,
                spacing="4",
                width="100%",
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
