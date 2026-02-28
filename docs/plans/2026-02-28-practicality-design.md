# Pixiu 实用性增强设计文档

**Date**: 2026-02-28
**Author**: AI Assistant
**Status**: Approved

---

## 1. 概述

本迭代目标是达到实用性，主要解决三个问题：

1. **数据获取** - 修复Baostock日期格式问题，实现混合数据源
2. **时间线择势** - 全时段分析，自动识别转势点
3. **AI全流程指导** - 三阶段AI介入，解释+评估+优化

### 优先级顺序

使用场景 > 数据获取 > 策略智能

### 实现方案

渐进式增强（3个迭代）：

- 阶段1：数据修复
- 阶段2：时间线择势
- 阶段3：AI全流程指导

---

## 2. 数据层修复

### 2.1 问题分析

当前 `_fetch_from_baostock` 的问题：

1. 日期格式可能不兼容（需要YYYYMMDD格式）
2. 登录后没有正确检查登录状态
3. 异常处理不够健壮

### 2.2 数据源选择策略

```python
def get_data_source(market: str) -> str:
    if market == "A股":
        return "baostock"  # 优先
    elif market == "index":
        return "baostock"  # 大盘指数
    else:
        return "akshare"   # 港股/美股
```

### 2.3 修复要点

| 修复点      | 原问题       | 解决方案                           |
| -------- | --------- | ------------------------------ |
| 日期格式     | 日期可能带"-"  | 统一转换为YYYYMMDD                  |
| 登录检查     | lg可能为None | 登录后检查error_code                |
| 超时处理     | 无超时机制     | 添加30s超时                        |
| Fallback | 直接用mock   | 顺序尝试：Baostock → AKShare → Mock |

### 2.4 新增功能

- **数据源状态检测**：启动时检测各数据源可用性
- **智能切换**：某数据源失败自动切换备用源

### 2.5 配置项

```python
DATA_SOURCE_PRIORITY = {
    "A股": ["baostock", "akshare", "mock"],
    "港股": ["akshare", "mock"],
    "美股": ["akshare", "mock"],
}
```

---

## 3. 时间线择势分析

### 3.1 核心概念

**滚动窗口择势**：在整个时间范围内，用60-90天的滑动窗口识别每个时间点的市场状态。

### 3.2 新增类：`RegimeTimelineAnalyzer`

```python
class RegimeTimelineAnalyzer:
    """时间线择势分析器"""

    def __init__(self, window: int = 60, 
                 adx_threshold: float = 25,
                 slope_threshold: float = 0.005):
        self.window = window
        self.adx_threshold = adx_threshold
        self.slope_threshold = slope_threshold

    def analyze_timeline(self, df: pd.DataFrame) -> RegimeTimeline:
        """
        返回整个时间线上的择势状态

        返回结构:
        {
            "segments": [
                {
                    "start_date": "2025-01-15",
                    "end_date": "2025-03-20",
                    "regime": "trend",
                    "confidence": 0.78,
                    "indicators": {"adx": 32.1, "slope": 0.008, "vol": 0.025}
                },
                ...
            ],
            "turning_points": [
                {
                    "date": "2025-03-20",
                    "from": "trend",
                    "to": "range", 
                    "trigger": "ADX跌破25 + 波动率收缩",
                    "confidence": 0.85
                },
                ...
            ],
            "current": {
                "regime": "range",
                "duration_days": 45,
                "next_turn_probability": 0.32
            }
        }
        """
```

### 3.3 转势识别算法

| 转势类型  | 触发条件                            | 置信度计算 |
| ----- | ------------------------------- | ----- |
| 趋势→震荡 | ADX从>25跌破25；MA斜率绝对值<0.005       | 多指标投票 |
| 震荡→趋势 | ADX从<25突破25；MA斜率绝对值>0.005；波动率扩张 | 多指标投票 |

### 3.4 转势点判断逻辑

```python
def detect_turning_point(self, prev_regime: str, curr_regime: str,
                         indicators: Dict) -> Optional[Dict]:
    """检测转势点"""
    if prev_regime == curr_regime:
        return None

    trigger_parts = []
    if "adx_cross" in indicators:
        trigger_parts.append(f"ADX{'突破' if curr_regime == 'trend' else '跌破'}25")
    if "slope_change" in indicators:
        trigger_parts.append("MA斜率变化")
    if "vol_change" in indicators:
        trigger_parts.append("波动率" + ("扩张" if curr_regime == 'trend' else "收缩"))

    return {
        "from": prev_regime,
        "to": curr_regime,
        "trigger": " + ".join(trigger_parts),
        "confidence": self._calc_confidence(indicators)
    }
```

### 3.5 时间线可视化

```
[2025-01] ━━━趋势━━━━━━━━━━━━━━━━━▶ [2025-03-20] ⚡转势
                                              │
[2025-03] ━━━━━━━━━震荡━━━━━━━━━━━━▶ [2025-06-15] ⚡转势
                                              │
[2025-06] ━━━━━趋势━━━━━━━━━━━━━━▶ [2025-09]
```

### 3.6 与策略关联

每个segment自动关联推荐策略：

```python
SEGMENT_STRATEGY_MAP = {
    "trend": {
        "primary": ["趋势强度策略", "均线交叉策略"],
        "secondary": ["最优执行策略"],
        "avoid": ["网格交易策略"]
    },
    "range": {
        "primary": ["网格交易策略", "RSI策略"],
        "secondary": ["波动率套利策略"],
        "avoid": ["趋势强度策略"]
    }
}
```

---

## 4. AI全流程指导

### 4.1 三阶段AI介入

```
┌─────────────────────────────────────────────────────────────────┐
│  阶段1：择势分析                                                  │
│  ├── 分析转势原因（哪个指标触发、幅度多大）                        │
│  ├── 预判后续可能的走势                                          │
│  └── 给出操作建议（等待确认/立即行动）                            │
├─────────────────────────────────────────────────────────────────┤
│  阶段2：策略推荐                                                  │
│  ├── 解释为什么推荐这些策略（结合择势结果）                        │
│  ├── 对比各策略的预期表现                                        │
│  └── 给出最佳组合建议                                            │
├─────────────────────────────────────────────────────────────────┤
│  阶段3：回测评估                                                  │
│  ├── 分析各策略实际表现 vs 预期                                  │
│  ├── 识别策略失效的时段（可能对应转势期）                          │
│  └── 给出参数优化建议                                            │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Prompt设计

#### 择势解释Prompt

```
你是专业量化分析师。请分析以下择势结果：

股票：{stock_name} ({stock_code})
时间范围：{start_date} ~ {end_date}

时间线分析：
{timeline_segments}

转势点：
{turning_points}

请回答：
1. 当前处于什么阶段？持续多久了？
2. 最近一次转势是什么触发的？
3. 预判未来可能的走势（基于历史模式）
4. 操作建议
```

#### 策略推荐Prompt

```
基于择势分析，请推荐策略：

择势结果：{regime_summary}
推荐策略列表：{recommended_strategies}
各策略特点：{strategy_descriptions}

请回答：
1. 为什么这些策略适合当前市场状态？
2. 哪个策略作为主力？哪些作为辅助？
3. 如果组合使用，建议什么权重？
4. 需要注意的风险点
```

#### 回测评估Prompt

```
请评估以下回测结果：

策略：{strategy_name}
择势状态：{regime_during_backtest}

回测指标：
- 总收益：{total_return}
- 夏普比率：{sharpe_ratio}
- 最大回撤：{max_drawdown}
- 胜率：{win_rate}

请回答：
1. 表现是否符合预期？（结合择势状态分析）
2. 哪些时段表现好/差？是否对应特定市场状态？
3. 参数优化建议
4. 是否适合实盘使用？
```

### 4.3 UI增强

- 每个阶段增加"AI解释"按钮
- 择势页面增加"时间线+AI解读"区域
- 回测结果页面增加"AI评估"区域

---

## 5. 整体架构

### 5.1 文件结构

```
pixiu/
├── pixiu/
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── regime_detector.py      # 已有
│   │   └── regime_timeline.py      # 新增：时间线择势分析
│   │
│   ├── services/
│   │   ├── data_service.py         # 修改：数据源修复
│   │   └── ai_service.py           # 增强：三阶段AI指导
│   │
│   ├── components/
│   │   ├── timeline_view.py        # 新增：时间线可视化组件
│   │   └── ai_explanation.py       # 新增：AI解释组件
│   │
│   ├── pages/
│   │   └── home.py                 # 修改：集成新功能
│   │
│   └── state.py                    # 修改：支持时间线状态
```

### 5.2 数据流

```
用户输入股票+时间范围
        ↓
DataService.fetch_history() ──→ 真实数据（Baostock/AKShare）
        ↓
RegimeTimelineAnalyzer.analyze() ──→ 时间线+转势点
        ↓
┌───────┴───────┐
│               │
▼               ▼
AI解释择势      策略推荐
        ↓
用户选择策略
        ↓
BacktestEngine.run() ──→ 回测结果
        ↓
AI评估回测 ──→ 优化建议
```

### 5.3 配置项

```python
# pixiu/config.py 新增
class Config:
    # 择势参数
    REGIME_WINDOW_DAYS = 60        # 默认窗口
    REGIME_ADX_THRESHOLD = 25      # ADX阈值
    REGIME_SLOPE_THRESHOLD = 0.005 # 斜率阈值

    # 数据源
    DATA_SOURCE_PRIORITY = {
        "A股": ["baostock", "akshare", "mock"],
        "港股": ["akshare", "mock"],
        "美股": ["akshare", "mock"],
    }
```

---

## 6. 实施阶段

### Phase 1: 数据修复

- [ ] 修复Baostock日期格式问题
- [ ] 实现数据源优先级切换
- [ ] 添加数据源状态检测
- [ ] 测试真实数据获取

### Phase 2: 时间线择势

- [ ] 实现 `RegimeTimelineAnalyzer` 类
- [ ] 实现转势点检测算法
- [ ] 创建时间线可视化组件
- [ ] 集成到现有择势流程

### Phase 3: AI全流程

- [ ] 实现三阶段AI Prompt
- [ ] 增强AI服务接口
- [ ] 添加UI上的AI解释按钮
- [ ] 测试AI响应质量

---

## 7. 成功标准

1. **数据获取成功** - A股用Baostock获取真实数据，港股/美股用AKShare
2. **时间线可视化** - 能看到整段时间的择势状态和转势点
3. **转势识别准确** - 转势点有明确触发原因说明
4. **AI解释有价值** - AI能给出有洞见的分析和建议
5. **用户体验流畅** - 整个流程一气呵成
