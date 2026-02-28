# 策略规则发现系统设计文档

**Date**: 2026-03-01
**Status**: Approved

---

## 1. 概述

### 目标

构建一个自动化的策略规则发现系统：

1. 选择股票后，自动匹配对应大盘指数
2. 进行大盘+个股双时间线择势分析
3. 根据择势组合，AI推荐候选策略
4. 批量回测找出最佳策略，形成规则
5. 规则保存到SQLite，支持导出
6. 用验证集验证规则有效性

### 核心价值

发现并保存规则：**"大盘震荡+个股趋势涨 → 布朗运动策略"**

### 架构方案

单页流程式：输入 → 择势 → 发现 → 验证 → 规则库

---

## 2. 整体流程

```
输入: 股票代码 + 时间范围
  ↓
自动匹配大盘指数 (创业板→创业板指, 科创板→科创50, 蓝筹→沪深300, 其他→上证/深证)
  ↓
阶段1: 双时间线择势分析
  ├── 大盘择势时间线 (60日滚动窗口)
  ├── 个股择势时间线 (60日滚动窗口)
  └── 生成择势组合序列: [{日期, 大盘状态, 个股状态}, ...]
  ↓
阶段2: 规则发现 (训练集 80%)
  ├── 按择势组合分组时间段
  ├── AI推荐候选策略 (3-5个)
  ├── 批量回测每个策略
  └── 记录: {择势组合, 策略, 收益率} → 保存最优
  ↓
阶段3: 规则验证 (验证集 20%)
  ├── 用已保存规则指导策略选择
  ├── 回测验证集
  └── 对比: 规则指导 vs 基准收益
```

### 数据划分

- 训练集: 80% (时间顺序前段)
- 验证集: 20% (时间顺序后段)

---

## 3. 择势组合与规则存储

### 择势组合定义

```python
RegimeCombo = {
    "market": "trend_up" | "trend_down" | "range",  # 大盘状态
    "stock": "trend_up" | "trend_down" | "range"    # 个股状态
}
```

共 3×3 = 9种组合

### SQLite 存储结构

```sql
CREATE TABLE strategy_rules (
    id INTEGER PRIMARY KEY,
    market_regime TEXT NOT NULL,      -- 大盘状态
    stock_regime TEXT NOT NULL,       -- 个股状态
    best_strategy TEXT NOT NULL,      -- 最佳策略名
    train_return REAL,                -- 训练集收益率
    valid_return REAL,                -- 验证集收益率
    baseline_return REAL,             -- 基准收益率
    confidence REAL,                  -- 置信度
    train_period TEXT,                -- 训练时间段
    stock_code TEXT,                  -- 来源股票
    created_at TIMESTAMP
);

CREATE TABLE backtest_history (
    id INTEGER PRIMARY KEY,
    rule_id INTEGER,
    strategy_name TEXT,
    total_return REAL,
    sharpe REAL,
    max_drawdown REAL,
    FOREIGN KEY (rule_id) REFERENCES strategy_rules(id)
);
```

### 规则去重逻辑

同一择势组合，多次发现时：
- 新规则置信度更高 → 覆盖旧规则
- 置信度相近 → 保留收益更高的

### 功能

- 展示: 列表展示所有规则
- 导出: JSON/CSV格式导出

---

## 4. 图表可视化 (ECharts)

### 主图表布局

```
┌────────────────────────────────────────────────────────────────────┐
│  价格 ─┬──────────────────────────────────────────────────────     │
│        │   ╭─绿─╮      ╭─黄───╮      ╭─绿─────╮                  │
│        │───│趋势│──────│震荡  │──────│趋势    │───               │
│        │   ╰───╯      ╰──────╯      ╰────────╯                  │
│        │        ⚡           ⚡                                     │
│        │     转折点1      转折点2                                  │
│        └──────────────────────────────────────────────────        │
│              2024-01       2024-06       2024-12    2025-03       │
│                                                                    │
│   [图例] ■ 趋势涨(绿) ■ 趋势跌(红) ■ 震荡(黄) ⚡ 转折点             │
└────────────────────────────────────────────────────────────────────┘
```

### 交互功能

1. **区域缩放** - 鼠标框选放大某时段
2. **转折点悬停** - 显示触发原因
3. **时间线联动** - 上方大盘图，下方个股图，时间轴对齐
4. **策略标注** - 每个segment显示当时使用的策略

### 颜色编码

- 趋势涨: 绿色
- 趋势跌: 红色
- 震荡: 黄色
- 转折点: ⚡ 标记

### ECharts集成

通过Reflex的`rx.html`嵌入ECharts实例，或使用第三方`reflex-echarts`库。

---

## 5. AI策略推荐逻辑

### 推荐流程

```
择势组合 → WebSearch常识 → GLM AI精筛 → 3-5个候选策略 → 批量回测
```

### Prompt设计

```python
STRATEGY_RECOMMEND_PROMPT = """
你是量化策略专家。根据以下市场择势状态，推荐最合适的交易策略。

当前择势状态：
- 大盘: {market_regime} (趋势涨/趋势跌/震荡)
- 个股: {stock_regime} (趋势涨/趋势跌/震荡)

可用策略列表：
{available_strategies}

请回答：
1. 推荐哪些策略（3-5个），按优先级排序
2. 每个策略为什么适合当前择势组合
3. 预期风险点和注意事项

以JSON格式返回：
{
  "recommended": ["策略1", "策略2", ...],
  "reasons": {"策略1": "原因", ...},
  "warnings": ["风险点1", ...]
}
"""
```

### 缓存机制

同一择势组合的推荐结果缓存24小时

### Fallback

如果GLM调用失败：
1. 使用内置常识规则
2. 或直接测试所有策略

---

## 6. 页面布局

### 单页五区域

```
┌─────────────────────────────────────────────────────────────────┐
│  区域1: 输入区 (顶部固定)                                         │
│  [股票代码] [开始日期] [结束日期] [训练比例80%▼] [开始分析]        │
│  自动匹配大盘: 创业板指 (399006)                                   │
├─────────────────────────────────────────────────────────────────┤
│  区域2: 择势时间线 (ECharts, 可缩放)                              │
│  大盘K线 + 择势色带 + 转折点标注                                   │
│  个股K线 + 择势色带 + 转折点标注                                   │
├─────────────────────────────────────────────────────────────────┤
│  区域3: 规则发现结果 (训练集)                                      │
│  择势组合 | 最佳策略 | 收益率 | 置信度                             │
│  [导出规则] [查看详情]                                             │
├─────────────────────────────────────────────────────────────────┤
│  区域4: 规则验证结果 (验证集)                                      │
│  规则指导收益 | 基准收益 | 超额收益                                 │
│  [查看交易记录]                                                    │
├─────────────────────────────────────────────────────────────────┤
│  区域5: 已保存规则库 (底部可折叠)                                   │
│  [展开] 历史发现的规则 (共N条)  [导出全部]                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. 文件结构

### 新增文件

```
pixiu/
├── pixiu/
│   ├── analysis/
│   │   ├── regime_timeline.py        # 已有
│   │   └── regime_matcher.py         # 新增: 智能匹配大盘指数
│   │
│   ├── services/
│   │   ├── data_service.py           # 已有
│   │   ├── rule_discovery_service.py # 新增: 规则发现服务
│   │   ├── rule_storage.py           # 新增: SQLite规则存储
│   │   └── strategy_recommender.py   # 新增: AI策略推荐
│   │
│   ├── components/
│   │   ├── echarts_kline.py          # 新增: ECharts K线组件
│   │   ├── rule_table.py             # 新增: 规则表格组件
│   │   └── timeline_view.py          # 已有，增强
│   │
│   ├── pages/
│   │   ├── home.py                   # 修改: 集成新流程
│   │   └── rules.py                  # 新增: 规则库管理页
│   │
│   └── models/
│       └── rules.py                  # 新增: 规则数据模型
│
├── data/
│   └── pixiu_rules.db                # 新增: SQLite数据库
```

### 核心类

```python
# regime_matcher.py
def match_index(stock_code: str) -> str:
    """根据股票代码智能匹配大盘指数"""

# rule_discovery_service.py
class RuleDiscoveryService:
    def discover_rules(self, stock_df, index_df, train_ratio=0.8) -> List[Rule]
    def validate_rules(self, rules, valid_df) -> ValidationResult

# rule_storage.py
class RuleStorage:
    def save_rule(self, rule: Rule) -> int
    def get_best_strategy(self, market_regime, stock_regime) -> str
    def export_rules(self, format='json') -> str
    def list_rules(self) -> List[Rule]

# strategy_recommender.py
class StrategyRecommender:
    def recommend(self, market_regime, stock_regime) -> List[str]
```

---

## 8. 大盘指数匹配规则

```python
INDEX_MAPPING = {
    # 科创板 (688xxx) → 科创50
    "688": "sh000688",
    
    # 创业板 (300xxx) → 创业板指
    "300": "sz399006",
    
    # 沪深300成分股 → 沪深300
    "hs300": "sh000300",
    
    # 上证主板 (60xxxx) → 上证指数
    "60": "sh000001",
    
    # 深证主板 (000xxx/001xxx) → 深证成指
    "000": "sz399001",
    "001": "sz399001",
    
    # 中小板 (002xxx) → 中小板指
    "002": "sz399005",
}
```

---

## 9. 规则可信度标准

规则保存条件：**验证集收益 > 基准收益**

```python
def should_save_rule(rule: Rule, baseline_return: float) -> bool:
    return rule.valid_return > baseline_return

def calc_confidence(rule: Rule, baseline: float) -> float:
    """置信度 = 超额收益 / 基准收益波动"""
    excess = rule.valid_return - baseline
    return min(1.0, max(0.0, excess / abs(baseline) * 2))
```

---

## 10. 成功标准

1. **大盘匹配准确** - 根据股票代码自动匹配正确的大盘指数
2. **择势可视化清晰** - ECharts图表能清晰展示择势状态和转折点
3. **AI推荐有效** - 推荐的策略在回测中表现优于随机选择
4. **规则可验证** - 验证集收益能超过基准
5. **规则可导出** - 支持JSON/CSV导出
6. **交互流畅** - 图表缩放、悬停等交互正常工作
