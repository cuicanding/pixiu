# Pixiu UI流程重构设计

**Date**: 2026-02-28
**Author**: AI Assistant
**Status**: Approved

---

## 1. 问题分析

### 当前问题

1. **UI流程混乱**
   - 策略选择区域在选择股票后才显示，用户不知道下一步该做什么
   - 策略卡片点击无视觉反馈
   - 没有步骤进度指示

2. **数据库问题**
   - `daily_quotes`表从未创建
   - `Database.create_tables()`没被调用
   - 回测时报错 `no such table: daily_quotes`

3. **数据流问题**
   - 数据获取和缓存逻辑不完整
   - AKShare API失败时没有优雅降级

---

## 2. 设计方案

### 整体架构

采用**6步分步向导模式**：

```
Step 1: 选择市场 → Step 2: 搜索股票 → Step 3: 择势分析 → Step 4: 选择策略 → Step 5: 配置参数 → Step 6: 查看结果
```

### 数据流

```
AKShare API → 成功返回数据
     ↓
   失败
     ↓
模拟数据生成 → 随机OHLCV数据
     ↓
SQLite缓存 → daily_quotes表
     ↓
策略计算 → 回测引擎 → 结果展示
```

---

## 3. 详细设计

### Step 1: 选择市场

**组件**:
- 三个市场按钮：A股、港股、美股
- 显示当前选中状态（高亮）
- 显示可选股票数量

**状态**:
- `current_market: str` - 当前市场
- 按钮使用 `rx.cond()` 显示选中状态

### Step 2: 搜索并选择股票

**组件**:
- 搜索输入框 + 搜索按钮
- 搜索结果列表（带选择按钮）
- 已选股票显示区域

**数据源**:
- 优先 AKShare API
- 失败时使用 MOCK_STOCKS 模拟数据

**状态**:
- `search_keyword: str` - 搜索关键词
- `search_results: List[Dict]` - 搜索结果
- `selected_stock: str` - 已选股票代码
- `selected_stock_name: str` - 已选股票名称

### Step 3: 择势分析（市场+个股）

**组件**:
- 大盘状态卡片（上证指数/恒生指数等）
- 个股状态卡片（ADX、MA斜率、波动率）
- 综合判断 + 策略推荐

**择势逻辑**:

| 大盘 | 个股 | 推荐策略 |
|------|------|----------|
| 趋势 | 趋势 | 均线、趋势强度 |
| 趋势 | 震荡 | 网格、RSI |
| 震荡 | 趋势 | 趋势强度 |
| 震荡 | 震荡 | 网格、RSI、波动率套利 |

**状态**:
- `market_regime: str` - 大盘状态
- `stock_regime: str` - 个股状态
- `regime_analysis: Dict` - 详细分析数据

### Step 4: 策略选择（智能推荐）

**组件**:
- 推荐策略区（基于择势分析自动推荐）
- 其他可用策略区
- 每个策略卡片带选中状态（checkbox）
- 策略描述

**状态**:
- `available_strategies: List[Dict]` - 所有策略
- `selected_strategies: List[str]` - 已选策略
- `recommended_strategies: List[str]` - 推荐策略

### Step 5: 参数配置

**组件**:
- 初始资金输入
- 手续费率输入
- 仓位比例输入
- 开始回测按钮

**状态**:
- `initial_capital: float` - 初始资金
- `commission_rate: float` - 手续费率
- `position_size: float` - 仓位比例

### Step 6: 结果展示

**组件**:
- 每个策略的回测结果卡片
- 关键指标：总收益率、年化收益、最大回撤、夏普比率、胜率
- 查看详细报告按钮
- AI分析按钮

**状态**:
- `backtest_results: List[Dict]` - 回测结果列表
- `ai_report: str` - AI分析报告

---

## 4. 数据层设计

### 数据库初始化

**修复**: 在应用启动或回测前自动调用 `Database.create_tables()`

```python
# state.py
async def run_backtest(self):
    db = Database("data/stocks.db")
    await db.create_tables()  # 确保表存在
    ...
```

### 数据获取流程

```python
async def get_stock_data(code: str, market: str) -> pd.DataFrame:
    # 1. 尝试从缓存获取
    cached = await db.get_quotes(code)
    if cached:
        return cached
    
    # 2. 尝试AKShare API
    try:
        df = await fetch_from_akshare(code, market)
        await save_to_db(df)
        return df
    except Exception:
        # 3. 降级到模拟数据
        return generate_mock_data(code)
```

### 模拟数据生成

```python
def generate_mock_history(code: str, days: int = 500) -> pd.DataFrame:
    """生成随机OHLCV数据"""
    np.random.seed(hash(code) % 2**32)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    base_price = np.random.uniform(10, 100)
    returns = np.random.randn(days) * 0.02
    close = base_price * np.exp(np.cumsum(returns))
    
    return pd.DataFrame({
        'trade_date': dates,
        'open': close * (1 + np.random.randn(days) * 0.005),
        'high': close * (1 + np.abs(np.random.randn(days) * 0.01)),
        'low': close * (1 - np.abs(np.random.randn(days) * 0.01)),
        'close': close,
        'volume': np.random.randint(1e6, 1e7, days),
    })
```

---

## 5. UI组件设计

### 策略卡片（带选中状态）

```python
def render_strategy(s: dict) -> rx.Component:
    is_selected = State.selected_strategies.contains(s["name"])
    is_recommended = State.recommended_strategies.contains(s["name"])
    
    return rx.box(
        rx.hstack(
            rx.checkbox(is_checked=is_selected),
            rx.text(s["name"]),
            rx.cond(is_recommended, rx.badge("推荐")),
        ),
        border=f"1px solid {'green.500' if is_selected else 'gray.700'}",
        on_click=State.toggle_strategy(s["name"]),
    )
```

### 步骤指示器

```python
def step_indicator(current_step: int) -> rx.Component:
    steps = ["选择市场", "搜索股票", "择势分析", "选择策略", "配置参数", "查看结果"]
    return rx.hstack(
        *[
            rx.badge(
                f"{i+1}. {step}",
                color_scheme="cyan" if i < current_step else "gray"
            )
            for i, step in enumerate(steps)
        ]
    )
```

---

## 6. 错误处理

### 网络错误
- AKShare API失败 → 自动降级模拟数据
- 显示提示："使用模拟数据进行回测"

### 数据库错误
- 表不存在 → 自动创建
- 数据为空 → 提示用户"暂无数据，使用模拟数据"

### 策略计算错误
- 捕获异常 → 显示错误信息
- 提供重试按钮

---

## 7. 实施计划

### Phase 1: 修复关键问题
- [ ] 数据库表自动创建
- [ ] 策略选择视觉反馈

### Phase 2: 重构UI流程
- [ ] 实现6步向导组件
- [ ] 添加择势分析步骤
- [ ] 智能策略推荐

### Phase 3: 优化体验
- [ ] 步骤进度指示
- [ ] 加载状态优化
- [ ] 结果可视化

---

## 8. 成功标准

1. 用户能清晰知道当前在哪一步
2. 每步操作有明确的视觉反馈
3. 数据库错误不再出现
4. AKShare失败时能优雅降级
5. 择势分析和策略推荐逻辑正确
