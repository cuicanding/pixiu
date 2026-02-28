# Pixiu 量化平台 - 可视化与AI解释功能设计

## 概述

本次迭代目标：让不懂量化的用户能够直观理解量化过程。

核心需求：
1. 真实数据接入（baostock替代akshare）
2. 时间范围选择（用户可自定义回测周期）
3. 强化表达和解释（走势图表、信号点、交易动作点）
4. AI辅助理解（GLM-5解释概念）

## 新流程设计

```
┌──────────────────────────────────────────────────────────────────┐
│  步骤1: 选择市场                                                  │
│  [A股] [港股] [美股]                                             │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  步骤2: 搜索股票 + 选择时间范围（合并）                            │
│  ┌─────────────────────────────────────────┐                     │
│  │ 搜索框: [输入股票代码或名称...]  [搜索]   │                     │
│  │ 搜索结果列表...                          │                     │
│  └─────────────────────────────────────────┘                     │
│  ┌─────────────────────────────────────────┐                     │
│  │ 时间范围:                                │                     │
│  │ ○ 近3个月  ○ 近6个月  ○ 近12个月  ○ 近2年  │                     │
│  │ ○ 今年    ○ 去年    ○ 2023年    ○ 2024年  │                     │
│  │ ○ 自定义: [开始日期] ~ [结束日期]         │                     │
│  └─────────────────────────────────────────┘                     │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  步骤3: 择势分析（基于选定时间范围）                               │
│  [走势图表 + ADX/MA/波动率 + 趋势/震荡判断]                       │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  步骤4: 选择策略                                                  │
│  [推荐策略 + 可选策略列表]                                        │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  步骤5: 配置参数                                                  │
│  初始资金、手续费率、仓位比例                                      │
│  [开始回测]                                                       │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  步骤6: 查看结果                                                  │
│  [K线图+买卖点] [资金曲线] [回撤曲线] [指标卡片] [AI解读]          │
└──────────────────────────────────────────────────────────────────┘
```

## 开发方案

采用**渐进式迭代**，分5个阶段：

| 阶段 | 内容 | 预计工作量 |
|------|------|------------|
| 0 | 时间范围选择UI | 1小时 |
| 1 | 数据源改造（baostock+时间参数） | 1-2小时 |
| 2 | 回测结果页可视化 | 2-3小时 |
| 3 | 择势分析页可视化 | 1-2小时 |
| 4 | AI解释功能 | 2-3小时 |

---

## 阶段0：时间范围选择UI

### 0.1 UI设计

时间范围选择与股票搜索合并在同一页面：

```
┌─────────────────────────────────────────────────────────────┐
│ 搜索股票                                                     │
│ 当前市场: A股                                                │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [输入股票代码或名称...]                    [搜索]       │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ 搜索结果:                                                    │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ 000001 平安银行                              [点击选择] │ │
│ │ 000002 万科A                                [点击选择] │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ 已选择: 000001 平安银行                                     │
│                                                             │
│ ─────────────────────────────────────────────────────────── │
│                                                             │
│ 回测时间范围                                                 │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ 快捷选项:                                                │ │
│ │ [近3个月] [近6个月] [近12个月] [近2年]                   │ │
│ │                                                         │ │
│ │ 年度选项:                                                │ │
│ │ [今年] [去年] [2023年] [2024年]                          │ │
│ │                                                         │ │
│ │ 自定义:                                                  │ │
│ │ 开始: [2024-01-01]  结束: [2024-12-31]                  │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ 当前选择: 2024-01-01 ~ 2024-12-31                           │
│                                                             │
│                                   [下一步：开始择势分析]     │
└─────────────────────────────────────────────────────────────┘
```

### 0.2 State新增字段

```python
class State(rx.State):
    # 时间范围相关
    time_range_mode: str = "quick"  # quick, year, custom
    quick_range: str = "12m"  # 3m, 6m, 12m, 2y
    year_range: str = "this_year"  # this_year, last_year, 2023, 2024
    custom_start_date: str = ""
    custom_end_date: str = ""
    
    # 计算后的实际日期
    backtest_start_date: str = ""
    backtest_end_date: str = ""
    
    def set_quick_range(self, range_key: str):
        self.time_range_mode = "quick"
        self.quick_range = range_key
        self._update_date_range()
    
    def set_year_range(self, year_key: str):
        self.time_range_mode = "year"
        self.year_range = year_key
        self._update_date_range()
    
    def set_custom_start(self, date: str):
        self.time_range_mode = "custom"
        self.custom_start_date = date
        self._update_date_range()
    
    def set_custom_end(self, date: str):
        self.time_range_mode = "custom"
        self.custom_end_date = date
        self._update_date_range()
    
    def _update_date_range(self):
        """根据选择的时间模式计算实际日期范围"""
        from datetime import datetime, timedelta
        
        today = datetime.now()
        
        if self.time_range_mode == "quick":
            if self.quick_range == "3m":
                start = today - timedelta(days=90)
            elif self.quick_range == "6m":
                start = today - timedelta(days=180)
            elif self.quick_range == "12m":
                start = today - timedelta(days=365)
            elif self.quick_range == "2y":
                start = today - timedelta(days=730)
            else:
                start = today - timedelta(days=365)
            end = today
            
        elif self.time_range_mode == "year":
            if self.year_range == "this_year":
                start = datetime(today.year, 1, 1)
                end = today
            elif self.year_range == "last_year":
                start = datetime(today.year - 1, 1, 1)
                end = datetime(today.year - 1, 12, 31)
            elif self.year_range == "2023":
                start = datetime(2023, 1, 1)
                end = datetime(2023, 12, 31)
            elif self.year_range == "2024":
                start = datetime(2024, 1, 1)
                end = datetime(2024, 12, 31)
            else:
                start = today - timedelta(days=365)
                end = today
                
        else:  # custom
            try:
                start = datetime.strptime(self.custom_start_date, "%Y-%m-%d")
                end = datetime.strptime(self.custom_end_date, "%Y-%m-%d")
            except:
                start = today - timedelta(days=365)
                end = today
        
        self.backtest_start_date = start.strftime("%Y-%m-%d")
        self.backtest_end_date = end.strftime("%Y-%m-%d")
```

### 0.3 页面组件更新

**文件**: `pixiu/pages/home.py` - `step_stock_search()`

在原有股票搜索基础上增加时间范围选择区域。

---

## 阶段1：数据源改造

### 1.1 架构变更

```
DataService
├── _fetch_from_baostock(code, market, start_date, end_date)  # 新增：优先使用baostock，支持时间参数
├── _fetch_from_akshare(code, market)   # 保留：作为备用
└── _generate_mock_history(code) # 保留：最后兜底
```

### 1.2 Baostock适配

**特点**：
- 需要先`bs.login()`登录
- A股数据完整，港股/美股有限
- 免费无限制

**字段映射**：
| baostock | pixiu |
|----------|-------|
| date | trade_date |
| open | open |
| high | high |
| low | low |
| close | close |
| volume | volume |
| amount | amount |

### 1.3 优先级策略

```
1. 尝试baostock（新，带时间参数）
2. 失败则尝试akshare（旧）
3. 都失败则用mock数据
```

### 1.4 代码变更

**文件**: `pixiu/services/data_service.py`

```python
def _fetch_from_baostock(
    self, 
    code: str, 
    market: str,
    start_date: str = None,
    end_date: str = None
) -> pd.DataFrame:
    """从baostock获取数据"""
    import baostock as bs
    
    lg = bs.login()
    if lg.error_code != '0':
        raise Exception(f"baostock login failed: {lg.error_msg}")
    
    # 日期范围 - 使用传入参数或默认值
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    
    # baostock代码格式
    bs_code = f"sh{code}" if code.startswith("6") else f"sz{code}"
    
    rs = bs.query_history_k_data_plus(
        bs_code,
        "date,code,open,high,low,close,volume,amount",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3"
    )
    
    data_list = []
    while rs.next():
        data_list.append(rs.get_row_data())
    
    df = pd.DataFrame(data_list, columns=rs.fields)
    df = df.rename(columns={'date': 'trade_date'})
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    bs.logout()
    return df.sort_values('trade_date')

async def fetch_stock_history(
    self, 
    code: str, 
    market: str = "A股",
    start_date: str = None,
    end_date: str = None
) -> pd.DataFrame:
    """获取股票历史数据 - 尝试真实API，失败则用mock"""
    if not self.use_mock:
        try:
            df = await asyncio.wait_for(
                asyncio.to_thread(
                    self._fetch_from_baostock, 
                    code, market, start_date, end_date
                ),
                timeout=30
            )
            if df is not None and not df.empty:
                logger.info(f"成功从Baostock获取 {code} 数据")
                return df
        except Exception as e:
            logger.warning(f"Baostock获取失败: {e}，尝试akshare")
            
        # 尝试akshare作为备用
        try:
            df = await asyncio.wait_for(
                asyncio.to_thread(self._fetch_from_akshare, code, market),
                timeout=30
            )
            if df is not None and not df.empty:
                logger.info(f"成功从AKShare获取 {code} 数据")
                return df
        except Exception as e:
            logger.warning(f"AKShare获取失败: {e}，使用模拟数据")
            
    return self._generate_mock_history(code)
```

---

## 阶段2：回测结果页可视化

### 2.1 图表组成

回测结果页将包含4个子图：

```
┌─────────────────────────────────────┐
│  主图：K线 + 买卖点标注              │
│  (红箭头买入，绿箭头卖出)            │
├─────────────────────────────────────┤
│  副图1：资金曲线                      │
│  (总资产随时间变化)                   │
├─────────────────────────────────────┤
│  副图2：回撤曲线                      │
│  (从高点回落的百分比)                 │
├─────────────────────────────────────┤
│  副图3：策略信号强度 (可选)           │
│  (如果策略支持)                       │
└─────────────────────────────────────┘
```

### 2.2 技术选型

使用 **Plotly** 生成静态图片（Reflex支持）：
- 转换为base64图片嵌入页面
- 不依赖前端JS交互

### 2.3 数据结构

**State新增字段**:
```python
backtest_charts: Dict[str, str] = {}  # strategy_name -> base64_image
```

**BacktestResult已有字段**:
```python
equity_curve: List[float]    # 资金曲线
drawdown_curve: List[float]  # 回撤曲线
trades: List[Trade]          # 交易记录
```

### 2.4 图表生成函数

**新文件**: `pixiu/services/chart_service.py`

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import pandas as pd
from typing import List

def generate_backtest_chart(
    df: pd.DataFrame,
    trades: List,
    equity_curve: List[float],
    drawdown_curve: List[float]
) -> str:
    """生成回测结果图表，返回base64字符串"""
    
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.05
    )
    
    # K线图
    fig.add_trace(
        go.Candlestick(
            x=df['trade_date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K线'
        ),
        row=1, col=1
    )
    
    # 买卖点
    buy_trades = [t for t in trades if t.signal_type == "BUY"]
    sell_trades = [t for t in trades if t.signal_type == "SELL"]
    
    fig.add_trace(
        go.Scatter(
            x=[t.trade_date for t in buy_trades],
            y=[t.price for t in buy_trades],
            mode='markers',
            marker=dict(symbol='triangle-up', size=12, color='red'),
            name='买入'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[t.trade_date for t in sell_trades],
            y=[t.price for t in sell_trades],
            mode='markers',
            marker=dict(symbol='triangle-down', size=12, color='green'),
            name='卖出'
        ),
        row=1, col=1
    )
    
    # 资金曲线
    fig.add_trace(
        go.Scatter(
            x=df['trade_date'],
            y=equity_curve,
            mode='lines',
            name='资金曲线',
            line=dict(color='blue')
        ),
        row=2, col=1
    )
    
    # 回撤曲线
    fig.add_trace(
        go.Scatter(
            x=df['trade_date'],
            y=[d * 100 for d in drawdown_curve],
            mode='lines',
            name='回撤',
            fill='tozeroy',
            line=dict(color='red')
        ),
        row=3, col=1
    )
    
    fig.update_layout(height=800, showlegend=True, template='plotly_dark')
    
    # 转换为base64
    img_bytes = fig.to_image(format="png", width=1000, height=800)
    return base64.b64encode(img_bytes).decode()
```

---

## 阶段3：择势分析页可视化

### 3.1 图表组成

```
┌─────────────────────────────────────┐
│  个股走势 + MA均线 + ADX指标         │
│  标注当前趋势/震荡状态                │
├─────────────────────────────────────┤
│  底部显示关键指标数值                 │
│  ADX: 32.5 | MA斜率: 0.002 | 波动率: 0.015 │
└─────────────────────────────────────┘
```

### 3.2 视觉提示

- **趋势状态**: 绿色背景标注"趋势↑"或"趋势↓"
- **震荡状态**: 黄色背景标注"震荡"
- **推荐策略**: 在图表下方高亮显示

### 3.3 数据结构

**State新增字段**:
```python
regime_chart: str = ""  # base64图片
```

### 3.4 图表生成

```python
def generate_regime_chart(df: pd.DataFrame, analysis: dict) -> str:
    """生成择势分析图表"""
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1
    )
    
    # 价格+MA
    fig.add_trace(
        go.Scatter(x=df['trade_date'], y=df['close'], name='收盘价'),
        row=1, col=1
    )
    
    ma20 = df['close'].rolling(20).mean()
    fig.add_trace(
        go.Scatter(x=df['trade_date'], y=ma20, name='MA20', line=dict(dash='dash')),
        row=1, col=1
    )
    
    # ADX
    adx = _calculate_adx(df)
    fig.add_trace(
        go.Scatter(x=df['trade_date'], y=adx, name='ADX', line=dict(color='orange')),
        row=2, col=1
    )
    
    # 状态标注
    regime = analysis.get('regime', 'unknown')
    fig.add_annotation(
        text="趋势" if regime == 'trend' else "震荡",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=16, color="white"),
        bgcolor="green" if regime == 'trend' else "orange"
    )
    
    fig.update_layout(height=500, template='plotly_dark')
    
    img_bytes = fig.to_image(format="png", width=800, height=500)
    return base64.b64encode(img_bytes).decode()

def _calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """计算ADX指标"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(period).mean()
    
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = dx.rolling(period).mean()
    
    return adx
```

---

## 阶段4：AI解释功能

### 4.1 交互设计

在每个需要解释的概念旁边添加"?"图标按钮：

```
夏普比率  [?] ← 点击弹出AI解释
```

点击后弹出模态框，显示GLM-5生成的解释。

### 4.2 解释位置

| 页面 | 解释项 |
|------|--------|
| **回测结果页** | 每个指标旁（总收益、夏普、回撤、胜率等） |
| **择势分析页** | ADX、MA斜率、波动率、趋势/震荡状态 |
| **策略选择页** | 每个策略的信号含义 |

### 4.3 提示词模板

**新文件**: `pixiu/services/explain_prompts.py`

```python
EXPLAIN_PROMPTS = {
    "total_return": """用简单语言解释：
1. 什么是总收益率？
2. 数值{value:.2%}代表什么水平？（优秀/一般/较差）
3. 对普通投资者意味着什么？""",
    
    "sharpe_ratio": """用简单语言解释：
1. 什么是夏普比率？
2. 数值{value:.2f}代表什么水平？（>1优秀，0.5-1一般，<0.5较差）
3. 为什么这个指标重要？""",
    
    "max_drawdown": """用简单语言解释：
1. 什么是最大回撤？
2. {value:.2%}的回撤意味着什么？
3. 投资者应该如何应对这种回撤？""",
    
    "win_rate": """用简单语言解释：
1. 什么是胜率？
2. 胜率{value:.2%}意味着什么？
3. 胜率高的策略一定好吗？""",
    
    "regime": """用简单语言解释：
1. 当前个股处于{regime}状态，这是什么意思？
2. 这对投资者意味着什么？
3. 应该采取什么策略？""",
    
    "adx": """用简单语言解释：
1. 什么是ADX指标？
2. 当前ADX值{value:.2f}代表什么？（<20弱趋势，20-25初现趋势，>25强趋势）
3. 投资者应该如何利用这个信息？""",
    
    "strategy_signal": """用简单语言解释：
1. {strategy}策略是什么？
2. 它是如何产生交易信号的？
3. 当前信号{signal}意味着什么操作？"""
}
```

### 4.4 组件设计

**新组件**: `pixiu/components/explain_button.py`

```python
import reflex as rx
from pixiu.state import State

def explain_button(concept: str, value: str = "") -> rx.Component:
    return rx.hstack(
        rx.tooltip(
            rx.icon_button(
                rx.icon("help-circle", size=16),
                size="sm",
                variant="ghost",
                on_click=State.explain_concept(concept, value),
            ),
            label="点击获取AI解释",
        ),
        align_items="center",
    )

def explain_modal() -> rx.Component:
    return rx.modal(
        rx.modal_overlay(
            rx.modal_content(
                rx.modal_header("AI 解释"),
                rx.modal_body(
                    rx.cond(
                        State.ai_explaining,
                        rx.spinner(),
                        rx.text(State.current_explanation, white_space="pre-wrap"),
                    )
                ),
                rx.modal_footer(
                    rx.button("关闭", on_click=State.close_explain_modal),
                ),
            )
        ),
        is_open=State.explain_modal_open,
    )
```

### 4.5 State扩展

```python
class State(rx.State):
    # 新增字段
    explain_modal_open: bool = False
    current_explanation: str = ""
    ai_explaining: bool = False
    
    async def explain_concept(self, concept: str, value: str):
        """生成概念解释"""
        self.explain_modal_open = True
        self.ai_explaining = True
        self.current_explanation = ""
        yield
        
        try:
            from pixiu.services.ai_service import AIReportService
            from pixiu.services.explain_prompts import EXPLAIN_PROMPTS
            
            prompt_template = EXPLAIN_PROMPTS.get(concept, "请解释{concept}")
            prompt = prompt_template.format(
                concept=concept,
                value=value,
                regime=self.stock_regime,
                strategy=self.selected_strategies[0] if self.selected_strategies else ""
            )
            
            ai_service = AIReportService(self.glm_api_key)
            self.current_explanation = await ai_service._call_api(prompt)
            
        except Exception as e:
            self.current_explanation = f"解释生成失败: {str(e)}"
        finally:
            self.ai_explaining = False
        yield
    
    def close_explain_modal(self):
        self.explain_modal_open = False
```

---

## 文件变更汇总

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `pixiu/services/data_service.py` | 修改 | 添加baostock数据源，支持时间参数 |
| `pixiu/services/chart_service.py` | 新增 | 图表生成服务 |
| `pixiu/services/explain_prompts.py` | 新增 | AI解释提示词模板 |
| `pixiu/components/explain_button.py` | 新增 | 解释按钮组件 |
| `pixiu/pages/home.py` | 修改 | 添加时间选择、图表和解释按钮 |
| `pixiu/state.py` | 修改 | 添加时间范围、图表和解释相关状态 |
| `requirements.txt` | 修改 | 添加baostock, kaleido依赖 |

---

## 依赖添加

```
baostock>=0.8.8
kaleido>=0.2.1  # plotly静态图片导出
```

---

## 实施顺序

1. **阶段0** - 修改State添加时间范围字段，更新home.py的step_stock_search组件
2. **阶段1** - 修改data_service.py添加baostock支持，更新fetch_stock_history接口
3. **阶段2** - 创建chart_service.py，修改home.py的step_results组件
4. **阶段3** - 在chart_service.py添加择势图表，修改home.py的step_regime_analysis组件
5. **阶段4** - 创建explain_prompts.py和explain_button.py，修改State添加解释方法
