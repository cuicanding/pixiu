# Pixiu 可视化与AI解释功能 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为Pixiu量化平台添加真实数据接入、时间范围选择、可视化图表和AI解释功能

**Architecture:** 5阶段渐进式开发 - 先添加时间选择UI，再改造数据源支持baostock，然后添加回测结果可视化、择势分析可视化，最后添加AI解释功能

**Tech Stack:** Python, Reflex, Plotly, baostock, GLM-5 API

---

## Task 1: 添加时间范围选择状态

**Files:**
- Modify: `pixiu/state.py`
- Test: `tests/test_state_time_range.py`

**Step 1: 写测试**

```python
# tests/test_state_time_range.py
import pytest
from pixiu.state import State

def test_state_has_time_range_fields():
    state = State()
    assert hasattr(state, 'time_range_mode')
    assert hasattr(state, 'quick_range')
    assert hasattr(state, 'year_range')
    assert hasattr(state, 'backtest_start_date')
    assert hasattr(state, 'backtest_end_date')

def test_set_quick_range_updates_dates():
    state = State()
    state.set_quick_range("12m")
    assert state.time_range_mode == "quick"
    assert state.quick_range == "12m"
    assert state.backtest_start_date != ""
    assert state.backtest_end_date != ""

def test_set_year_range_this_year():
    state = State()
    state.set_year_range("this_year")
    assert state.time_range_mode == "year"
    assert state.year_range == "this_year"
```

**Step 2: 运行测试确认失败**

```bash
pytest tests/test_state_time_range.py -v
```
Expected: FAIL (属性不存在)

**Step 3: 添加时间范围字段到State**

在 `pixiu/state.py` 的 `State` 类中添加字段和方法：

```python
class State(rx.State):
    # ... 现有字段 ...
    
    # 时间范围相关
    time_range_mode: str = "quick"
    quick_range: str = "12m"
    year_range: str = "this_year"
    custom_start_date: str = ""
    custom_end_date: str = ""
    backtest_start_date: str = ""
    backtest_end_date: str = ""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_strategies()
        self._load_settings()
        self._update_date_range()
    
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
        else:
            try:
                start = datetime.strptime(self.custom_start_date, "%Y-%m-%d")
                end = datetime.strptime(self.custom_end_date, "%Y-%m-%d")
            except:
                start = today - timedelta(days=365)
                end = today
        
        self.backtest_start_date = start.strftime("%Y-%m-%d")
        self.backtest_end_date = end.strftime("%Y-%m-%d")
```

**Step 4: 运行测试确认通过**

```bash
pytest tests/test_state_time_range.py -v
```
Expected: PASS

**Step 5: 提交**

```bash
git add pixiu/state.py tests/test_state_time_range.py
git commit -m "feat(state): add time range selection fields and methods"
```

---

## Task 2: 更新股票搜索页面添加时间范围选择UI

**Files:**
- Modify: `pixiu/pages/home.py`
- Test: 手动测试UI

**Step 1: 更新step_stock_search函数**

修改 `pixiu/pages/home.py` 中的 `step_stock_search()` 函数，在股票搜索区域下方添加时间范围选择：

```python
def step_stock_search() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.text("搜索股票", font_size="lg", font_weight="bold"),
            rx.text(f"当前市场: {State.current_market}", font_size="sm", color="gray.400"),
            
            rx.hstack(
                rx.input(
                    placeholder="输入股票代码或名称...",
                    value=State.search_keyword,
                    on_change=State.set_search_keyword,
                    width="100%",
                    size="3",
                ),
                rx.button(
                    "搜索",
                    on_click=State.search_stocks,
                    color_scheme="cyan",
                    size="3",
                    is_loading=State.is_loading,
                ),
                width="100%",
                spacing="2",
            ),
            
            rx.cond(State.is_loading, rx.hstack(rx.spinner(), rx.text(State.loading_message))),
            rx.cond(State.error_message != "", rx.text(State.error_message, color="red.400")),
            
            rx.box(
                rx.foreach(State.search_results, render_search_result),
                max_height="200px",
                overflow_y="auto",
                width="100%",
            ),
            
            rx.cond(
                State.selected_stock != "",
                rx.box(
                    rx.hstack(
                        rx.text("已选择:", color="gray.400"),
                        rx.badge(State.selected_stock, color_scheme="cyan"),
                        rx.text(State.selected_stock_name, font_weight="bold"),
                    ),
                    padding="0.75rem",
                    bg="gray.800",
                    border_radius="md",
                ),
            ),
            
            rx.divider(margin_y="1rem"),
            
            rx.text("回测时间范围", font_size="lg", font_weight="bold"),
            
            rx.text("快捷选项:", font_size="sm", color="gray.400"),
            rx.hstack(
                rx.button("近3个月", size="2", variant=rx.cond(State.quick_range == "3m", "solid", "outline"), color_scheme="cyan", on_click=State.set_quick_range("3m")),
                rx.button("近6个月", size="2", variant=rx.cond(State.quick_range == "6m", "solid", "outline"), color_scheme="cyan", on_click=State.set_quick_range("6m")),
                rx.button("近12个月", size="2", variant=rx.cond(State.quick_range == "12m", "solid", "outline"), color_scheme="cyan", on_click=State.set_quick_range("12m")),
                rx.button("近2年", size="2", variant=rx.cond(State.quick_range == "2y", "solid", "outline"), color_scheme="cyan", on_click=State.set_quick_range("2y")),
                spacing="2",
                flex_wrap="wrap",
            ),
            
            rx.text("年度选项:", font_size="sm", color="gray.400", margin_top="0.5rem"),
            rx.hstack(
                rx.button("今年", size="2", variant=rx.cond(State.year_range == "this_year", "solid", "outline"), color_scheme="cyan", on_click=State.set_year_range("this_year")),
                rx.button("去年", size="2", variant=rx.cond(State.year_range == "last_year", "solid", "outline"), color_scheme="cyan", on_click=State.set_year_range("last_year")),
                rx.button("2023年", size="2", variant=rx.cond(State.year_range == "2023", "solid", "outline"), color_scheme="cyan", on_click=State.set_year_range("2023")),
                rx.button("2024年", size="2", variant=rx.cond(State.year_range == "2024", "solid", "outline"), color_scheme="cyan", on_click=State.set_year_range("2024")),
                spacing="2",
                flex_wrap="wrap",
            ),
            
            rx.text("自定义:", font_size="sm", color="gray.400", margin_top="0.5rem"),
            rx.hstack(
                rx.text("开始:", font_size="sm"),
                rx.input(type="date", value=State.custom_start_date, on_change=State.set_custom_start, width="140px", size="2"),
                rx.text("结束:", font_size="sm"),
                rx.input(type="date", value=State.custom_end_date, on_change=State.set_custom_end, width="140px", size="2"),
                spacing="2",
                align_items="center",
            ),
            
            rx.box(
                rx.hstack(
                    rx.text("当前时间范围:", color="gray.400"),
                    rx.badge(State.backtest_start_date, color_scheme="cyan"),
                    rx.text("~", color="gray.400"),
                    rx.badge(State.backtest_end_date, color_scheme="cyan"),
                ),
                padding="0.75rem",
                bg="gray.900",
                border_radius="md",
                margin_top="1rem",
            ),
            
            spacing="3",
        ),
        padding="1.5rem",
        border="1px solid gray.700",
        border_radius="lg",
        width="100%",
    )
```

**Step 2: 手动测试**

```bash
reflex run
```
访问 http://localhost:3000，选择市场后进入股票搜索页面，验证时间范围选择UI正常显示和工作。

**Step 3: 提交**

```bash
git add pixiu/pages/home.py
git commit -m "feat(ui): add time range selection to stock search page"
```

---

## Task 3: 添加baostock数据源支持

**Files:**
- Modify: `pixiu/services/data_service.py`
- Test: `tests/test_baostock.py`

**Step 1: 写测试**

```python
# tests/test_baostock.py
import pytest
from pixiu.services.data_service import DataService
from pixiu.services.database import Database
import pandas as pd

@pytest.fixture
def data_service():
    db = Database(":memory:")
    return DataService(db, use_mock=False)

def test_fetch_from_baostock_returns_dataframe(data_service):
    df = data_service._fetch_from_baostock(
        "000001", 
        "A股",
        "2024-01-01",
        "2024-01-31"
    )
    assert isinstance(df, pd.DataFrame)
    if not df.empty:
        assert 'trade_date' in df.columns
        assert 'close' in df.columns

def test_fetch_from_baostock_with_none_dates(data_service):
    df = data_service._fetch_from_baostock("000001", "A股")
    assert isinstance(df, pd.DataFrame)
```

**Step 2: 运行测试确认失败**

```bash
pytest tests/test_baostock.py -v
```
Expected: FAIL (方法不存在)

**Step 3: 实现baostock数据获取方法**

在 `pixiu/services/data_service.py` 中添加：

```python
def _fetch_from_baostock(
    self, 
    code: str, 
    market: str,
    start_date: str = None,
    end_date: str = None
) -> pd.DataFrame:
    """从baostock获取数据"""
    try:
        import baostock as bs
    except ImportError:
        raise ImportError("baostock not installed, run: pip install baostock")
    
    lg = bs.login()
    if lg.error_code != '0':
        raise Exception(f"baostock login failed: {lg.error_msg}")
    
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    
    if market == "A股":
        bs_code = f"sh{code}" if code.startswith("6") else f"sz{code}"
    else:
        bs.logout()
        raise Exception(f"baostock only supports A股, got {market}")
    
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
    
    bs.logout()
    
    if not data_list:
        return pd.DataFrame()
    
    df = pd.DataFrame(data_list, columns=rs.fields)
    df = df.rename(columns={'date': 'trade_date'})
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df.sort_values('trade_date').reset_index(drop=True)
```

**Step 4: 更新fetch_stock_history方法**

修改 `pixiu/services/data_service.py` 中的 `fetch_stock_history` 方法：

```python
async def fetch_stock_history(
    self, 
    code: str, 
    market: str = "A股",
    start_date: str = None,
    end_date: str = None
) -> pd.DataFrame:
    """获取股票历史数据 - 优先baostock，备用akshare，最后mock"""
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
                logger.info(f"成功从Baostock获取 {code} 数据: {len(df)}条")
                return df
        except Exception as e:
            logger.warning(f"Baostock获取失败: {e}")
        
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

**Step 5: 运行测试确认通过**

```bash
pytest tests/test_baostock.py -v
```
Expected: PASS

**Step 6: 提交**

```bash
git add pixiu/services/data_service.py tests/test_baostock.py
git commit -m "feat(data): add baostock data source support with time range"
```

---

## Task 4: 创建图表服务

**Files:**
- Create: `pixiu/services/chart_service.py`
- Test: `tests/test_chart_service.py`

**Step 1: 写测试**

```python
# tests/test_chart_service.py
import pytest
import pandas as pd
from pixiu.services.chart_service import generate_backtest_chart, generate_regime_chart

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'trade_date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'open': [10.0] * 100,
        'high': [10.5] * 100,
        'low': [9.5] * 100,
        'close': [10.0 + i * 0.01 for i in range(100)],
        'volume': [1000000] * 100,
    })

@pytest.fixture
def sample_trades():
    from pixiu.models.backtest import Trade
    return [
        Trade(trade_date='2024-01-10', signal_type='BUY', price=10.0, shares=1000, amount=10000.0, commission=3.0),
        Trade(trade_date='2024-01-20', signal_type='SELL', price=10.5, shares=1000, amount=10500.0, commission=3.15),
    ]

def test_generate_backtest_chart_returns_base64(sample_df, sample_trades):
    equity = [100000 + i * 10 for i in range(100)]
    drawdown = [0.0] * 50 + [-0.01 * (i-50) for i in range(50, 100)]
    
    result = generate_backtest_chart(sample_df, sample_trades, equity, drawdown)
    assert isinstance(result, str)
    assert len(result) > 100

def test_generate_regime_chart_returns_base64(sample_df):
    analysis = {'regime': 'trend', 'adx': 30.0}
    result = generate_regime_chart(sample_df, analysis)
    assert isinstance(result, str)
    assert len(result) > 100
```

**Step 2: 运行测试确认失败**

```bash
pytest tests/test_chart_service.py -v
```
Expected: FAIL (模块不存在)

**Step 3: 创建图表服务**

```python
# pixiu/services/chart_service.py
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
        vertical_spacing=0.05,
        subplot_titles=('价格走势', '资金曲线', '回撤曲线')
    )
    
    dates = df['trade_date'].tolist()
    
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K线',
            increasing_line_color='red',
            decreasing_line_color='green',
        ),
        row=1, col=1
    )
    
    if trades:
        buy_trades = [t for t in trades if t.signal_type == "BUY"]
        sell_trades = [t for t in trades if t.signal_type == "SELL"]
        
        if buy_trades:
            fig.add_trace(
                go.Scatter(
                    x=[str(t.trade_date) for t in buy_trades],
                    y=[float(t.price) for t in buy_trades],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=12, color='red'),
                    name='买入',
                    hovertemplate='买入<br>日期: %{x}<br>价格: %{y}<extra></extra>'
                ),
                row=1, col=1
            )
        
        if sell_trades:
            fig.add_trace(
                go.Scatter(
                    x=[str(t.trade_date) for t in sell_trades],
                    y=[float(t.price) for t in sell_trades],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=12, color='green'),
                    name='卖出',
                    hovertemplate='卖出<br>日期: %{x}<br>价格: %{y}<extra></extra>'
                ),
                row=1, col=1
            )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=equity_curve,
            mode='lines',
            name='资金',
            line=dict(color='#3b82f6', width=2),
            hovertemplate='%{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=[d * 100 for d in drawdown_curve],
            mode='lines',
            name='回撤',
            fill='tozeroy',
            line=dict(color='#ef4444', width=1),
            hovertemplate='%{y:.2f}%<extra></extra>'
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=50, t=50, b=50),
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    img_bytes = fig.to_image(format="png", width=900, height=800, scale=1.5)
    return base64.b64encode(img_bytes).decode()


def generate_regime_chart(df: pd.DataFrame, analysis: dict) -> str:
    """生成择势分析图表"""
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1,
        subplot_titles=('价格走势', 'ADX指标')
    )
    
    dates = df['trade_date'].tolist()
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df['close'],
            name='收盘价',
            line=dict(color='#3b82f6', width=2),
        ),
        row=1, col=1
    )
    
    ma20 = df['close'].rolling(20).mean()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ma20,
            name='MA20',
            line=dict(color='#f59e0b', width=1, dash='dash'),
        ),
        row=1, col=1
    )
    
    adx = _calculate_adx(df)
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=adx,
            name='ADX',
            line=dict(color='#ef4444', width=2),
        ),
        row=2, col=1
    )
    
    fig.add_hline(y=25, line_dash="dash", line_color="gray", row=2, col=1,
                  annotation_text="趋势线", annotation_position="right")
    
    regime = analysis.get('regime', 'unknown')
    regime_text = "趋势行情" if regime == 'trend' else "震荡行情"
    regime_color = "#22c55e" if regime == 'trend' else "#f59e0b"
    
    fig.add_annotation(
        text=regime_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=14, color="white"),
        bgcolor=regime_color,
        bordercolor=regime_color,
        borderwidth=2,
        borderpad=4,
    )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=50, t=50, b=50),
    )
    
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    img_bytes = fig.to_image(format="png", width=800, height=500, scale=1.5)
    return base64.b64encode(img_bytes).decode()


def _calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """计算ADX指标"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(period).mean()
    
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, float('nan')))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, float('nan')))
    
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float('nan')))
    adx = dx.rolling(period).mean()
    
    return adx.fillna(0)
```

**Step 4: 运行测试确认通过**

```bash
pytest tests/test_chart_service.py -v
```
Expected: PASS

**Step 5: 提交**

```bash
git add pixiu/services/chart_service.py tests/test_chart_service.py
git commit -m "feat(chart): add backtest and regime chart generation service"
```

---

## Task 5: 集成图表到回测结果页

**Files:**
- Modify: `pixiu/state.py`
- Modify: `pixiu/pages/home.py`

**Step 1: 在State中添加图表生成逻辑**

在 `pixiu/state.py` 的 State 类中添加：

```python
backtest_charts: Dict[str, str] = {}

async def run_backtest(self):
    # ... 现有代码 ...
    
    # 在回测循环中，生成图表
    for i, strategy_name in enumerate(self.selected_strategies):
        # ... 现有回测代码 ...
        
        # 生成图表
        from pixiu.services.chart_service import generate_backtest_chart
        chart_base64 = generate_backtest_chart(
            df_with_signals, 
            result.trades, 
            result.equity_curve, 
            result.drawdown_curve
        )
        
        self.backtest_charts[strategy_name] = chart_base64
        
        # ... 添加结果到backtest_results ...
```

**Step 2: 更新step_results函数显示图表**

修改 `pixiu/pages/home.py` 中的 `render_backtest_result` 和 `step_results` 函数：

```python
def render_backtest_result(result: dict) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text(result["strategy"], font_weight="bold", font_size="lg"),
                rx.spacer(),
                rx.badge(
                    f"{result['total_return']*100:.1f}%",
                    color_scheme=rx.cond(result["total_return"] >= 0, "green", "red"),
                ),
            ),
            
            rx.grid(
                rx.box(
                    rx.text("年化收益", font_size="sm", color="gray.400"),
                    rx.text(f"{result['annualized_return']*100:.1f}%", font_weight="bold"),
                ),
                rx.box(
                    rx.text("最大回撤", font_size="sm", color="gray.400"),
                    rx.text(f"{result['max_drawdown']*100:.1f}%", color="red.400"),
                ),
                rx.box(
                    rx.text("夏普比率", font_size="sm", color="gray.400"),
                    rx.text(f"{result['sharpe_ratio']:.2f}"),
                ),
                rx.box(
                    rx.text("胜率", font_size="sm", color="gray.400"),
                    rx.text(f"{result['win_rate']*100:.0f}%"),
                ),
                columns="4",
                spacing="4",
            ),
            
            rx.box(
                rx.image(
                    src=f"data:image/png;base64,{State.backtest_charts[result['strategy']]}",
                    width="100%",
                    border_radius="md",
                ),
                margin_top="1rem",
            ),
            
            spacing="3",
        ),
        padding="1rem",
        bg="gray.800",
        border_radius="md",
        width="100%",
    )
```

**Step 3: 提交**

```bash
git add pixiu/state.py pixiu/pages/home.py
git commit -m "feat(ui): integrate backtest charts into results page"
```

---

## Task 6: 集成图表到择势分析页

**Files:**
- Modify: `pixiu/state.py`
- Modify: `pixiu/pages/home.py`

**Step 1: 在State中添加择势图表字段和生成逻辑**

在 `pixiu/state.py` 中添加：

```python
regime_chart: str = ""

async def analyze_regime(self):
    # ... 现有代码 ...
    
    # 生成择势图表
    if df is not None and not df.empty:
        from pixiu.services.chart_service import generate_regime_chart
        self.regime_chart = generate_regime_chart(df, stock_analysis)
```

**Step 2: 更新step_regime_analysis显示图表**

修改 `pixiu/pages/home.py` 中的 `step_regime_analysis()` 函数，在分析结果区域添加图表显示。

**Step 3: 提交**

```bash
git add pixiu/state.py pixiu/pages/home.py
git commit -m "feat(ui): add regime analysis chart visualization"
```

---

## Task 7: 创建AI解释提示词模板

**Files:**
- Create: `pixiu/services/explain_prompts.py`

**Step 1: 创建提示词模板文件**

```python
# pixiu/services/explain_prompts.py

EXPLAIN_PROMPTS = {
    "total_return": """用简单易懂的中文解释：
1. 什么是总收益率？
2. 数值 {value} 代表什么水平？（优秀/一般/较差，给出判断依据）
3. 对普通投资者意味着什么？请用生活化的比喻。""",
    
    "sharpe_ratio": """用简单易懂的中文解释：
1. 什么是夏普比率？请用大白话解释。
2. 数值 {value} 代表什么水平？
   - 大于1：优秀
   - 0.5-1：一般
   - 小于0.5：较差
3. 为什么这个指标对投资者很重要？""",
    
    "max_drawdown": """用简单易懂的中文解释：
1. 什么是最大回撤？用通俗的比喻解释。
2. {value} 的回撤意味着什么？
3. 投资者应该如何理解和应对这种程度的回撤？""",
    
    "win_rate": """用简单易懂的中文解释：
1. 什么是胜率？
2. 胜率 {value} 意味着什么？
3. 胜率高的策略一定好吗？请解释胜率和盈利的关系。""",
    
    "annualized_return": """用简单易懂的中文解释：
1. 什么是年化收益率？
2. {value} 的年化收益是什么水平？
3. 这个收益和银行存款、理财产品相比如何？""",
    
    "regime": """用简单易懂的中文解释：
1. 当前个股处于 {regime} 状态，这是什么意思？
2. 这对投资者意味着什么？
3. 在这种市场状态下，投资者应该采取什么策略？""",
    
    "adx": """用简单易懂的中文解释：
1. 什么是ADX指标？它的作用是什么？
2. 当前ADX值 {value} 代表什么？
   - 小于20：弱趋势或无趋势
   - 20-25：趋势开始形成
   - 大于25：强趋势
3. 投资者应该如何利用这个信息来做决策？""",
    
    "volatility": """用简单易懂的中文解释：
1. 什么是波动率？
2. 当前波动率 {value} 意味着什么？（高波动/低波动）
3. 波动率对投资者有什么影响？""",
    
    "strategy_trend": """用简单易懂的中文解释趋势强度策略：
1. 这个策略的核心思想是什么？
2. 它是如何判断买入和卖出时机的？
3. 这个策略适合什么样的市场环境？有什么优缺点？""",
    
    "strategy_rsi": """用简单易懂的中文解释RSI策略：
1. 什么是RSI指标？
2. 这个策略是如何产生交易信号的？
3. 使用这个策略需要注意什么？""",
    
    "strategy_grid": """用简单易懂的中文解释网格交易策略：
1. 什么是网格交易？用简单的比喻解释。
2. 这个策略适合什么样的市场？
3. 有什么风险？如何控制？""",
}

def get_prompt(concept: str, **kwargs) -> str:
    """获取格式化后的提示词"""
    template = EXPLAIN_PROMPTS.get(concept, "请解释 {concept}")
    return template.format(**kwargs)
```

**Step 2: 提交**

```bash
git add pixiu/services/explain_prompts.py
git commit -m "feat(ai): add explanation prompt templates for concepts"
```

---

## Task 8: 添加AI解释功能到State

**Files:**
- Modify: `pixiu/state.py`

**Step 1: 添加AI解释相关字段和方法**

```python
explain_modal_open: bool = False
current_explanation: str = ""
ai_explaining: bool = False

async def explain_concept(self, concept: str, value: str = ""):
    """生成概念解释"""
    self.explain_modal_open = True
    self.ai_explaining = True
    self.current_explanation = ""
    yield
    
    try:
        from pixiu.services.ai_service import AIReportService
        from pixiu.services.explain_prompts import get_prompt
        
        prompt = get_prompt(
            concept,
            value=value,
            regime="趋势" if self.stock_regime == "trend" else "震荡",
            strategy=self.selected_strategies[0] if self.selected_strategies else "未知"
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

**Step 2: 提交**

```bash
git add pixiu/state.py
git commit -m "feat(state): add AI explanation methods"
```

---

## Task 9: 创建解释按钮组件

**Files:**
- Create: `pixiu/components/explain_button.py`

**Step 1: 创建组件**

```python
# pixiu/components/explain_button.py
import reflex as rx
from pixiu.state import State

def explain_button(concept: str, value: str = "") -> rx.Component:
    """带AI解释功能的按钮"""
    return rx.icon_button(
        rx.icon("help-circle", size=16),
        size="sm",
        variant="ghost",
        color_scheme="cyan",
        on_click=lambda: State.explain_concept(concept, value),
        cursor="pointer",
        _hover={"bg": "gray.700"},
    )

def metric_with_explain(label: str, value: str, concept: str) -> rx.Component:
    """带解释按钮的指标显示"""
    return rx.hstack(
        rx.text(label, font_size="sm", color="gray.400"),
        explain_button(concept, value),
        rx.text(value, font_weight="bold"),
        align_items="center",
        spacing="1",
    )

def explain_modal() -> rx.Component:
    """AI解释模态框"""
    return rx.modal(
        rx.modal_overlay(
            rx.modal_content(
                rx.modal_header(
                    rx.hstack(
                        rx.icon("sparkles", color="cyan.400"),
                        rx.text("AI 解释"),
                        align_items="center",
                    )
                ),
                rx.modal_body(
                    rx.cond(
                        State.ai_explaining,
                        rx.vstack(
                            rx.spinner(size="lg"),
                            rx.text("AI正在生成解释...", color="gray.400"),
                            align_items="center",
                            padding="2rem",
                        ),
                        rx.box(
                            rx.text(State.current_explanation, white_space="pre-wrap", line_height="1.8"),
                            max_height="60vh",
                            overflow_y="auto",
                        )
                    )
                ),
                rx.modal_footer(
                    rx.button(
                        "关闭",
                        on_click=State.close_explain_modal,
                        variant="outline",
                    ),
                ),
                bg="gray.800",
                border_color="gray.700",
            )
        ),
        is_open=State.explain_modal_open,
        size="2xl",
    )
```

**Step 2: 提交**

```bash
git add pixiu/components/explain_button.py
git commit -m "feat(ui): add explain button and modal components"
```

---

## Task 10: 集成AI解释到各页面

**Files:**
- Modify: `pixiu/pages/home.py`
- Modify: `pixiu/pixiu.py`

**Step 1: 在回测结果页添加解释按钮**

修改 `render_backtest_result` 函数，为每个指标添加解释按钮。

**Step 2: 在择势分析页添加解释按钮**

修改 `step_regime_analysis` 函数，为ADX、MA斜率、波动率添加解释按钮。

**Step 3: 在应用中添加全局解释模态框**

在 `pixiu/pixiu.py` 中导入并添加 `explain_modal` 组件。

**Step 4: 提交**

```bash
git add pixiu/pages/home.py pixiu/pixiu.py
git commit -m "feat(ui): integrate AI explanation buttons across all pages"
```

---

## Task 11: 添加依赖并完成测试

**Files:**
- Modify: `requirements.txt`

**Step 1: 添加依赖**

```
baostock>=0.8.8
kaleido>=0.2.1
```

**Step 2: 安装依赖并运行完整测试**

```bash
pip install baostock kaleido
pytest tests/ -v
```

**Step 3: 手动端到端测试**

```bash
reflex run
```
完整走一遍流程，验证所有功能正常。

**Step 4: 最终提交**

```bash
git add requirements.txt
git commit -m "chore: add baostock and kaleido dependencies"
```

---

## 完成清单

- [ ] Task 1: 添加时间范围选择状态
- [ ] Task 2: 更新股票搜索页面UI
- [ ] Task 3: 添加baostock数据源
- [ ] Task 4: 创建图表服务
- [ ] Task 5: 集成图表到回测结果页
- [ ] Task 6: 集成图表到择势分析页
- [ ] Task 7: 创建AI解释提示词
- [ ] Task 8: 添加AI解释功能
- [ ] Task 9: 创建解释按钮组件
- [ ] Task 10: 集成AI解释到各页面
- [ ] Task 11: 添加依赖并测试
