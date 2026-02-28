# Pixiu UI流程重构实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 重构Pixiu UI为6步分步向导模式，修复数据库问题，实现择势分析和智能策略推荐

**Architecture:** 采用Reflex响应式组件 + 状态驱动的步骤切换，数据层使用模拟数据兜底，择势分析结合市场+个股状态推荐策略

**Tech Stack:** Python, Reflex, SQLite (aiosqlite), Pandas, NumPy

---

## Task 1: 修复数据库表自动创建

**Files:**

- Modify: `pixiu/services/database.py:11-15`
- Modify: `pixiu/state.py:135-148`

**Step 1: Add ensure_tables method to Database class**

In `pixiu/services/database.py`, add after `__init__`:

```python
async def ensure_tables(self):
    """Ensure tables exist, create if not."""
    await self.create_tables()
```

**Step 2: Add database initialization on app startup**

In `pixiu/state.py`, modify `__init__` to call database init:

```python
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._load_strategies()
    self._load_settings()
    self._db_initialized = False
```

**Step 3: Add async init method to State class**

Add new method in `pixiu/state.py`:

```python
async def ensure_db_initialized(self):
    """Ensure database tables exist."""
    if self._db_initialized:
        return
    try:
        from pathlib import Path
        db_path = Path("data/stocks.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db = Database(str(db_path))
        await db.ensure_tables()
        self._db_initialized = True
    except Exception as e:
        self.error_message = f"数据库初始化失败: {str(e)}"
```

**Step 4: Call ensure_db_initialized in run_backtest**

Modify `run_backtest` in `pixiu/state.py` to add at start:

```python
async def run_backtest(self):
    await self.ensure_db_initialized()
    # ... rest of the method
```

**Step 5: Test database initialization**

Run: `cd pixiu && python -c "import asyncio; from pixiu.services.database import Database; asyncio.run(Database('data/test.db').ensure_tables())"`

Expected: No error, test.db created with tables

**Step 6: Commit**

```bash
git add pixiu/services/database.py pixiu/state.py
git commit -m "fix: ensure database tables are created before use"
```

---

## Task 2: 添加步骤状态管理

**Files:**

- Modify: `pixiu/state.py:15-50`

**Step 1: Add step constants and state variable**

Add after `class State(rx.State):`:

```python
    STEP_MARKET = 1
    STEP_SEARCH = 2
    STEP_REGIME = 3
    STEP_STRATEGY = 4
    STEP_CONFIG = 5
    STEP_RESULT = 6

    current_step: int = 1
    max_step: int = 1
```

**Step 2: Add step navigation methods**

Add methods in State class:

```python
    def go_to_step(self, step: int):
        if step <= self.max_step:
            self.current_step = step

    def next_step(self):
        if self.current_step < 6:
            self.current_step += 1
            if self.current_step > self.max_step:
                self.max_step = self.current_step

    def prev_step(self):
        if self.current_step > 1:
            self.current_step -= 1

    def reset_flow(self):
        self.current_step = 1
        self.max_step = 1
        self.selected_stock = ""
        self.selected_stock_name = ""
        self.selected_strategies = []
        self.backtest_results = []
        self.regime_analysis = {}
        self.market_regime = "unknown"
        self.stock_regime = "unknown"
```

**Step 3: Update select_stock to advance step**

Modify `select_stock` method:

```python
async def select_stock(self, code: str):
    self.selected_stock = code
    self.selected_stock_name = ""
    for stock in self.search_results:
        if stock["code"] == code:
            self.selected_stock_name = stock["name"]
            break
    if self.selected_stock_name:
        self.current_step = self.STEP_REGIME
        self.max_step = max(self.max_step, self.STEP_REGIME)
    yield
```

**Step 4: Commit**

```bash
git add pixiu/state.py
git commit -m "feat: add step state management for wizard flow"
```

---

## Task 3: 实现择势分析逻辑

**Files:**

- Modify: `pixiu/state.py:292-311`
- Modify: `pixiu/analysis/regime_detector.py:111-122`

**Step 1: Add market index regime analysis**

In `pixiu/state.py`, add new state variable:

```python
    market_index_data: Dict = {}
    using_mock_data: bool = False
```

**Step 2: Add get_market_index method**

Add method in State class:

```python
    async def _get_market_index_data(self) -> pd.DataFrame:
        """Get market index data (上证指数 for A股, etc.)"""
        from pixiu.services.data_service import DataService
        db = Database("data/stocks.db")
        data_service = DataService(db, use_mock=True)

        market_codes = {
            "A股": "000001",  # 上证指数
            "港股": "HSI",    # 恒生指数
            "美股": "DJI",    # 道琼斯
        }
        code = market_codes.get(self.current_market, "000001")
        return data_service._generate_mock_history(code)
```

**Step 3: Rewrite analyze_regime method**

Replace existing `analyze_regime`:

```python
    async def analyze_regime(self):
        """Analyze both market and stock regime"""
        from pixiu.analysis import MarketRegimeDetector
        from pixiu.services.data_service import DataService

        self.is_loading = True
        self.loading_message = "分析市场和个股状态..."
        self.error_message = ""
        yield

        try:
            await self.ensure_db_initialized()

            db = Database("data/stocks.db")
            data_service = DataService(db, use_mock=True)
            detector = MarketRegimeDetector()

            # Analyze market index
            market_df = await self._get_market_index_data()
            if market_df is not None and not market_df.empty:
                market_analysis = detector.get_analysis_detail(market_df)
                self.market_regime = market_analysis["regime"]
                self.market_index_data = market_analysis

            # Analyze selected stock
            if self.selected_stock:
                df = await data_service.get_cached_data(self.selected_stock)
                if df is None or df.empty:
                    success, _ = await data_service.download_and_save(
                        self.selected_stock,
                        self.current_market
                    )
                    if success:
                        df = await data_service.get_cached_data(self.selected_stock)

                if df is not None and not df.empty:
                    stock_analysis = detector.get_analysis_detail(df)
                    self.stock_regime = stock_analysis["regime"]
                    self.regime_analysis = stock_analysis
                    self.using_mock_data = True

            # Advance to strategy selection
            self.current_step = self.STEP_STRATEGY
            self.max_step = max(self.max_step, self.STEP_STRATEGY)

        except Exception as e:
            self.error_message = f"择势分析失败: {str(e)}"
        finally:
            self.is_loading = False
            self.loading_message = ""
        yield
```

**Step 4: Commit**

```bash
git add pixiu/state.py
git commit -m "feat: implement market+stock regime analysis"
```

---

## Task 4: 实现智能策略推荐

**Files:**

- Modify: `pixiu/state.py:54-66`

**Step 1: Add strategy recommendation mapping**

Add constant after STEP constants:

```python
    REGIME_STRATEGY_MAP = {
        ("trend", "trend"): ["趋势强度策略", "均线策略", "动量策略"],
        ("trend", "range"): ["网格交易策略", "RSI策略", "波动率套利策略"],
        ("range", "trend"): ["趋势强度策略", "动量策略"],
        ("range", "range"): ["网格交易策略", "RSI策略", "波动率套利策略", "均值回归策略"],
    }
```

**Step 2: Add recommended_strategies state variable and computed var**

Add state variable:

```python
    recommended_strategies: List[str] = []
```

Add computed var:

```python
    @rx.var
    def regime_recommendations(self) -> List[str]:
        """Get strategy recommendations based on regime analysis."""
        key = (self.market_regime, self.stock_regime)
        return self.REGIME_STRATEGY_MAP.get(key, [])
```

**Step 3: Update analyze_regime to set recommendations**

Modify end of `analyze_regime` method, before advancing step:

```python
            # Set recommendations
            self.recommended_strategies = self.regime_recommendations
```

**Step 4: Commit**

```bash
git add pixiu/state.py
git commit -m "feat: add intelligent strategy recommendation based on regime"
```

---

## Task 5: 创建步骤指示器组件

**Files:**

- Modify: `pixiu/pages/home.py:7-30`

**Step 1: Add step indicator component**

Add new function before `page()`:

```python
def step_indicator() -> rx.Component:
    """Step progress indicator."""
    steps = [
        ("选择市场", State.STEP_MARKET),
        ("搜索股票", State.STEP_SEARCH),
        ("择势分析", State.STEP_REGIME),
        ("选择策略", State.STEP_STRATEGY),
        ("配置参数", State.STEP_CONFIG),
        ("查看结果", State.STEP_RESULT),
    ]
    return rx.hstack(
        *[
            rx.box(
                rx.hstack(
                    rx.badge(
                        str(i + 1),
                        color_scheme="cyan" if State.current_step > i else "gray",
                        variant="solid" if State.current_step > i else "outline",
                    ),
                    rx.text(
                        name,
                        font_size="0.75rem",
                        color="cyan.400" if State.current_step > i else "gray.500",
                    ),
                    spacing="2",
                ),
                padding_x="0.5rem",
            )
            for i, (name, _) in enumerate(steps)
        ],
        spacing="1",
        width="100%",
        overflow_x="auto",
        padding_y="0.5rem",
    )
```

**Step 2: Add step indicator to page**

In `page()` function, add after the header hstack:

```python
            step_indicator(),
            rx.divider(),
```

**Step 3: Commit**

```bash
git add pixiu/pages/home.py
git commit -m "feat: add step progress indicator component"
```

---

## Task 6: 实现市场选择步骤

**Files:**

- Modify: `pixiu/pages/home.py:30-62`

**Step 1: Create market selection component**

Add new function:

```python
def step_market_selection() -> rx.Component:
    """Step 1: Market selection."""
    return rx.box(
        rx.vstack(
            rx.text("选择市场", font_size="lg", font_weight="bold"),
            rx.text("选择要分析的股票市场", font_size="sm", color="gray.400"),
            rx.hstack(
                rx.button(
                    "A股",
                    size="lg",
                    variant=rx.cond(State.current_market == "A股", "solid", "outline"),
                    color_scheme=rx.cond(State.current_market == "A股", "cyan", "gray"),
                    on_click=State.set_market_a,
                ),
                rx.button(
                    "港股",
                    size="lg",
                    variant=rx.cond(State.current_market == "港股", "solid", "outline"),
                    color_scheme=rx.cond(State.current_market == "港股", "cyan", "gray"),
                    on_click=State.set_market_hk,
                ),
                rx.button(
                    "美股",
                    size="lg",
                    variant=rx.cond(State.current_market == "美股", "solid", "outline"),
                    color_scheme=rx.cond(State.current_market == "美股", "cyan", "gray"),
                    on_click=State.set_market_us,
                ),
                spacing="4",
            ),
            spacing="3",
        ),
        padding="1.5rem",
        border="1px solid gray.700",
        border_radius="lg",
        width="100%",
    )
```

**Step 2: Commit**

```bash
git add pixiu/pages/home.py
git commit -m "feat: add market selection step component"
```

---

## Task 7: 实现股票搜索步骤

**Files:**

- Modify: `pixiu/pages/home.py`

**Step 1: Create stock search component**

Add new function:

```python
def step_stock_search() -> rx.Component:
    """Step 2: Search and select stock."""
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
                    size="lg",
                ),
                rx.button(
                    "搜索",
                    on_click=State.search_stocks,
                    color_scheme="cyan",
                    size="lg",
                    is_loading=State.is_loading,
                ),
                width="100%",
                spacing="2",
            ),

            rx.cond(
                State.is_loading,
                rx.hstack(rx.spinner(), rx.text(State.loading_message)),
            ),

            rx.cond(
                State.error_message != "",
                rx.text(State.error_message, color="red.400"),
            ),

            rx.box(
                rx.foreach(State.search_results, render_search_result),
                max_height="250px",
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

            spacing="3",
        ),
        padding="1.5rem",
        border="1px solid gray.700",
        border_radius="lg",
        width="100%",
    )
```

**Step 2: Update render_search_result for better UX**

Replace existing `render_search_result`:

```python
def render_search_result(stock: dict) -> rx.Component:
    """Render a search result item."""
    is_selected = State.selected_stock == stock["code"]
    return rx.box(
        rx.hstack(
            rx.badge(stock["code"], color_scheme="gray"),
            rx.text(stock["name"], font_weight="medium"),
            rx.spacer(),
            rx.cond(
                is_selected,
                rx.badge("已选", color_scheme="green"),
                rx.text("点击选择", font_size="sm", color="gray.500"),
            ),
        ),
        padding="0.75rem",
        border=rx.cond(is_selected, "1px solid cyan.500", "1px solid transparent"),
        border_radius="md",
        bg=rx.cond(is_selected, "gray.800", "transparent"),
        cursor="pointer",
        on_click=State.select_stock(stock["code"]),
        _hover={"bg": "gray.800"},
    )
```

**Step 3: Commit**

```bash
git add pixiu/pages/home.py
git commit -m "feat: add stock search step component with better UX"
```

---

## Task 8: 实现择势分析步骤

**Files:**

- Modify: `pixiu/pages/home.py`

**Step 1: Create regime analysis component**

Add new function:

```python
def step_regime_analysis() -> rx.Component:
    """Step 3: Market and stock regime analysis."""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text("择势分析", font_size="lg", font_weight="bold"),
                rx.spacer(),
                rx.button(
                    "开始分析",
                    on_click=State.analyze_regime,
                    color_scheme="cyan",
                    is_loading=State.is_loading,
                ),
            ),

            rx.text("分析市场状态和个股走势，智能推荐策略", font_size="sm", color="gray.400"),

            rx.cond(
                State.is_loading,
                rx.vstack(
                    rx.spinner(size="lg"),
                    rx.text(State.loading_message),
                    align_items="center",
                    padding="2rem",
                ),
            ),

            rx.cond(
                State.stock_regime != "unknown",
                rx.vstack(
                    rx.grid(
                        rx.box(
                            rx.text("大盘状态", font_size="sm", color="gray.400"),
                            rx.hstack(
                                rx.badge(
                                    rx.cond(State.market_regime == "trend", "趋势", "震荡"),
                                    color_scheme=rx.cond(State.market_regime == "trend", "green", "yellow"),
                                ),
                                rx.text(
                                    rx.cond(State.market_regime == "trend", "适合跟踪策略", "适合均值回归"),
                                    font_size="sm",
                                ),
                            ),
                            padding="1rem",
                            bg="gray.800",
                            border_radius="md",
                        ),
                        rx.box(
                            rx.text("个股状态", font_size="sm", color="gray.400"),
                            rx.hstack(
                                rx.badge(
                                    rx.cond(State.stock_regime == "trend", "趋势", "震荡"),
                                    color_scheme=rx.cond(State.stock_regime == "trend", "green", "yellow"),
                                ),
                                rx.text(
                                    rx.cond(State.stock_regime == "trend", "趋势明显", "横盘整理"),
                                    font_size="sm",
                                ),
                            ),
                            padding="1rem",
                            bg="gray.800",
                            border_radius="md",
                        ),
                        columns="2",
                        spacing="4",
                        width="100%",
                    ),

                    rx.box(
                        rx.text("分析指标", font_size="sm", color="gray.400", margin_bottom="0.5rem"),
                        rx.hstack(
                            rx.text(f"ADX: {State.regime_analysis.get('adx', 0):.1f}", font_size="sm"),
                            rx.text("|", color="gray.600"),
                            rx.text(f"MA斜率: {State.regime_analysis.get('ma_slope', 0):.4f}", font_size="sm"),
                            rx.text("|", color="gray.600"),
                            rx.text(f"波动率: {State.regime_analysis.get('volatility', 0):.4f}", font_size="sm"),
                        ),
                        padding="0.75rem",
                        bg="gray.900",
                        border_radius="md",
                    ),

                    rx.cond(
                        State.using_mock_data,
                        rx.badge("使用模拟数据", color_scheme="yellow", variant="subtle"),
                    ),

                    spacing="4",
                ),
            ),

            spacing="3",
        ),
        padding="1.5rem",
        border="1px solid gray.700",
        border_radius="lg",
        width="100%",
    )
```

**Step 2: Commit**

```bash
git add pixiu/pages/home.py
git commit -m "feat: add regime analysis step with market+stock indicators"
```

---

## Task 9: 实现策略选择步骤（带推荐）

**Files:**

- Modify: `pixiu/pages/home.py:144-154`

**Step 1: Update render_strategy with recommendation badge**

Replace existing `render_strategy`:

```python
def render_strategy(s: dict) -> rx.Component:
    """Render a strategy card with selection state."""
    is_selected = State.selected_strategies.contains(s["name"])
    is_recommended = State.recommended_strategies.contains(s["name"])

    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.checkbox(
                    is_checked=is_selected,
                    on_change=State.toggle_strategy(s["name"]),
                ),
                rx.text(s["name"], font_weight="bold"),
                rx.cond(
                    is_recommended,
                    rx.badge("推荐", color_scheme="green", size="sm"),
                ),
                width="100%",
                justify="between",
            ),
            rx.text(
                s.get("description", ""),
                font_size="sm",
                color="gray.500",
            ),
            align_items="start",
            spacing="1",
        ),
        padding="0.75rem",
        border=rx.cond(
            is_selected,
            "2px solid cyan.500",
            rx.cond(
                is_recommended,
                "1px solid green.700",
                "1px solid gray.700"
            )
        ),
        border_radius="md",
        cursor="pointer",
        bg=rx.cond(is_selected, "gray.800", "transparent"),
        on_click=State.toggle_strategy(s["name"]),
        _hover={"bg": "gray.800"},
    )
```

**Step 2: Create strategy selection step component**

Add new function:

```python
def step_strategy_selection() -> rx.Component:
    """Step 4: Strategy selection with recommendations."""
    return rx.box(
        rx.vstack(
            rx.text("选择策略", font_size="lg", font_weight="bold"),

            rx.cond(
                State.recommended_strategies.length() > 0,
                rx.box(
                    rx.text("基于择势分析推荐:", font_size="sm", color="green.400", margin_bottom="0.5rem"),
                    rx.text(
                        rx.cond(
                            State.market_regime == "trend",
                            rx.cond(
                                State.stock_regime == "trend",
                                "大盘+个股均为趋势行情，推荐趋势跟踪策略",
                                "大盘趋势但个股震荡，推荐网格或RSI策略"
                            ),
                            rx.cond(
                                State.stock_regime == "trend",
                                "大盘震荡但个股有趋势，可尝试趋势策略",
                                "大盘+个股均震荡，推荐均值回归策略"
                            )
                        ),
                        font_size="sm",
                        color="gray.400",
                    ),
                    padding="0.75rem",
                    bg="green.900",
                    border_radius="md",
                    margin_bottom="1rem",
                ),
            ),

            rx.grid(
                rx.foreach(State.available_strategies, render_strategy),
                columns="2",
                spacing="3",
                width="100%",
            ),

            rx.hstack(
                rx.text(f"已选 {State.selected_strategies.length()} 个策略", color="gray.400"),
            ),

            spacing="3",
        ),
        padding="1.5rem",
        border="1px solid gray.700",
        border_radius="lg",
        width="100%",
    )
```

**Step 3: Commit**

```bash
git add pixiu/pages/home.py
git commit -m "feat: add strategy selection with recommendation badges"
```

---

## Task 10: 实现参数配置步骤

**Files:**

- Modify: `pixiu/pages/home.py`

**Step 1: Create config step component**

Add new function:

```python
def step_config() -> rx.Component:
    """Step 5: Backtest configuration."""
    return rx.box(
        rx.vstack(
            rx.text("配置参数", font_size="lg", font_weight="bold"),

            rx.vstack(
                rx.hstack(
                    rx.text("初始资金:", width="100px"),
                    rx.input(
                        value=State.initial_capital,
                        on_change=State.set_initial_capital,
                        type="number",
                        width="200px",
                    ),
                ),
                rx.hstack(
                    rx.text("手续费率:", width="100px"),
                    rx.input(
                        value=State.commission_rate,
                        on_change=State.set_commission_rate,
                        type="number",
                        width="200px",
                    ),
                ),
                rx.hstack(
                    rx.text("仓位比例:", width="100px"),
                    rx.input(
                        value=State.position_size,
                        on_change=State.set_position_size,
                        type="number",
                        width="200px",
                    ),
                ),
                spacing="3",
            ),

            rx.button(
                "开始回测",
                on_click=State.run_backtest,
                color_scheme="cyan",
                size="lg",
                width="100%",
                is_loading=State.is_loading,
            ),

            rx.cond(
                State.is_loading,
                rx.vstack(
                    rx.progress(value=State.progress, width="100%"),
                    rx.text(State.loading_message, color="gray.400"),
                ),
            ),

            spacing="4",
        ),
        padding="1.5rem",
        border="1px solid gray.700",
        border_radius="lg",
        width="100%",
    )
```

**Step 2: Update run_backtest to advance to result step**

In `pixiu/state.py`, modify `run_backtest` end:

```python
            self.progress = 100
            self.loading_message = "回测完成"
            self.current_step = self.STEP_RESULT
            self.max_step = max(self.max_step, self.STEP_RESULT)
            yield
```

**Step 3: Commit**

```bash
git add pixiu/pages/home.py pixiu/state.py
git commit -m "feat: add backtest configuration step"
```

---

## Task 11: 实现结果展示步骤

**Files:**

- Modify: `pixiu/pages/home.py`

**Step 1: Create result display component**

Add new function:

```python
def render_backtest_result(result: dict) -> rx.Component:
    """Render a single backtest result card."""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text(result["strategy"], font_weight="bold", font_size="lg"),
                rx.spacer(),
                rx.badge(
                    f"+{result['total_return']:.1f}%",
                    color_scheme=rx.cond(result["total_return"] > 0, "green", "red"),
                ),
            ),

            rx.grid(
                rx.box(
                    rx.text("年化收益", font_size="sm", color="gray.400"),
                    rx.text(f"{result['annualized_return']:.1f}%", font_weight="bold"),
                ),
                rx.box(
                    rx.text("最大回撤", font_size="sm", color="gray.400"),
                    rx.text(f"{result['max_drawdown']:.1f}%", color="red.400"),
                ),
                rx.box(
                    rx.text("夏普比率", font_size="sm", color="gray.400"),
                    rx.text(f"{result['sharpe_ratio']:.2f}"),
                ),
                rx.box(
                    rx.text("胜率", font_size="sm", color="gray.400"),
                    rx.text(f"{result['win_rate']:.1f}%"),
                ),
                columns="4",
                spacing="4",
            ),

            spacing="3",
        ),
        padding="1rem",
        bg="gray.800",
        border_radius="md",
        width="100%",
    )


def step_results() -> rx.Component:
    """Step 6: Display backtest results."""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text("回测结果", font_size="lg", font_weight="bold"),
                rx.spacer(),
                rx.button(
                    "重新开始",
                    on_click=State.reset_flow,
                    variant="outline",
                    color_scheme="gray",
                ),
            ),

            rx.foreach(State.backtest_results, render_backtest_result),

            rx.cond(
                State.backtest_results.length() == 0,
                rx.text("暂无回测结果", color="gray.400"),
            ),

            spacing="4",
        ),
        padding="1.5rem",
        border="1px solid gray.700",
        border_radius="lg",
        width="100%",
    )
```

**Step 2: Commit**

```bash
git add pixiu/pages/home.py
git commit -m "feat: add backtest results display step"
```

---

## Task 12: 整合向导流程到主页面

**Files:**

- Modify: `pixiu/pages/home.py:7-129`

**Step 1: Rewrite page function to use step components**

Replace the entire `page()` function:

```python
def page() -> rx.Component:
    """Main page with wizard flow."""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.heading("Pixiu 量化分析", size="6"),
                rx.spacer(),
                rx.badge("v0.3.0", color_scheme="cyan"),
                width="100%",
            ),

            step_indicator(),
            rx.divider(),

            rx.cond(
                State.current_step == State.STEP_MARKET,
                step_market_selection(),
            ),
            rx.cond(
                State.current_step == State.STEP_SEARCH,
                step_stock_search(),
            ),
            rx.cond(
                State.current_step == State.STEP_REGIME,
                step_regime_analysis(),
            ),
            rx.cond(
                State.current_step == State.STEP_STRATEGY,
                step_strategy_selection(),
            ),
            rx.cond(
                State.current_step == State.STEP_CONFIG,
                step_config(),
            ),
            rx.cond(
                State.current_step == State.STEP_RESULT,
                step_results(),
            ),

            rx.hstack(
                rx.button(
                    "上一步",
                    on_click=State.prev_step,
                    variant="outline",
                    is_disabled=State.current_step == 1,
                ),
                rx.spacer(),
                rx.button(
                    "下一步",
                    on_click=State.next_step,
                    color_scheme="cyan",
                    is_disabled=State.current_step >= State.max_step,
                ),
                width="100%",
            ),

            rx.spacer(),

            rx.hstack(
                rx.text("Pixiu 2024", color="gray.400"),
                rx.spacer(),
                rx.link("设置", href="/settings", color_scheme="cyan"),
                width="100%",
            ),

            spacing="4",
            width="100%",
        ),
        width="100%",
        max_width="800px",
        margin="0 auto",
        padding="2rem",
        min_height="100vh",
        bg="gray.950",
    )
```

**Step 2: Commit**

```bash
git add pixiu/pages/home.py
git commit -m "feat: integrate wizard flow into main page"
```

---

## Task 13: 修复State中的contains方法

**Files:**

- Modify: `pixiu/state.py`

**Step 1: Add contains methods for list state variables**

In State class, add helper vars:

```python
    @rx.var
    def has_selected_strategies(self) -> bool:
        return len(self.selected_strategies) > 0

    def is_strategy_selected(self, name: str) -> bool:
        return name in self.selected_strategies

    def is_strategy_recommended(self, name: str) -> bool:
        return name in self.recommended_strategies
```

**Step 2: Update render_strategy to use helper methods**

In `pixiu/pages/home.py`, update `render_strategy`:

```python
def render_strategy(s: dict) -> rx.Component:
    """Render a strategy card with selection state."""
    name = s["name"]

    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.checkbox(
                    is_checked=State.selected_strategies.contains(name),
                ),
                rx.text(name, font_weight="bold"),
                rx.cond(
                    State.recommended_strategies.contains(name),
                    rx.badge("推荐", color_scheme="green", size="sm"),
                ),
                width="100%",
                justify="between",
            ),
            rx.text(
                s.get("description", ""),
                font_size="sm",
                color="gray.500",
            ),
            align_items="start",
            spacing="1",
        ),
        padding="0.75rem",
        border=rx.cond(
            State.selected_strategies.contains(name),
            "2px solid cyan.500",
            rx.cond(
                State.recommended_strategies.contains(name),
                "1px solid green.700",
                "1px solid gray.700"
            )
        ),
        border_radius="md",
        cursor="pointer",
        bg=rx.cond(State.selected_strategies.contains(name), "gray.800", "transparent"),
        on_click=State.toggle_strategy(name),
        _hover={"bg": "gray.800"},
    )
```

**Step 3: Commit**

```bash
git add pixiu/state.py pixiu/pages/home.py
git commit -m "fix: use contains method for list state checks in Reflex"
```

---

## Task 14: 测试和修复

**Step 1: Run the application**

```bash
cd pixiu && reflex run
```

Expected: App starts without errors

**Step 2: Test wizard flow**

Manual test:

1. Select market → advances to step 2
2. Search stock → select stock → advances to step 3
3. Analyze regime → shows analysis → advances to step 4
4. Select strategies → shows selection state
5. Configure → run backtest → advances to step 6
6. View results → shows backtest results

**Step 3: Fix any issues**

Based on testing, fix any issues found.

---

## Summary

This plan transforms Pixiu from a single-page form into a guided 6-step wizard:

1. **Task 1**: Database auto-initialization
2. **Task 2**: Step state management
3. **Task 3**: Regime analysis logic
4. **Task 4**: Strategy recommendation
5. **Task 5**: Step indicator UI
6. **Tasks 6-11**: Individual step components
7. **Task 12**: Main page integration
8. **Task 13**: Reflex compatibility fixes
9. **Task 14**: Testing and polish

Each task follows TDD principles where applicable and includes verification steps.
