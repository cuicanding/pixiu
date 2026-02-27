# Pixiu Phase 2 Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform Pixiu from placeholder UI to fully functional quantitative analysis app with real data, backtesting, AI reports, and beautiful dark tech-themed interface.

**Architecture:** Single-page app with Reflex framework. State class connects UI to backend services (DataService, BacktestEngine, AIService). Dark theme with cyan/purple accents. Plotly charts for visualization.

**Tech Stack:** Reflex 0.8.x, Plotly, Akshare, ZhipuAI, SQLite

---

## Phase 1: Foundation - Theme & Components

### Task 1.1: Create Theme Configuration

**Files:**
- Create: `pixiu/theme.py`

**Step 1: Create theme module with dark color palette**

```python
"""Dark tech theme configuration for Pixiu."""

COLORS = {
    "bg_primary": "#0a0a0f",
    "bg_secondary": "#12121a",
    "bg_card": "#1a1a24",
    "bg_hover": "#252532",
    "accent_cyan": "#00d9ff",
    "accent_purple": "#7c3aed",
    "accent_green": "#10b981",
    "accent_red": "#ef4444",
    "accent_orange": "#f59e0b",
    "text_primary": "#ffffff",
    "text_secondary": "#a0a0b0",
    "text_muted": "#6b7280",
    "border": "#2a2a3a",
}

def get_theme():
    return rx.theme(
        appearance="dark",
        has_background=True,
        radius="medium",
        accent_color="cyan",
    )
```

**Step 2: Commit**

```bash
git add pixiu/theme.py
git commit -m "feat: add dark tech theme configuration"
```

---

### Task 1.2: Create Metric Card Component

**Files:**
- Create: `pixiu/components/__init__.py`
- Create: `pixiu/components/metric_card.py`

**Step 1: Create components package**

```python
# pixiu/components/__init__.py
"""Reusable UI components."""
from .metric_card import metric_card

__all__ = ["metric_card"]
```

**Step 2: Create metric card component**

```python
# pixiu/components/metric_card.py
"""Metric display card component."""
import reflex as rx
from typing import Optional


def metric_card(
    title: str,
    value: str,
    subtitle: Optional[str] = None,
    color: str = "cyan",
) -> rx.Component:
    """Display a metric with title and value.
    
    Args:
        title: Metric label
        value: Metric value to display
        subtitle: Optional subtitle/description
        color: Accent color (cyan, green, red, orange)
    """
    color_map = {
        "cyan": "#00d9ff",
        "green": "#10b981",
        "red": "#ef4444",
        "orange": "#f59e0b",
        "purple": "#7c3aed",
    }
    
    return rx.box(
        rx.vstack(
            rx.text(
                title,
                font_size="0.875rem",
                color="#a0a0b0",
            ),
            rx.text(
                value,
                font_size="1.5rem",
                font_weight="bold",
                color=color_map.get(color, "#00d9ff"),
            ),
            rx.cond(
                subtitle != None,
                rx.text(
                    subtitle,
                    font_size="0.75rem",
                    color="#6b7280",
                ),
            ),
            spacing="0.5rem",
            align_items="flex-start",
        ),
        bg="#1a1a24",
        padding="1rem",
        border_radius="0.5rem",
        border="1px solid #2a2a3a",
        width="100%",
    )
```

**Step 3: Commit**

```bash
git add pixiu/components/
git commit -m "feat: add metric card component"
```

---

### Task 1.3: Create Stock Card Component

**Files:**
- Create: `pixiu/components/stock_card.py`
- Modify: `pixiu/components/__init__.py`

**Step 1: Create stock card component**

```python
# pixiu/components/stock_card.py
"""Stock search result card component."""
import reflex as rx


def stock_card(stock: dict, on_click_handler) -> rx.Component:
    """Display a stock search result.
    
    Args:
        stock: Dict with 'code' and 'name' keys
        on_click_handler: Event handler for selection
    """
    return rx.box(
        rx.hstack(
            rx.text(
                stock["code"],
                font_weight="bold",
                color="#00d9ff",
                min_width="80px",
            ),
            rx.text(
                stock["name"],
                color="#ffffff",
            ),
            rx.spacer(),
            rx.text(
                stock.get("market", ""),
                font_size="0.75rem",
                color="#6b7280",
            ),
            width="100%",
        ),
        padding="0.75rem 1rem",
        cursor="pointer",
        border_radius="0.375rem",
        on_click=on_click_handler,
        _hover={
            "bg": "#252532",
        },
    )
```

**Step 2: Update __init__.py**

```python
# pixiu/components/__init__.py
"""Reusable UI components."""
from .metric_card import metric_card
from .stock_card import stock_card

__all__ = ["metric_card", "stock_card"]
```

**Step 3: Commit**

```bash
git add pixiu/components/
git commit -m "feat: add stock card component"
```

---

### Task 1.4: Create Strategy Card Component

**Files:**
- Create: `pixiu/components/strategy_card.py`
- Modify: `pixiu/components/__init__.py`

**Step 1: Create strategy card component**

```python
# pixiu/components/strategy_card.py
"""Strategy selection card component."""
import reflex as rx


def strategy_card(
    strategy: dict,
    is_selected: bool,
    on_click_handler,
) -> rx.Component:
    """Display a selectable strategy card.
    
    Args:
        strategy: Dict with 'name' and 'description' keys
        is_selected: Whether this strategy is selected
        on_click_handler: Event handler for toggle
    """
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text(
                    "✓" if is_selected else "○",
                    color="#10b981" if is_selected else "#6b7280",
                    font_size="1.25rem",
                ),
                rx.text(
                    strategy["name"],
                    font_weight="bold",
                    color="#ffffff",
                ),
                spacing="0.5rem",
                width="100%",
            ),
            rx.text(
                strategy.get("description", ""),
                font_size="0.75rem",
                color="#a0a0b0",
            ),
            spacing="0.5rem",
            align_items="flex-start",
        ),
        padding="1rem",
        border_radius="0.5rem",
        cursor="pointer",
        border=f"2px solid {'#10b981' if is_selected else '#2a2a3a'}",
        bg="#1a1a24" if is_selected else "transparent",
        on_click=on_click_handler,
        _hover={
            "border_color": "#00d9ff",
        },
    )
```

**Step 2: Update __init__.py**

```python
# pixiu/components/__init__.py
"""Reusable UI components."""
from .metric_card import metric_card
from .stock_card import stock_card
from .strategy_card import strategy_card

__all__ = ["metric_card", "stock_card", "strategy_card"]
```

**Step 3: Commit**

```bash
git add pixiu/components/
git commit -m "feat: add strategy card component"
```

---

## Phase 2: State Management - Connect Real Services

### Task 2.1: Rewrite State Class with Real Service Integration

**Files:**
- Modify: `pixiu/state.py` (complete rewrite)

**Step 1: Write new state class with real service calls**

```python
# pixiu/state.py
"""Global state management - connected to real services."""

import reflex as rx
from typing import List, Dict, Optional
from pathlib import Path

from pixiu.services.database import Database
from pixiu.services.data_service import DataService
from pixiu.services.backtest_service import BacktestEngine, BacktestConfig
from pixiu.services.ai_service import AIService
from pixiu.strategies import get_all_strategies
from pixiu.config import config


class State(rx.State):
    """Application global state with real service integration."""
    
    # UI State
    is_loading: bool = False
    loading_message: str = ""
    progress: int = 0
    error_message: str = ""
    
    # Market & Search
    current_market: str = "A股"
    search_keyword: str = ""
    search_results: List[Dict] = []
    
    # Stock Selection
    selected_stock: str = ""
    selected_stock_name: str = ""
    stock_info: Dict = {}
    
    # Strategy Selection
    available_strategies: List[Dict] = []
    selected_strategies: List[str] = []
    
    # Backtest Configuration
    initial_capital: float = 100000.0
    commission_rate: float = 0.0003
    position_size: float = 0.95
    
    # Backtest Results
    backtest_results: List[Dict] = []
    
    # AI Report
    ai_report: str = ""
    ai_generating: bool = False
    
    # Settings
    glm_api_key: str = ""
    
    def __init__(self):
        super().__init__()
        self._load_strategies()
        self._load_settings()
    
    def _load_strategies(self):
        """Load available strategies from registry."""
        strategies = get_all_strategies()
        self.available_strategies = [
            {
                "name": s.name,
                "description": s.description,
            }
            for s in strategies
        ]
    
    def _load_settings(self):
        """Load settings from config."""
        self.glm_api_key = config.glm_api_key or ""
        self.initial_capital = config.initial_capital
        self.commission_rate = config.commission_rate
        self.position_size = config.position_size
    
    # Market Selection
    def set_market_a(self):
        self.current_market = "A股"
    
    def set_market_hk(self):
        self.current_market = "港股"
    
    def set_market_us(self):
        self.current_market = "美股"
    
    # Search
    def set_search_keyword(self, keyword: str):
        self.search_keyword = keyword
    
    async def search_stocks(self):
        """Search stocks using real DataService."""
        if not self.search_keyword:
            return
        
        self.is_loading = True
        self.error_message = ""
        self.search_results = []
        yield
        
        try:
            db_path = Path("data/stocks.db")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            db = Database(str(db_path))
            data_service = DataService(db)
            
            results = await data_service.search_stocks(
                self.search_keyword,
                self.current_market
            )
            
            self.search_results = [
                {"code": s.code, "name": s.name, "market": s.market}
                for s in results[:10]
            ]
        except Exception as e:
            self.error_message = f"搜索失败: {str(e)}"
        finally:
            self.is_loading = False
            yield
    
    # Stock Selection
    async def select_stock(self, code: str):
        """Select a stock and load its data."""
        self.selected_stock = code
        self.selected_stock_name = ""
        
        for stock in self.search_results:
            if stock["code"] == code:
                self.selected_stock_name = stock["name"]
                break
        
        yield
    
    # Strategy Selection
    def toggle_strategy(self, strategy_name: str):
        """Toggle strategy selection."""
        if strategy_name in self.selected_strategies:
            self.selected_strategies.remove(strategy_name)
        else:
            self.selected_strategies.append(strategy_name)
    
    # Backtest
    async def run_backtest(self):
        """Run backtest using real BacktestEngine."""
        if not self.selected_stock or not self.selected_strategies:
            self.error_message = "请先选择股票和策略"
            yield
            return
        
        self.is_loading = True
        self.progress = 0
        self.error_message = ""
        self.backtest_results = []
        yield
        
        try:
            db = Database("data/stocks.db")
            data_service = DataService(db)
            
            self.loading_message = "加载股票数据..."
            yield
            
            df = await data_service.get_cached_data(self.selected_stock)
            
            if df.empty:
                success, _ = await data_service.download_and_save(
                    self.selected_stock,
                    self.current_market
                )
                if success:
                    df = await data_service.get_cached_data(self.selected_stock)
            
            if df.empty:
                self.error_message = "无法获取股票数据"
                self.is_loading = False
                yield
                return
            
            from pixiu.strategies import get_strategy
            
            backtest_config = BacktestConfig(
                initial_capital=self.initial_capital,
                commission_rate=self.commission_rate,
                position_size=self.position_size,
            )
            
            for i, strategy_name in enumerate(self.selected_strategies):
                self.loading_message = f"回测策略: {strategy_name}"
                self.progress = int((i / len(self.selected_strategies)) * 80)
                yield
                
                strategy = get_strategy(strategy_name)
                engine = BacktestEngine(backtest_config)
                result = engine.run(df, strategy)
                
                self.backtest_results.append({
                    "strategy": strategy_name,
                    "total_return": result.total_return,
                    "annualized_return": result.annualized_return,
                    "max_drawdown": result.max_drawdown,
                    "sharpe_ratio": result.sharpe_ratio,
                    "win_rate": result.win_rate,
                    "profit_loss_ratio": result.profit_loss_ratio,
                    "trades": [
                        {
                            "date": t.date.strftime("%Y-%m-%d"),
                            "type": t.trade_type,
                            "price": float(t.price),
                            "shares": float(t.shares),
                            "pnl": float(t.pnl) if t.pnl else 0,
                        }
                        for t in result.trades[:50]
                    ],
                })
            
            self.progress = 100
            yield
            
            return rx.redirect("/backtest")
            
        except Exception as e:
            self.error_message = f"回测失败: {str(e)}"
        finally:
            self.is_loading = False
            self.loading_message = ""
            yield
    
    # AI Report
    async def generate_ai_report(self):
        """Generate AI analysis report."""
        if not self.glm_api_key:
            self.error_message = "请先在设置中配置 GLM API Key"
            yield
            return
        
        if not self.backtest_results:
            self.error_message = "请先执行回测"
            yield
            return
        
        self.ai_generating = True
        self.ai_report = ""
        yield
        
        try:
            ai_service = AIService(self.glm_api_key)
            self.ai_report = await ai_service.generate_analysis_report(
                self.selected_stock,
                self.selected_stock_name,
                self.backtest_results,
            )
        except Exception as e:
            self.error_message = f"AI 报告生成失败: {str(e)}"
            self.ai_report = ""
        finally:
            self.ai_generating = False
            yield
    
    # Settings
    def set_glm_api_key(self, key: str):
        self.glm_api_key = key
    
    def set_initial_capital(self, value: str):
        try:
            self.initial_capital = float(value)
        except ValueError:
            pass
    
    def set_commission_rate(self, value: str):
        try:
            self.commission_rate = float(value)
        except ValueError:
            pass
    
    def set_position_size(self, value: str):
        try:
            self.position_size = float(value)
        except ValueError:
            pass
    
    def save_settings(self):
        """Save settings to config and environment."""
        import os
        os.environ["GLM_API_KEY"] = self.glm_api_key
        config.glm_api_key = self.glm_api_key
        config.initial_capital = self.initial_capital
        config.commission_rate = self.commission_rate
        config.position_size = self.position_size
    
    def clear_error(self):
        """Clear error message."""
        self.error_message = ""
```

**Step 2: Commit**

```bash
git add pixiu/state.py
git commit -m "feat: wire state management to real backend services"
```

---

## Phase 3: Redesign Pages

### Task 3.1: Redesign Home Page

**Files:**
- Modify: `pixiu/pages/home.py` (complete rewrite)

**Step 1: Create new home page with 4-step flow**

```python
# pixiu/pages/home.py
"""Home page - 4-step quantitative analysis flow."""

import reflex as rx
from pixiu.state import State
from pixiu.components import stock_card, strategy_card


def page() -> rx.Component:
    """Home page with clear 4-step flow."""
    return rx.box(
        rx.vstack(
            # Header
            _header(),
            
            # Step 1: Select Market & Search
            _step_1_search(),
            
            # Step 2: Select Stock
            rx.cond(
                State.search_results.length() > 0,
                _step_2_select_stock(),
            ),
            
            # Step 3: Select Strategy
            rx.cond(
                State.selected_stock != "",
                _step_3_select_strategy(),
            ),
            
            # Step 4: Configure & Run
            rx.cond(
                State.selected_strategies.length() > 0,
                _step_4_run(),
            ),
            
            # Error Display
            rx.cond(
                State.error_message != "",
                rx.box(
                    rx.hstack(
                        rx.text("⚠", font_size="1.25rem"),
                        rx.text(State.error_message, color="#ef4444"),
                        rx.spacer(),
                        rx.icon_button(
                            "X",
                            on_click=State.clear_error,
                            variant="ghost",
                            size="sm",
                        ),
                    ),
                    bg="#1a1a24",
                    padding="1rem",
                    border_radius="0.5rem",
                    border="1px solid #ef4444",
                    margin_top="1rem",
                ),
            ),
            
            # Loading Overlay
            rx.cond(
                State.is_loading,
                rx.box(
                    rx.vstack(
                        rx.spinner(size="lg", color="#00d9ff"),
                        rx.text(
                            State.loading_message,
                            color="#a0a0b0",
                        ),
                        rx.progress(
                            value=State.progress,
                            width="200px",
                            color_scheme="cyan",
                        ),
                        spacing="1rem",
                        align_items="center",
                    ),
                    position="fixed",
                    top="50%",
                    left="50%",
                    transform="translate(-50%, -50%)",
                    bg="#1a1a24",
                    padding="2rem",
                    border_radius="1rem",
                    border="1px solid #2a2a3a",
                    z_index=1000,
                ),
            ),
            
            spacing="1.5rem",
            width="100%",
            max_width="900px",
            margin="0 auto",
            padding="2rem",
        ),
        min_height="100vh",
        bg="#0a0a0f",
    )


def _header() -> rx.Component:
    """Page header with title and market selector."""
    return rx.hstack(
        rx.vstack(
            rx.text(
                "PIXIU",
                font_size="2rem",
                font_weight="bold",
                color="#00d9ff",
                letter_spacing="0.1em",
            ),
            rx.text(
                "量化分析平台",
                font_size="0.875rem",
                color="#a0a0b0",
            ),
            spacing="0",
        ),
        rx.spacer(),
        rx.hstack(
            _market_button("A股", State.current_market == "A股", State.set_market_a),
            _market_button("港股", State.current_market == "港股", State.set_market_hk),
            _market_button("美股", State.current_market == "美股", State.set_market_us),
            spacing="0.5rem",
        ),
        width="100%",
        margin_bottom="1rem",
    )


def _market_button(label: str, is_active: bool, on_click) -> rx.Component:
    """Market selection button."""
    return rx.button(
        label,
        on_click=on_click,
        variant="solid" if is_active else "outline",
        color_scheme="cyan",
        size="sm",
    )


def _step_1_search() -> rx.Component:
    """Step 1: Market and search."""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text(
                    "STEP 1",
                    font_size="0.75rem",
                    font_weight="bold",
                    color="#00d9ff",
                    letter_spacing="0.1em",
                ),
                rx.text(
                    "搜索股票",
                    font_size="1rem",
                    font_weight="bold",
                    color="#ffffff",
                ),
                spacing="0.5rem",
            ),
            rx.hstack(
                rx.input(
                    placeholder="输入股票代码或名称，如 000001 或 平安",
                    value=State.search_keyword,
                    on_change=State.set_search_keyword,
                    bg="#1a1a24",
                    border="1px solid #2a2a3a",
                    width="100%",
                    on_key_down=lambda e: State.search_stocks if e.key == "Enter" else None,
                ),
                rx.button(
                    "搜索",
                    on_click=State.search_stocks,
                    color_scheme="cyan",
                    is_loading=State.is_loading,
                ),
                width="100%",
            ),
            spacing="0.75rem",
            align_items="flex-start",
        ),
        bg="#12121a",
        padding="1.5rem",
        border_radius="0.5rem",
        border="1px solid #2a2a3a",
        width="100%",
    )


def _step_2_select_stock() -> rx.Component:
    """Step 2: Select stock from results."""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text(
                    "STEP 2",
                    font_size="0.75rem",
                    font_weight="bold",
                    color="#00d9ff",
                    letter_spacing="0.1em",
                ),
                rx.text(
                    "选择股票",
                    font_size="1rem",
                    font_weight="bold",
                    color="#ffffff",
                ),
                rx.spacer(),
                rx.cond(
                    State.selected_stock != "",
                    rx.hstack(
                        rx.text(
                            "已选择:",
                            font_size="0.875rem",
                            color="#a0a0b0",
                        ),
                        rx.text(
                            f"{State.selected_stock} {State.selected_stock_name}",
                            font_size="0.875rem",
                            font_weight="bold",
                            color="#10b981",
                        ),
                    ),
                ),
                spacing="0.5rem",
                width="100%",
            ),
            rx.box(
                rx.foreach(
                    State.search_results,
                    lambda stock: stock_card(
                        stock,
                        on_click_handler=lambda: State.select_stock(stock["code"]),
                    ),
                ),
                max_height="200px",
                overflow_y="auto",
                width="100%",
            ),
            spacing="0.75rem",
            align_items="flex-start",
        ),
        bg="#12121a",
        padding="1.5rem",
        border_radius="0.5rem",
        border="1px solid #2a2a3a",
        width="100%",
    )


def _step_3_select_strategy() -> rx.Component:
    """Step 3: Select strategies."""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text(
                    "STEP 3",
                    font_size="0.75rem",
                    font_weight="bold",
                    color="#00d9ff",
                    letter_spacing="0.1em",
                ),
                rx.text(
                    "选择策略",
                    font_size="1rem",
                    font_weight="bold",
                    color="#ffffff",
                ),
                rx.spacer(),
                rx.text(
                    f"已选择 {State.selected_strategies.length()} 个",
                    font_size="0.875rem",
                    color="#a0a0b0",
                ),
                spacing="0.5rem",
                width="100%",
            ),
            rx.grid(
                rx.foreach(
                    State.available_strategies,
                    lambda s: strategy_card(
                        s,
                        State.selected_strategies.contains(s["name"]),
                        on_click_handler=lambda: State.toggle_strategy(s["name"]),
                    ),
                ),
                columns="3",
                spacing="0.75rem",
            ),
            spacing="0.75rem",
            align_items="flex-start",
        ),
        bg="#12121a",
        padding="1.5rem",
        border_radius="0.5rem",
        border="1px solid #2a2a3a",
        width="100%",
    )


def _step_4_run() -> rx.Component:
    """Step 4: Configure and run backtest."""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text(
                    "STEP 4",
                    font_size="0.75rem",
                    font_weight="bold",
                    color="#00d9ff",
                    letter_spacing="0.1em",
                ),
                rx.text(
                    "配置并执行",
                    font_size="1rem",
                    font_weight="bold",
                    color="#ffffff",
                ),
                rx.link(
                    "高级设置 →",
                    href="/settings",
                    font_size="0.875rem",
                    color="#00d9ff",
                ),
                spacing="0.5rem",
                width="100%",
            ),
            rx.hstack(
                rx.vstack(
                    rx.text("初始资金", font_size="0.75rem", color="#a0a0b0"),
                    rx.text(f"¥{State.initial_capital:,.0f}", font_size="1rem", color="#ffffff"),
                    spacing="0.25rem",
                ),
                rx.vstack(
                    rx.text("手续费率", font_size="0.75rem", color="#a0a0b0"),
                    rx.text(f"{State.commission_rate*100:.2f}%", font_size="1rem", color="#ffffff"),
                    spacing="0.25rem",
                ),
                rx.vstack(
                    rx.text("仓位比例", font_size="0.75rem", color="#a0a0b0"),
                    rx.text(f"{State.position_size*100:.0f}%", font_size="1rem", color="#ffffff"),
                    spacing="0.25rem",
                ),
                rx.spacer(),
                rx.button(
                    "开始回测分析",
                    on_click=State.run_backtest,
                    color_scheme="cyan",
                    size="lg",
                ),
                width="100%",
            ),
            spacing="0.75rem",
            align_items="flex-start",
        ),
        bg="#12121a",
        padding="1.5rem",
        border_radius="0.5rem",
        border="1px solid #2a2a3a",
        width="100%",
    )
```

**Step 2: Commit**

```bash
git add pixiu/pages/home.py
git commit -m "feat: redesign home page with 4-step flow and dark theme"
```

---

### Task 3.2: Redesign Backtest Results Page

**Files:**
- Modify: `pixiu/pages/backtest.py` (complete rewrite)

**Step 1: Create new backtest results page**

```python
# pixiu/pages/backtest.py
"""Backtest results page with metrics and AI report."""

import reflex as rx
from pixiu.state import State
from pixiu.components import metric_card


def page() -> rx.Component:
    """Backtest results display."""
    return rx.box(
        rx.vstack(
            # Header
            rx.hstack(
                rx.button(
                    "← 返回",
                    on_click=rx.redirect("/"),
                    variant="ghost",
                    color_scheme="cyan",
                ),
                rx.spacer(),
                rx.text(
                    f"{State.selected_stock} {State.selected_stock_name}",
                    font_size="1.5rem",
                    font_weight="bold",
                    color="#ffffff",
                ),
                rx.spacer(),
                rx.badge("回测报告", color_scheme="cyan"),
                width="100%",
            ),
            
            rx.divider(border_color="#2a2a3a"),
            
            # No results message
            rx.cond(
                State.backtest_results.length() == 0,
                rx.box(
                    rx.text(
                        "暂无回测结果，请先执行回测分析",
                        color="#a0a0b0",
                    ),
                    padding="4rem",
                    text_align="center",
                ),
                
                # Results display
                rx.vstack(
                    # Metrics for each strategy
                    rx.foreach(
                        State.backtest_results,
                        _strategy_result,
                    ),
                    
                    rx.divider(border_color="#2a2a3a"),
                    
                    # AI Report Section
                    _ai_report_section(),
                    
                    spacing="1.5rem",
                    width="100%",
                ),
            ),
            
            spacing="1.5rem",
            width="100%",
            max_width="1200px",
            margin="0 auto",
            padding="2rem",
        ),
        min_height="100vh",
        bg="#0a0a0f",
    )


def _strategy_result(result: dict) -> rx.Component:
    """Display results for a single strategy."""
    total_return = result.get("total_return", 0)
    sharpe = result.get("sharpe_ratio", 0)
    
    return rx.box(
        rx.vstack(
            rx.text(
                result.get("strategy", "Unknown"),
                font_size="1.25rem",
                font_weight="bold",
                color="#ffffff",
            ),
            
            # Metrics Grid
            rx.grid(
                metric_card(
                    "总收益率",
                    f"{total_return*100:+.2f}%",
                    color="green" if total_return > 0 else "red",
                ),
                metric_card(
                    "年化收益",
                    f"{result.get('annualized_return', 0)*100:+.2f}%",
                    color="green" if result.get('annualized_return', 0) > 0 else "red",
                ),
                metric_card(
                    "最大回撤",
                    f"{result.get('max_drawdown', 0)*100:.2f}%",
                    color="red",
                ),
                metric_card(
                    "夏普比率",
                    f"{sharpe:.2f}",
                    color="green" if sharpe > 1 else "orange" if sharpe > 0 else "red",
                ),
                metric_card(
                    "胜率",
                    f"{result.get('win_rate', 0)*100:.1f}%",
                    color="green" if result.get('win_rate', 0) > 0.5 else "red",
                ),
                metric_card(
                    "盈亏比",
                    f"{result.get('profit_loss_ratio', 0):.2f}",
                    color="green" if result.get('profit_loss_ratio', 0) > 1 else "red",
                ),
                columns="6",
                spacing="0.75rem",
            ),
            
            # Trade History
            rx.box(
                rx.text(
                    "交易记录",
                    font_size="0.875rem",
                    font_weight="bold",
                    color="#a0a0b0",
                    margin_bottom="0.5rem",
                ),
                rx.box(
                    rx.foreach(
                        result.get("trades", []),
                        _trade_row,
                    ),
                    max_height="200px",
                    overflow_y="auto",
                ),
                width="100%",
            ),
            
            spacing="1rem",
            align_items="flex-start",
        ),
        bg="#12121a",
        padding="1.5rem",
        border_radius="0.5rem",
        border="1px solid #2a2a3a",
        width="100%",
    )


def _trade_row(trade: dict) -> rx.Component:
    """Display a single trade."""
    is_buy = trade.get("type") == "buy"
    pnl = trade.get("pnl", 0)
    
    return rx.hstack(
        rx.text(trade.get("date", ""), color="#a0a0b0", min_width="80px"),
        rx.badge(
            "买入" if is_buy else "卖出",
            color_scheme="green" if is_buy else "red",
            variant="subtle",
        ),
        rx.text(f"¥{trade.get('price', 0):.2f}", color="#ffffff"),
        rx.text(f"{trade.get('shares', 0):.0f}股", color="#a0a0b0"),
        rx.spacer(),
        rx.cond(
            pnl != 0,
            rx.text(
                f"{pnl:+.2f}",
                color="#10b981" if pnl > 0 else "#ef4444",
            ),
        ),
        width="100%",
        padding="0.5rem 0",
        border_bottom="1px solid #2a2a3a",
    )


def _ai_report_section() -> rx.Component:
    """AI analysis report section."""
    return rx.box(
        rx.hstack(
            rx.text(
                "AI 分析报告",
                font_size="1.25rem",
                font_weight="bold",
                color="#ffffff",
            ),
            rx.spacer(),
            rx.cond(
                State.ai_report == "",
                rx.button(
                    "生成报告" if not State.ai_generating else "生成中...",
                    on_click=State.generate_ai_report,
                    color_scheme="purple",
                    is_loading=State.ai_generating,
                ),
            ),
            width="100%",
        ),
        
        rx.cond(
            State.ai_report != "",
            rx.box(
                rx.text(
                    State.ai_report,
                    color="#a0a0b0",
                    white_space="pre-wrap",
                    line_height="1.75",
                ),
                bg="#1a1a24",
                padding="1.5rem",
                border_radius="0.5rem",
                margin_top="1rem",
            ),
            rx.cond(
                State.ai_generating,
                rx.box(
                    rx.hstack(
                        rx.spinner(color="#7c3aed"),
                        rx.text("正在生成 AI 分析报告...", color="#a0a0b0"),
                    ),
                    padding="2rem",
                    justify_content="center",
                ),
                rx.box(
                    rx.text(
                        "点击上方按钮生成 AI 智能分析报告",
                        color="#6b7280",
                    ),
                    padding="2rem",
                    text_align="center",
                ),
            ),
        ),
        
        width="100%",
    )
```

**Step 2: Commit**

```bash
git add pixiu/pages/backtest.py
git commit -m "feat: redesign backtest page with metrics display and AI report"
```

---

### Task 3.3: Redesign Settings Page

**Files:**
- Modify: `pixiu/pages/settings.py` (complete rewrite)

**Step 1: Create new settings page**

```python
# pixiu/pages/settings.py
"""Settings page - clean form layout."""

import reflex as rx
from pixiu.state import State


def page() -> rx.Component:
    """Settings page."""
    return rx.box(
        rx.vstack(
            # Header
            rx.hstack(
                rx.button(
                    "← 返回",
                    on_click=rx.redirect("/"),
                    variant="ghost",
                    color_scheme="cyan",
                ),
                rx.spacer(),
                rx.text(
                    "系统设置",
                    font_size="1.5rem",
                    font_weight="bold",
                    color="#ffffff",
                ),
                rx.spacer(),
                width="100%",
            ),
            
            rx.divider(border_color="#2a2a3a"),
            
            # API Configuration
            _section(
                "🔑 API 配置",
                rx.vstack(
                    rx.text("GLM API Key", color="#a0a0b0", font_size="0.875rem"),
                    rx.input(
                        placeholder="输入 GLM API Key",
                        value=State.glm_api_key,
                        on_change=State.set_glm_api_key,
                        type="password",
                        bg="#1a1a24",
                        border="1px solid #2a2a3a",
                        width="100%",
                    ),
                    rx.text(
                        "用于生成 AI 智能分析报告。可在 zhipuai.cn 获取",
                        font_size="0.75rem",
                        color="#6b7280",
                    ),
                    spacing="0.5rem",
                    width="100%",
                ),
            ),
            
            # Backtest Parameters
            _section(
                "💰 回测参数",
                rx.grid(
                    _input_field(
                        "初始资金",
                        State.initial_capital,
                        State.set_initial_capital,
                        "100000",
                    ),
                    _input_field(
                        "手续费率",
                        State.commission_rate,
                        State.set_commission_rate,
                        "0.0003",
                    ),
                    _input_field(
                        "仓位比例",
                        State.position_size,
                        State.set_position_size,
                        "0.95",
                    ),
                    columns="3",
                    spacing="1rem",
                ),
            ),
            
            # Save Button
            rx.button(
                "保存设置",
                on_click=State.save_settings,
                color_scheme="cyan",
                size="lg",
                margin_top="1rem",
            ),
            
            rx.spacer(),
            
            spacing="1.5rem",
            width="100%",
            max_width="600px",
            margin="0 auto",
            padding="2rem",
        ),
        min_height="100vh",
        bg="#0a0a0f",
    )


def _section(title: str, content: rx.Component) -> rx.Component:
    """Create a settings section."""
    return rx.box(
        rx.text(title, font_size="1rem", font_weight="bold", color="#ffffff"),
        rx.box(content, margin_top="1rem"),
        bg="#12121a",
        padding="1.5rem",
        border_radius="0.5rem",
        border="1px solid #2a2a3a",
        width="100%",
    )


def _input_field(
    label: str,
    value: float,
    on_change,
    placeholder: str,
) -> rx.Component:
    """Create an input field with label."""
    return rx.vstack(
        rx.text(label, color="#a0a0b0", font_size="0.875rem"),
        rx.input(
            value=str(value),
            on_change=on_change,
            placeholder=placeholder,
            bg="#1a1a24",
            border="1px solid #2a2a3a",
            width="100%",
        ),
        spacing="0.5rem",
        width="100%",
    )
```

**Step 2: Commit**

```bash
git add pixiu/pages/settings.py
git commit -m "feat: redesign settings page with clean dark layout"
```

---

## Phase 4: Integration & Testing

### Task 4.1: Update App Entry Point

**Files:**
- Modify: `pixiu/pixiu.py`

**Step 1: Update app with theme**

```python
# pixiu/pixiu.py
"""Pixiu quantitative analysis application."""

import reflex as rx
from pixiu.pages.home import page as home_page
from pixiu.pages.backtest import page as backtest_page
from pixiu.pages.settings import page as settings_page


app = rx.App(
    theme=rx.theme(
        appearance="dark",
        has_background=True,
        accent_color="cyan",
    )
)

app.add_page(home_page, route="/", title="Pixiu 量化分析")
app.add_page(backtest_page, route="/backtest", title="回测报告 - Pixiu")
app.add_page(settings_page, route="/settings", title="设置 - Pixiu")
```

**Step 2: Commit**

```bash
git add pixiu/pixiu.py
git commit -m "feat: update app entry with dark theme"
```

---

### Task 4.2: Test Full Flow

**Step 1: Stop any running instances**

```bash
pkill -f reflex || true
```

**Step 2: Start application**

```bash
reflex run
```

**Step 3: Verify each function**
- [ ] Home page loads with dark theme
- [ ] Market selection works (A股/港股/美股)
- [ ] Stock search returns real results from akshare
- [ ] Stock selection updates state
- [ ] Strategy cards display from registry
- [ ] Strategy selection toggles correctly
- [ ] Backtest execution runs without error
- [ ] Results page shows metrics
- [ ] AI report generation works (if API key set)
- [ ] Settings page saves configuration

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete Pixiu Phase 2 redesign with real data integration"
```

---

## Success Criteria

1. ✅ Search returns real stock data from akshare
2. ✅ Backtest runs with real BacktestEngine
3. ✅ AI report generates with real GLM-5
4. ✅ UI is dark, readable, and clear
5. ✅ 4-step flow is intuitive
6. ✅ No runtime errors
7. ✅ All pages render correctly

---

## Estimated Time

- Phase 1 (Foundation): 30 minutes
- Phase 2 (State): 30 minutes
- Phase 3 (Pages): 45 minutes
- Phase 4 (Testing): 15 minutes

**Total: ~2 hours**
