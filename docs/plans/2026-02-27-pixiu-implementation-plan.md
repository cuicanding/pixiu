# Pixiu 量化分析软件实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 构建一个支持A股/港股/美股的量化分析桌面软件，提供策略实验、回测分析和AI智能解读功能。

**Architecture:** 基于Reflex纯Python全栈框架，采用分层架构（UI层、服务层、策略层、数据层）。策略采用插件式设计，通过注册机制动态加载。所有耗时操作使用异步处理避免UI阻塞。

**Tech Stack:** Reflex, SQLite, Pandas, NumPy, SciPy, Plotly, akshare, GLM-5 API

---

## Phase 1: 项目初始化与框架搭建

### Task 1.1: 创建Reflex项目结构

**Files:**
- Create: `pixiu/__init__.py`
- Create: `pixiu/pixiu.py`
- Create: `pixiu/config.py`
- Create: `requirements.txt`
- Create: `rxconfig.py`

**Step 1: 创建项目目录**

```bash
mkdir -p pixiu/pages pixiu/components pixiu/services pixiu/strategies pixiu/models pixiu/utils data/cache assets
```

**Step 2: 创建 requirements.txt**

```txt
reflex>=0.4.0
akshare>=1.12.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
plotly>=5.18.0
zhipuai>=2.0.0
sqlalchemy>=2.0.0
aiosqlite>=0.19.0
python-dateutil>=2.8.0
```

**Step 3: 创建 rxconfig.py**

```python
import reflex as rx

config = rx.Config(
    app_name="pixiu",
    title="Pixiu 量化分析实验室",
    description="A股/港股/美股量化策略分析与回测平台",
)
```

**Step 4: 创建 pixiu/__init__.py**

```python
"""Pixiu - 量化分析实验室"""
__version__ = "0.1.0"
```

**Step 5: 创建 pixiu/config.py**

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    APP_NAME: str = "Pixiu"
    APP_VERSION: str = "0.1.0"
    
    DATA_DIR: Path = Path(__file__).parent.parent / "data"
    DB_PATH: Path = DATA_DIR / "stocks.db"
    CACHE_DIR: Path = DATA_DIR / "cache"
    
    GLM_MODEL: str = "glm-5"
    
    MARKETS: list[str] = ["A股", "港股", "美股"]
    
    DEFAULT_BACKTEST_CAPITAL: float = 100000.0
    DEFAULT_COMMISSION_RATE: float = 0.0003
    DEFAULT_SLIPPAGE_RATE: float = 0.0001

config = Config()

def ensure_directories():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
```

**Step 6: 创建 pixiu/pixiu.py (应用入口)**

```python
import reflex as rx
from pixiu.config import config, ensure_directories

ensure_directories()

app = rx.App()
```

**Step 7: 验证项目结构**

Run: `ls -la pixiu/`
Expected: 看到 __init__.py, pixiu.py, config.py 等文件

**Step 8: 提交**

```bash
git add .
git commit -m "feat: initialize Reflex project structure"
```

---

### Task 1.2: 设置SQLite数据库模型

**Files:**
- Create: `pixiu/models/__init__.py`
- Create: `pixiu/models/stock.py`
- Create: `pixiu/models/database.py`

**Step 1: 创建 models/__init__.py**

```python
from .stock import Stock, DailyQuote, StrategySignal, UpdateLog
from .database import init_database, get_session
```

**Step 2: 创建 models/database.py**

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from pixiu.config import config

engine = create_engine(f"sqlite:///{config.DB_PATH}", echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def init_database():
    from pixiu.models.stock import Stock, DailyQuote, StrategySignal, UpdateLog
    Base.metadata.create_all(bind=engine)

def get_session():
    return SessionLocal()
```

**Step 3: 创建 models/stock.py**

```python
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, date
from .database import Base

class Stock(Base):
    __tablename__ = "stocks"
    
    code = Column(String(20), primary_key=True)
    name = Column(String(50))
    market = Column(String(10))
    industry = Column(String(50))
    list_date = Column(Date, nullable=True)
    updated_at = Column(DateTime, default=datetime.now)
    
    quotes = relationship("DailyQuote", back_populates="stock", cascade="all, delete-orphan")

class DailyQuote(Base):
    __tablename__ = "daily_quotes"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(20), ForeignKey("stocks.code"))
    trade_date = Column(Date)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    amount = Column(Float)
    turnover_rate = Column(Float, nullable=True)
    
    stock = relationship("Stock", back_populates="quotes")
    
    __table_args__ = (
        {"unique_constraint": ("code", "trade_date")},
    )

class StrategySignal(Base):
    __tablename__ = "strategy_signals"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(20))
    strategy_name = Column(String(50))
    signal_date = Column(Date)
    signal_type = Column(String(10))
    confidence = Column(Float)
    price = Column(Float)
    metadata = Column(Text)

class UpdateLog(Base):
    __tablename__ = "update_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    market = Column(String(10))
    last_update = Column(DateTime)
    records_updated = Column(Integer)
```

**Step 4: 测试数据库初始化**

```python
# test_database.py
from pixiu.models.database import init_database, get_session
from pixiu.models.stock import Stock

init_database()
session = get_session()
print("Database initialized successfully")
session.close()
```

Run: `python test_database.py`
Expected: "Database initialized successfully"

**Step 5: 提交**

```bash
git add pixiu/models/
git commit -m "feat: add SQLite database models"
```

---

### Task 1.3: 实现数据获取服务

**Files:**
- Create: `pixiu/services/__init__.py`
- Create: `pixiu/services/data_service.py`

**Step 1: 创建 services/__init__.py**

```python
from .data_service import DataService
```

**Step 2: 创建 services/data_service.py (核心数据服务)**

```python
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy import select
from pixiu.models.database import get_session, init_database
from pixiu.models.stock import Stock, DailyQuote, UpdateLog
from pixiu.config import config

class DataService:
    MARKET_PREFIX = {
        "A股": {"sh": ".SH", "sz": ".SZ"},
        "港股": ".HK",
        "美股": ".US"
    }
    
    @staticmethod
    def search_stocks(keyword: str, market: str = "A股") -> list[dict]:
        """搜索股票"""
        try:
            if market == "A股":
                df = ak.stock_zh_a_spot_em()
                filtered = df[df['名称'].str.contains(keyword, na=False)]
                return filtered[['代码', '名称']].head(20).to_dict('records')
            elif market == "港股":
                df = ak.stock_hk_spot_em()
                filtered = df[df['名称'].str.contains(keyword, na=False)]
                return filtered[['代码', '名称']].head(20).to_dict('records')
            elif market == "美股":
                df = ak.stock_us_spot_em()
                filtered = df[df['名称'].str.contains(keyword, na=False, case=False)]
                return filtered[['代码', '名称']].head(20).to_dict('records')
        except Exception as e:
            print(f"Search error: {e}")
            return []
        return []
    
    @staticmethod
    async def fetch_stock_history(code: str, market: str, start_date: str = None) -> pd.DataFrame:
        """获取股票历史数据"""
        try:
            if market == "A股":
                if start_date is None:
                    start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y%m%d")
                df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, adjust="qfq")
                df = df.rename(columns={
                    '日期': 'trade_date', '开盘': 'open', '收盘': 'close',
                    '最高': 'high', '最低': 'low', '成交量': 'volume',
                    '成交额': 'amount', '换手率': 'turnover_rate'
                })
            elif market == "港股":
                if start_date is None:
                    start_date = (datetime.now() - timedelta(days=365*2)).strftime("%Y%m%d")
                df = ak.stock_hk_hist(symbol=code, period="daily", start_date=start_date, adjust="qfq")
                df = df.rename(columns={
                    '日期': 'trade_date', '开盘': 'open', '收盘': 'close',
                    '最高': 'high', '最低': 'low', '成交量': 'volume',
                    '成交额': 'amount'
                })
            elif market == "美股":
                if start_date is None:
                    start_date = (datetime.now() - timedelta(days=365*2)).strftime("%Y%m%d")
                df = ak.stock_us_hist(symbol=code, period="d", start_date=start_date)
                df = df.rename(columns={
                    '日期': 'trade_date', '开盘': 'open', '收盘': 'close',
                    '最高': 'high', '最低': 'low', '成交量': 'volume'
                })
            
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date')
            return df
        except Exception as e:
            print(f"Fetch history error: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def save_to_database(code: str, name: str, market: str, df: pd.DataFrame) -> int:
        """保存数据到数据库"""
        session = get_session()
        try:
            stock = session.query(Stock).filter_by(code=code).first()
            if not stock:
                stock = Stock(code=code, name=name, market=market)
                session.add(stock)
            
            records_added = 0
            for _, row in df.iterrows():
                existing = session.query(DailyQuote).filter_by(
                    code=code, trade_date=row['trade_date'].date()
                ).first()
                
                if not existing:
                    quote = DailyQuote(
                        code=code,
                        trade_date=row['trade_date'].date(),
                        open=row.get('open', 0),
                        high=row.get('high', 0),
                        low=row.get('low', 0),
                        close=row.get('close', 0),
                        volume=row.get('volume', 0),
                        amount=row.get('amount', 0),
                        turnover_rate=row.get('turnover_rate', 0)
                    )
                    session.add(quote)
                    records_added += 1
            
            session.commit()
            return records_added
        except Exception as e:
            session.rollback()
            print(f"Save to database error: {e}")
            return 0
        finally:
            session.close()
    
    @staticmethod
    def load_from_database(code: str) -> pd.DataFrame:
        """从数据库加载股票数据"""
        session = get_session()
        try:
            quotes = session.query(DailyQuote).filter_by(code=code).order_by(DailyQuote.trade_date).all()
            if not quotes:
                return pd.DataFrame()
            
            data = [{
                'trade_date': q.trade_date,
                'open': q.open,
                'high': q.high,
                'low': q.low,
                'close': q.close,
                'volume': q.volume,
                'amount': q.amount,
                'turnover_rate': q.turnover_rate
            } for q in quotes]
            
            return pd.DataFrame(data)
        finally:
            session.close()
    
    @staticmethod
    def get_last_update_date(code: str) -> Optional[datetime]:
        """获取最后更新日期"""
        session = get_session()
        try:
            quote = session.query(DailyQuote).filter_by(code=code).order_by(DailyQuote.trade_date.desc()).first()
            return quote.trade_date if quote else None
        finally:
            session.close()
```

**Step 3: 测试数据服务**

```python
# test_data_service.py
import asyncio
from pixiu.services.data_service import DataService

async def test():
    stocks = DataService.search_stocks("茅台", "A股")
    print(f"Found {len(stocks)} stocks")
    if stocks:
        print(stocks[0])

asyncio.run(test())
```

Run: `python test_data_service.py`
Expected: 显示找到的股票数量和第一个结果

**Step 4: 提交**

```bash
git add pixiu/services/
git commit -m "feat: add data service with akshare integration"
```

---

### Task 1.4: 创建基础UI框架和状态管理

**Files:**
- Create: `pixiu/state.py`
- Create: `pixiu/pages/__init__.py`
- Create: `pixiu/pages/home.py`
- Create: `pixiu/pages/analysis.py`
- Create: `pixiu/pages/backtest.py`
- Create: `pixiu/pages/settings.py`
- Create: `pixiu/components/__init__.py`
- Modify: `pixiu/pixiu.py`

**Step 1: 创建 state.py (全局状态)**

```python
import reflex as rx
import pandas as pd
from typing import Optional
from pixiu.services.data_service import DataService

class State(rx.State):
    is_loading: bool = False
    loading_message: str = ""
    progress: float = 0.0
    
    current_market: str = "A股"
    search_keyword: str = ""
    search_results: list[dict] = []
    
    current_stock_code: str = ""
    current_stock_name: str = ""
    stock_data: pd.DataFrame = pd.DataFrame()
    
    selected_strategies: list[str] = []
    backtest_result: dict = {}
    
    glm_api_key: str = ""
    ai_report: str = ""
    ai_generating: bool = False
    
    def set_market(self, market: str):
        self.current_market = market
        self.search_results = []
    
    def set_search_keyword(self, keyword: str):
        self.search_keyword = keyword
    
    def search_stocks(self):
        if not self.search_keyword:
            self.search_results = []
            return
        self.search_results = DataService.search_stocks(self.search_keyword, self.current_market)
    
    @rx.background
    async def select_stock(self, code: str, name: str):
        async with self:
            self.is_loading = True
            self.loading_message = "正在加载股票数据..."
            self.current_stock_code = code
            self.current_stock_name = name
        
        df = DataService.load_from_database(code)
        
        if df.empty:
            async with self:
                self.loading_message = "正在从网络获取数据..."
            df = await DataService.fetch_stock_history(code, self.current_market)
            if not df.empty:
                DataService.save_to_database(code, name, self.current_market, df)
        
        async with self:
            self.stock_data = df
            self.is_loading = False
            self.loading_message = ""
    
    def toggle_strategy(self, strategy_name: str):
        if strategy_name in self.selected_strategies:
            self.selected_strategies.remove(strategy_name)
        else:
            self.selected_strategies.append(strategy_name)
```

**Step 2: 创建 pages/__init__.py**

```python
from .home import home_page
from .analysis import analysis_page
from .backtest import backtest_page
from .settings import settings_page
```

**Step 3: 创建 pages/home.py**

```python
import reflex as rx
from pixiu.state import State
from pixiu.config import config

def stock_card(stock: dict) -> rx.Component:
    return rx.card(
        rx.hstack(
            rx.text(stock['代码'], font_weight="bold"),
            rx.text(stock['名称']),
            rx.button("选择", on_click=lambda: State.select_stock(stock['代码'], stock['名称'])),
            justify="between",
        ),
        cursor="pointer",
        on_click=lambda: State.select_stock(stock['代码'], stock['名称']),
    )

def home_page() -> rx.Component:
    return rx.vstack(
        rx.heading(f"📊 {config.APP_NAME} 量化分析实验室", size="lg"),
        
        rx.hstack(
            rx.select(
                config.MARKETS,
                value=State.current_market,
                on_change=State.set_market,
            ),
            rx.input(
                placeholder="搜索股票代码或名称...",
                value=State.search_keyword,
                on_change=State.set_search_keyword,
                on_key_down=lambda e: State.search_stocks() if e.key == "Enter" else None,
            ),
            rx.button("搜索", on_click=State.search_stocks),
        ),
        
        rx.text(f"当前股票: {State.current_stock_name} ({State.current_stock_code})")
            if State.current_stock_code else rx.text("请选择一只股票"),
        
        rx.box(
            rx.foreach(State.search_results, stock_card),
            max_height="400px",
            overflow_y="auto",
        ),
        
        rx.spinner() if State.is_loading else rx.fragment(),
        rx.text(State.loading_message) if State.is_loading else rx.fragment(),
        
        spacing="4",
        padding="4",
    )
```

**Step 4: 创建 pages/analysis.py**

```python
import reflex as rx
from pixiu.state import State

def analysis_page() -> rx.Component:
    return rx.vstack(
        rx.heading("📈 策略分析", size="lg"),
        
        rx.text(f"分析股票: {State.current_stock_name}"),
        
        rx.hstack(
            rx.badge("趋势强度", 
                color_scheme="green" if "趋势强度" in State.selected_strategies else "gray",
                on_click=lambda: State.toggle_strategy("趋势强度"),
                cursor="pointer",
            ),
            rx.badge("波动率套利",
                color_scheme="green" if "波动率套利" in State.selected_strategies else "gray",
                on_click=lambda: State.toggle_strategy("波动率套利"),
                cursor="pointer",
            ),
            rx.badge("卡尔曼滤波",
                color_scheme="green" if "卡尔曼滤波" in State.selected_strategies else "gray",
                on_click=lambda: State.toggle_strategy("卡尔曼滤波"),
                cursor="pointer",
            ),
        ),
        
        rx.button("开始分析", on_click=rx.redirect("/backtest")),
        
        spacing="4",
        padding="4",
    )
```

**Step 5: 创建 pages/backtest.py**

```python
import reflex as rx
from pixiu.state import State

def backtest_page() -> rx.Component:
    return rx.vstack(
        rx.heading("📋 回测结果", size="lg"),
        
        rx.hstack(
            rx.stat_group(
                rx.stat(
                    rx.stat_label("年化收益"),
                    rx.stat_number("+28.5%"),
                ),
                rx.stat(
                    rx.stat_label("最大回撤"),
                    rx.stat_number("-12.3%"),
                ),
            ),
        ),
        
        rx.box(
            rx.text("收益曲线图表区域"),
            min_height="300px",
            bg="gray.100",
            border_radius="md",
        ),
        
        rx.box(
            rx.heading("🤖 AI 分析报告", size="md"),
            rx.markdown(State.ai_report) if State.ai_report else rx.text("点击下方按钮生成AI报告"),
            rx.button("生成AI报告", on_click=lambda: None),
            padding="4",
            bg="gray.50",
            border_radius="md",
        ),
        
        spacing="4",
        padding="4",
    )
```

**Step 6: 创建 pages/settings.py**

```python
import reflex as rx
from pixiu.state import State

def settings_page() -> rx.Component:
    return rx.vstack(
        rx.heading("⚙️ 设置", size="lg"),
        
        rx.form(
            rx.vstack(
                rx.form_label("GLM API Key"),
                rx.input(
                    type="password",
                    value=State.glm_api_key,
                    on_change=State.set_glm_api_key,
                ),
                rx.button("保存设置"),
            ),
        ),
        
        rx.divider(),
        
        rx.vstack(
            rx.heading("数据管理", size="md"),
            rx.button("更新所有股票数据", on_click=lambda: None),
            rx.button("清除缓存", on_click=lambda: None),
        ),
        
        spacing="4",
        padding="4",
    )
```

**Step 7: 更新 pixiu/pixiu.py**

```python
import reflex as rx
from pixiu.config import config, ensure_directories
from pixiu.models.database import init_database
from pixiu.pages.home import home_page
from pixiu.pages.analysis import analysis_page
from pixiu.pages.backtest import backtest_page
from pixiu.pages.settings import settings_page
from pixiu.state import State

ensure_directories()
init_database()

app = rx.App()
app.add_page(home_page, route="/", title="首页")
app.add_page(analysis_page, route="/analysis", title="策略分析")
app.add_page(backtest_page, route="/backtest", title="回测结果")
app.add_page(settings_page, route="/settings", title="设置")
```

**Step 8: 创建 assets/styles.css**

```css
/* Pixiu 自定义样式 */
```

**Step 9: 测试应用启动**

Run: `reflex run`
Expected: 应用启动，浏览器打开显示首页

**Step 10: 提交**

```bash
git add pixiu/state.py pixiu/pages/ pixiu/components/ pixiu/pixiu.py assets/
git commit -m "feat: add UI framework with pages and state management"
```

---

## Phase 2: 策略层实现

### Task 2.1: 创建策略基类和注册机制

**Files:**
- Create: `pixiu/strategies/__init__.py`
- Create: `pixiu/strategies/base.py`

**Step 1: 创建 strategies/base.py**

```python
from abc import ABC, abstractmethod
import pandas as pd
from typing import Any

class BaseStrategy(ABC):
    name: str = ""
    description: str = ""
    params: dict[str, Any] = {}
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号
        
        Args:
            df: 行情数据，包含 trade_date, open, high, low, close, volume
            
        Returns:
            添加 signal 列的DataFrame
            signal: 1=买入, -1=卖出, 0=持有
        """
        pass
    
    @abstractmethod
    def get_required_columns(self) -> list[str]:
        """返回需要的数据列"""
        pass
    
    def get_params_schema(self) -> dict:
        """返回参数的schema用于UI渲染"""
        return {}
    
    def set_param(self, key: str, value: Any):
        """设置参数"""
        self.params[key] = value
    
    def get_documentation(self) -> str:
        """返回策略的数学原理说明"""
        return ""
```

**Step 2: 创建 strategies/__init__.py**

```python
from .base import BaseStrategy

STRATEGY_REGISTRY: dict[str, BaseStrategy] = {}

def register_strategy(cls):
    """装饰器：自动注册策略"""
    instance = cls()
    STRATEGY_REGISTRY[instance.name] = instance
    return cls

def get_all_strategies() -> list[BaseStrategy]:
    """获取所有已注册策略"""
    return list(STRATEGY_REGISTRY.values())

def get_strategy(name: str) -> BaseStrategy | None:
    """按名称获取策略"""
    return STRATEGY_REGISTRY.get(name)
```

**Step 3: 提交**

```bash
git add pixiu/strategies/
git commit -m "feat: add strategy base class and registry"
```

---

### Task 2.2: 实现趋势强度策略

**Files:**
- Create: `pixiu/strategies/trend_strength.py`
- Modify: `pixiu/strategies/__init__.py`

**Step 1: 创建 strategies/trend_strength.py**

```python
import numpy as np
import pandas as pd
from .base import BaseStrategy
from . import register_strategy

@register_strategy
class TrendStrengthStrategy(BaseStrategy):
    name = "趋势强度"
    description = "基于价格导数判断趋势强度，f'(t)>0表示上升趋势，f''(t)表示变化加速度"
    params = {
        "threshold": 0.02,
        "window": 20,
    }
    
    def get_required_columns(self) -> list[str]:
        return ["close"]
    
    def get_params_schema(self) -> dict:
        return {
            "threshold": {"type": "float", "min": 0.01, "max": 0.1, "default": 0.02, "label": "趋势强度阈值"},
            "window": {"type": "int", "min": 5, "max": 60, "default": 20, "label": "观察窗口(天)"},
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        threshold = self.params.get("threshold", 0.02)
        window = self.params.get("window", 20)
        
        df['price_derivative'] = np.gradient(df['close'].values)
        
        df['price_acceleration'] = np.gradient(df['price_derivative'].values)
        
        rolling_std = df['close'].rolling(window=window).std()
        df['trend_strength'] = df['price_derivative'] / rolling_std
        
        conditions = [
            (df['trend_strength'] > threshold) & (df['price_acceleration'] > 0),
            (df['trend_strength'] < -threshold) & (df['price_acceleration'] < 0),
        ]
        choices = [1, -1]
        
        df['signal'] = np.select(conditions, choices, default=0)
        df['signal_strength'] = np.abs(df['trend_strength'])
        
        return df
    
    def get_documentation(self) -> str:
        return """
## 趋势强度策略

### 数学原理
使用微积分中的导数概念分析价格趋势：

- **一阶导数 f'(t)**：价格变化率，正值表示上涨，负值表示下跌
- **二阶导数 f''(t)**：变化加速度，正值表示趋势加强，负值表示趋势减弱

### 信号生成逻辑
1. 当 `f'(t) > threshold` 且 `f''(t) > 0`：**买入信号**（强势上涨）
2. 当 `f'(t) < -threshold` 且 `f''(t) < 0`：**卖出信号**（强势下跌）
3. 其他情况：**持有**

### 参数说明
- **threshold**: 趋势强度阈值，越大则信号越少但更可靠
- **window**: 波动率计算窗口，用于标准化趋势强度
"""
```

**Step 2: 更新 strategies/__init__.py 添加导入**

```python
from .base import BaseStrategy
from .trend_strength import TrendStrengthStrategy

STRATEGY_REGISTRY: dict[str, BaseStrategy] = {}

def register_strategy(cls):
    instance = cls()
    STRATEGY_REGISTRY[instance.name] = instance
    return cls

def get_all_strategies() -> list[BaseStrategy]:
    return list(STRATEGY_REGISTRY.values())

def get_strategy(name: str) -> BaseStrategy | None:
    return STRATEGY_REGISTRY.get(name)
```

**Step 3: 测试策略**

```python
# test_strategy.py
import pandas as pd
import numpy as np
from pixiu.strategies import get_strategy, get_all_strategies

print("Available strategies:", [s.name for s in get_all_strategies()])

df = pd.DataFrame({
    'trade_date': pd.date_range('2023-01-01', periods=100),
    'close': np.cumsum(np.random.randn(100)) + 100,
})

strategy = get_strategy("趋势强度")
if strategy:
    result = strategy.generate_signals(df)
    print(result[['trade_date', 'close', 'signal', 'signal_strength']].tail(10))
```

Run: `python test_strategy.py`
Expected: 显示策略名称和生成的信号

**Step 4: 提交**

```bash
git add pixiu/strategies/
git commit -m "feat: add trend strength strategy with calculus-based signals"
```

---

### Task 2.3: 实现回测引擎

**Files:**
- Create: `pixiu/services/backtest_service.py`
- Create: `pixiu/models/backtest.py`
- Modify: `pixiu/services/__init__.py`

**Step 1: 创建 models/backtest.py**

```python
from dataclasses import dataclass, field
from typing import list
from datetime import date

@dataclass
class Trade:
    date: date
    type: str
    shares: float
    price: float
    commission: float

@dataclass
class BacktestResult:
    start_date: date
    end_date: date
    initial_capital: float
    final_capital: float
    
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_loss_ratio: float
    calmar_ratio: float
    
    total_trades: int
    winning_trades: int
    losing_trades: int
    
    trades: list[Trade] = field(default_factory=list)
    daily_values: list[float] = field(default_factory=list)
    drawdowns: list[float] = field(default_factory=list)

@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0
    commission_rate: float = 0.0003
    slippage_rate: float = 0.0001
    position_size: float = 0.95
    risk_free_rate: float = 0.03
```

**Step 2: 创建 services/backtest_service.py**

```python
import numpy as np
import pandas as pd
from datetime import date
from typing import Optional
from pixiu.models.backtest import BacktestConfig, BacktestResult, Trade

class BacktestEngine:
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
    
    def run(self, df: pd.DataFrame, signals: pd.Series) -> BacktestResult:
        """执行回测"""
        cash = self.config.initial_capital
        shares = 0.0
        portfolio_value = cash
        
        daily_values = []
        trades = []
        drawdowns = []
        peak_value = cash
        
        winning_trades = 0
        losing_trades = 0
        total_profit = 0.0
        total_loss = 0.0
        
        df = df.copy()
        df['signal'] = signals.reindex(df.index).fillna(0)
        
        for i, (idx, row) in enumerate(df.iterrows()):
            signal = row['signal']
            price = row['close']
            
            adjusted_price = price * (1 + self.config.slippage_rate * np.sign(signal))
            
            if signal == 1 and cash > 0 and shares == 0:
                position_value = cash * self.config.position_size
                shares_to_buy = position_value / adjusted_price
                commission = shares_to_buy * adjusted_price * self.config.commission_rate
                
                shares = shares_to_buy
                cash = cash - position_value - commission
                
                trades.append(Trade(
                    date=idx.date() if hasattr(idx, 'date') else idx,
                    type='BUY',
                    shares=shares,
                    price=adjusted_price,
                    commission=commission
                ))
            
            elif signal == -1 and shares > 0:
                commission = shares * adjusted_price * self.config.commission_rate
                sell_value = shares * adjusted_price - commission
                
                buy_trade = [t for t in trades if t.type == 'BUY'][-1] if trades else None
                if buy_trade:
                    profit = sell_value - (buy_trade.shares * buy_trade.price)
                    if profit > 0:
                        winning_trades += 1
                        total_profit += profit
                    else:
                        losing_trades += 1
                        total_loss += abs(profit)
                
                cash = cash + sell_value
                shares = 0
                
                trades.append(Trade(
                    date=idx.date() if hasattr(idx, 'date') else idx,
                    type='SELL',
                    shares=shares,
                    price=adjusted_price,
                    commission=commission
                ))
            
            portfolio_value = cash + shares * price
            daily_values.append(portfolio_value)
            
            if portfolio_value > peak_value:
                peak_value = portfolio_value
            drawdown = (peak_value - portfolio_value) / peak_value
            drawdowns.append(drawdown)
        
        final_value = cash + shares * df.iloc[-1]['close']
        
        total_trades = len([t for t in trades if t.type == 'SELL'])
        
        returns = pd.Series(daily_values).pct_change().dropna()
        annualized_return = (final_value / self.config.initial_capital) ** (252 / len(df)) - 1
        max_drawdown = max(drawdowns)
        
        sharpe_ratio = 0.0
        if len(returns) > 0 and returns.std() > 0:
            excess_return = annualized_return - self.config.risk_free_rate
            sharpe_ratio = excess_return / (returns.std() * np.sqrt(252))
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_loss_ratio = total_profit / total_loss if total_loss > 0 else 0
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        return BacktestResult(
            start_date=df.index[0].date() if hasattr(df.index[0], 'date') else df.index[0],
            end_date=df.index[-1].date() if hasattr(df.index[-1], 'date') else df.index[-1],
            initial_capital=self.config.initial_capital,
            final_capital=final_value,
            total_return=(final_value - self.config.initial_capital) / self.config.initial_capital,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            profit_loss_ratio=profit_loss_ratio,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            trades=trades,
            daily_values=daily_values,
            drawdowns=drawdowns
        )
```

**Step 3: 更新 services/__init__.py**

```python
from .data_service import DataService
from .backtest_service import BacktestEngine
```

**Step 4: 测试回测引擎**

```python
# test_backtest.py
import pandas as pd
import numpy as np
from pixiu.services.backtest_service import BacktestEngine
from pixiu.strategies import get_strategy

df = pd.DataFrame({
    'trade_date': pd.date_range('2023-01-01', periods=200),
    'open': 100 + np.cumsum(np.random.randn(200) * 0.5),
    'high': 101 + np.cumsum(np.random.randn(200) * 0.5),
    'low': 99 + np.cumsum(np.random.randn(200) * 0.5),
    'close': 100 + np.cumsum(np.random.randn(200) * 0.5),
    'volume': np.random.randint(1000000, 5000000, 200),
})
df.set_index('trade_date', inplace=True)

strategy = get_strategy("趋势强度")
result_df = strategy.generate_signals(df)

engine = BacktestEngine()
result = engine.run(df, result_df['signal'])

print(f"总收益: {result.total_return:.2%}")
print(f"年化收益: {result.annualized_return:.2%}")
print(f"最大回撤: {result.max_drawdown:.2%}")
print(f"夏普比率: {result.sharpe_ratio:.2f}")
```

Run: `python test_backtest.py`
Expected: 显示回测结果指标

**Step 5: 提交**

```bash
git add pixiu/services/backtest_service.py pixiu/models/backtest.py
git commit -m "feat: add backtest engine with performance metrics"
```

---

## Phase 3: UI集成与可视化

### Task 3.1: 添加Plotly图表组件

**Files:**
- Create: `pixiu/components/chart_panel.py`
- Create: `pixiu/utils/visualization.py`
- Modify: `pixiu/components/__init__.py`

**Step 1: 创建 utils/visualization.py**

```python
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Optional

def create_candlestick_chart(df: pd.DataFrame, signals: pd.Series = None) -> go.Figure:
    """创建K线图"""
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='K线',
    ))
    
    if signals is not None:
        buy_signals = df.index[signals == 1]
        sell_signals = df.index[signals == -1]
        
        if len(buy_signals) > 0:
            fig.add_trace(go.Scatter(
                x=buy_signals,
                y=df.loc[buy_signals, 'low'] * 0.99,
                mode='markers',
                marker=dict(symbol='triangle-up', size=15, color='red'),
                name='买入信号'
            ))
        
        if len(sell_signals) > 0:
            fig.add_trace(go.Scatter(
                x=sell_signals,
                y=df.loc[sell_signals, 'high'] * 1.01,
                mode='markers',
                marker=dict(symbol='triangle-down', size=15, color='green'),
                name='卖出信号'
            ))
    
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    
    return fig

def create_returns_chart(daily_values: list[float], benchmark: list[float] = None) -> go.Figure:
    """创建收益曲线图"""
    fig = go.Figure()
    
    returns = [(v / daily_values[0] - 1) * 100 for v in daily_values]
    
    fig.add_trace(go.Scatter(
        y=returns,
        mode='lines',
        name='策略收益',
        line=dict(color='blue', width=2),
    ))
    
    if benchmark:
        benchmark_returns = [(v / benchmark[0] - 1) * 100 for v in benchmark]
        fig.add_trace(go.Scatter(
            y=benchmark_returns,
            mode='lines',
            name='基准收益',
            line=dict(color='gray', width=1, dash='dash'),
        ))
    
    fig.update_layout(
        yaxis_title='收益率 (%)',
        height=300,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    
    return fig

def create_drawdown_chart(drawdowns: list[float]) -> go.Figure:
    """创建回撤图"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=[-d * 100 for d in drawdowns],
        mode='lines',
        fill='tozeroy',
        name='回撤',
        line=dict(color='red'),
    ))
    
    fig.update_layout(
        yaxis_title='回撤 (%)',
        height=200,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    
    return fig
```

**Step 2: 创建 components/chart_panel.py**

```python
import reflex as rx
from plotly.graph_objects import Figure
from typing import Optional

def chart_panel(figure: Figure, title: str = "") -> rx.Component:
    """图表面板组件"""
    return rx.box(
        rx.text(title, font_weight="bold", margin_bottom="0.5rem") if title else rx.fragment(),
        rx.plotly(data=figure.data, layout=figure.layout),
        border_radius="md",
        bg="white",
        shadow="sm",
        padding="1rem",
    )
```

**Step 3: 提交**

```bash
git add pixiu/components/chart_panel.py pixiu/utils/visualization.py
git commit -m "feat: add Plotly chart components for visualization"
```

---

### Task 3.2: 完善回测页面

**Files:**
- Modify: `pixiu/pages/backtest.py`
- Modify: `pixiu/state.py`

**Step 1: 更新 state.py 添加回测逻辑**

```python
# 在 State 类中添加

from pixiu.strategies import get_all_strategies, get_strategy
from pixiu.services.backtest_service import BacktestEngine
from pixiu.utils.visualization import create_candlestick_chart, create_returns_chart, create_drawdown_chart
import plotly.graph_objects as go

class State(rx.State):
    # ... 之前的状态变量 ...
    
    backtest_result: dict = {}
    candlestick_figure: go.Figure = None
    returns_figure: go.Figure = None
    drawdown_figure: go.Figure = None
    
    @rx.background
    async def run_backtest(self):
        async with self:
            self.is_loading = True
            self.loading_message = "正在执行回测..."
        
        if self.stock_data.empty or not self.selected_strategies:
            async with self:
                self.is_loading = False
            return
        
        strategy = get_strategy(self.selected_strategies[0])
        if not strategy:
            async with self:
                self.is_loading = False
            return
        
        df = self.stock_data.copy()
        df.set_index('trade_date', inplace=True)
        
        result_df = strategy.generate_signals(df)
        
        engine = BacktestEngine()
        result = engine.run(df, result_df['signal'])
        
        candlestick = create_candlestick_chart(df, result_df['signal'])
        returns = create_returns_chart(result.daily_values)
        drawdown = create_drawdown_chart(result.drawdowns)
        
        async with self:
            self.backtest_result = {
                'total_return': f"{result.total_return:.2%}",
                'annualized_return': f"{result.annualized_return:.2%}",
                'max_drawdown': f"{result.max_drawdown:.2%}",
                'sharpe_ratio': f"{result.sharpe_ratio:.2f}",
                'win_rate': f"{result.win_rate:.2%}",
                'profit_loss_ratio': f"{result.profit_loss_ratio:.2f}",
                'calmar_ratio': f"{result.calmar_ratio:.2f}",
                'total_trades': result.total_trades,
            }
            self.candlestick_figure = candlestick
            self.returns_figure = returns
            self.drawdown_figure = drawdown
            self.is_loading = False
            self.loading_message = ""
```

**Step 2: 更新 pages/backtest.py**

```python
import reflex as rx
from pixiu.state import State

def metric_card(label: str, value: str, color: str = "black") -> rx.Component:
    return rx.box(
        rx.text(label, font_size="sm", color="gray.500"),
        rx.text(value, font_size="xl", font_weight="bold", color=color),
        padding="1rem",
        bg="gray.50",
        border_radius="md",
    )

def backtest_page() -> rx.Component:
    return rx.vstack(
        rx.heading(f"📋 回测报告 - {State.current_stock_name}", size="lg"),
        
        rx.hstack(
            metric_card("年化收益", State.backtest_result.get('annualized_return', '--'), 
                       "green" if "+" in State.backtest_result.get('annualized_return', '') else "red"),
            metric_card("最大回撤", State.backtest_result.get('max_drawdown', '--'), "red"),
            metric_card("夏普比率", State.backtest_result.get('sharpe_ratio', '--'), "blue"),
            metric_card("胜率", State.backtest_result.get('win_rate', '--'), "purple"),
            spacing="4",
        ),
        
        rx.tabs(
            rx.tab_list(
                rx.tab("K线图"),
                rx.tab("收益曲线"),
                rx.tab("回撤分析"),
            ),
            rx.tab_panels(
                rx.tab_panel(
                    rx.plotly(
                        data=State.candlestick_figure.data if State.candlestick_figure else [],
                        layout=State.candlestick_figure.layout if State.candlestick_figure else {},
                    )
                ),
                rx.tab_panel(
                    rx.plotly(
                        data=State.returns_figure.data if State.returns_figure else [],
                        layout=State.returns_figure.layout if State.returns_figure else {},
                    )
                ),
                rx.tab_panel(
                    rx.plotly(
                        data=State.drawdown_figure.data if State.drawdown_figure else [],
                        layout=State.drawdown_figure.layout if State.drawdown_figure else {},
                    )
                ),
            ),
        ),
        
        rx.box(
            rx.heading("🤖 AI 智能分析", size="md"),
            rx.cond(
                State.ai_generating,
                rx.hstack(rx.spinner(), rx.text("AI正在分析中...")),
                rx.markdown(State.ai_report) if State.ai_report else rx.text("点击下方按钮生成AI报告", color="gray"),
            ),
            rx.button(
                "生成AI报告",
                on_click=State.generate_ai_report,
                margin_top="1rem",
            ),
            padding="1rem",
            bg="gray.50",
            border_radius="md",
            border_left="4px solid",
            border_color="blue.500",
        ),
        
        rx.hstack(
            rx.button("重新分析", on_click=rx.redirect("/analysis")),
            rx.button("返回首页", on_click=rx.redirect("/")),
        ),
        
        spacing="4",
        padding="4",
    )
```

**Step 3: 提交**

```bash
git add pixiu/state.py pixiu/pages/backtest.py
git commit -m "feat: complete backtest page with charts and metrics"
```

---

## Phase 4: GLM AI集成

### Task 4.1: 实现AI分析服务

**Files:**
- Create: `pixiu/services/ai_service.py`
- Modify: `pixiu/services/__init__.py`
- Modify: `pixiu/state.py`

**Step 1: 创建 services/ai_service.py**

```python
from zhipuai import ZhipuAI
from typing import Optional
import os

class AIReportService:
    _instance: Optional['AIReportService'] = None
    _client: Optional[ZhipuAI] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def configure(self, api_key: str):
        self._client = ZhipuAI(api_key=api_key)
    
    @property
    def is_configured(self) -> bool:
        return self._client is not None
    
    async def generate_analysis(
        self,
        backtest_result: dict,
        stock_name: str,
        strategy_name: str,
    ) -> str:
        if not self.is_configured:
            return "错误：请先在设置页面配置GLM API Key"
        
        system_prompt = """你是一位专业的量化投资分析师。
你的任务是将回测数据转化为通俗易懂的投资建议。
请用中文回答，结构清晰，包含：
1. 📊 策略表现总结
2. ✅ 策略优势分析  
3. ⚠️ 风险提示
4. 💡 优化建议
5. 🎯 适用场景

语气专业但亲切，避免过于技术性的表述。"""
        
        user_prompt = f"""请分析以下量化策略的回测结果：

**股票**: {stock_name}
**策略**: {strategy_name}

**回测指标**:
- 年化收益率: {backtest_result.get('annualized_return', '--')}
- 最大回撤: {backtest_result.get('max_drawdown', '--')}
- 夏普比率: {backtest_result.get('sharpe_ratio', '--')}
- 胜率: {backtest_result.get('win_rate', '--')}
- 盈亏比: {backtest_result.get('profit_loss_ratio', '--')}
- 卡玛比率: {backtest_result.get('calmar_ratio', '--')}
- 总交易次数: {backtest_result.get('total_trades', '--')}

请给出专业的分析报告。"""
        
        try:
            response = self._client.chat.completions.create(
                model="glm-5",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"生成AI报告时出错: {str(e)}"

ai_service = AIReportService()
```

**Step 2: 更新 services/__init__.py**

```python
from .data_service import DataService
from .backtest_service import BacktestEngine
from .ai_service import ai_service
```

**Step 3: 更新 state.py 添加AI生成方法**

```python
# 在 State 类中添加

from pixiu.services.ai_service import ai_service

class State(rx.State):
    # ... 之前的状态 ...
    
    def set_glm_api_key(self, key: str):
        self.glm_api_key = key
        ai_service.configure(key)
    
    @rx.background
    async def generate_ai_report(self):
        if not ai_service.is_configured:
            async with self:
                self.ai_report = "请先在设置页面配置GLM API Key"
            return
        
        async with self:
            self.ai_generating = True
        
        report = await ai_service.generate_analysis(
            self.backtest_result,
            self.current_stock_name,
            self.selected_strategies[0] if self.selected_strategies else "未知策略"
        )
        
        async with self:
            self.ai_report = report
            self.ai_generating = False
```

**Step 4: 测试AI服务**

```python
# test_ai_service.py
import asyncio
from pixiu.services.ai_service import ai_service

async def test():
    ai_service.configure("your_api_key_here")
    
    result = {
        'annualized_return': '+28.5%',
        'max_drawdown': '-12.3%',
        'sharpe_ratio': '1.85',
        'win_rate': '62.5%',
        'profit_loss_ratio': '2.1',
        'calmar_ratio': '2.3',
        'total_trades': '48'
    }
    
    report = await ai_service.generate_analysis(result, "贵州茅台", "趋势强度")
    print(report)

asyncio.run(test())
```

**Step 5: 提交**

```bash
git add pixiu/services/ai_service.py pixiu/state.py
git commit -m "feat: add GLM-5 AI analysis service integration"
```

---

## Phase 5: 完善与测试

### Task 5.1: 添加更多策略

**Files:**
- Create: `pixiu/strategies/volatility.py`
- Create: `pixiu/strategies/kalman_filter.py`

**Step 1: 创建 strategies/volatility.py**

```python
import numpy as np
import pandas as pd
from .base import BaseStrategy
from . import register_strategy

@register_strategy
class VolatilityStrategy(BaseStrategy):
    name = "波动率套利"
    description = "基于波动率积分判断超买超卖区域，利用均值回归原理"
    params = {
        "window": 20,
        "entry_threshold": 2.0,
        "exit_threshold": 0.5,
    }
    
    def get_required_columns(self) -> list[str]:
        return ["close", "high", "low"]
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        window = self.params.get("window", 20)
        entry_threshold = self.params.get("entry_threshold", 2.0)
        exit_threshold = self.params.get("exit_threshold", 0.5)
        
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window).std() * np.sqrt(252)
        
        df['vol_ma'] = df['volatility'].rolling(window).mean()
        df['vol_std'] = df['volatility'].rolling(window).std()
        df['vol_zscore'] = (df['volatility'] - df['vol_ma']) / df['vol_std']
        
        true_range = df['high'] - df['low']
        df['atr'] = true_range.rolling(window).mean()
        df['price_zscore'] = (df['close'] - df['close'].rolling(window).mean()) / df['close'].rolling(window).std()
        
        conditions = [
            (df['vol_zscore'] > entry_threshold) & (df['price_zscore'] < -1),
            (df['vol_zscore'] < -entry_threshold) | (df['price_zscore'] > 2),
        ]
        choices = [1, -1]
        
        df['signal'] = np.select(conditions, choices, default=0)
        
        return df
    
    def get_documentation(self) -> str:
        return """
## 波动率套利策略

### 数学原理
利用波动率的均值回归特性：
- **波动率积分**: 通过累积波动率变化判断极端状态
- **Z-Score标准化**: 识别偏离均值的标准差倍数

### 策略逻辑
1. 高波动+低价格 → 买入机会
2. 低波动+高价格 → 卖出信号
"""
```

**Step 2: 创建 strategies/kalman_filter.py**

```python
import numpy as np
import pandas as pd
from scipy import linalg
from .base import BaseStrategy
from . import register_strategy

@register_strategy
class KalmanFilterStrategy(BaseStrategy):
    name = "卡尔曼滤波"
    description = "使用卡尔曼滤波估计价格真实状态，过滤噪声"
    params = {
        "process_variance": 1e-5,
        "measurement_variance": 1e-3,
    }
    
    def get_required_columns(self) -> list[str]:
        return ["close"]
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        Q = self.params.get("process_variance", 1e-5)
        R = self.params.get("measurement_variance", 1e-3)
        
        n = len(df)
        x = np.zeros(n)
        P = np.zeros(n)
        
        x[0] = df['close'].iloc[0]
        P[0] = 1.0
        
        for i in range(1, n):
            x[i] = x[i-1]
            P[i] = P[i-1] + Q
            
            K = P[i] / (P[i] + R)
            x[i] = x[i] + K * (df['close'].iloc[i] - x[i])
            P[i] = (1 - K) * P[i]
        
        df['kalman_estimate'] = x
        df['kalma_derivative'] = np.gradient(x)
        df['residual'] = df['close'] - x
        
        residual_std = df['residual'].rolling(20).std()
        df['signal'] = np.where(
            df['residual'] < -2 * residual_std, 1,
            np.where(df['residual'] > 2 * residual_std, -1, 0)
        )
        
        return df
    
    def get_documentation(self) -> str:
        return """
## 卡尔曼滤波策略

### 数学原理
卡尔曼滤波是一种最优递归滤波算法：
- **状态预测**: x̂(k|k-1) = x̂(k-1|k-1)
- **协方差预测**: P(k|k-1) = P(k-1|k-1) + Q
- **卡尔曼增益**: K = P(k|k-1) / (P(k|k-1) + R)
- **状态更新**: x̂(k|k) = x̂(k|k-1) + K(z(k) - x̂(k|k-1))

### 策略逻辑
当价格显著低于滤波估计值时买入，显著高于时卖出。
"""
```

**Step 3: 更新 strategies/__init__.py**

```python
from .base import BaseStrategy
from .trend_strength import TrendStrengthStrategy
from .volatility import VolatilityStrategy
from .kalman_filter import KalmanFilterStrategy

STRATEGY_REGISTRY: dict[str, BaseStrategy] = {}

def register_strategy(cls):
    instance = cls()
    STRATEGY_REGISTRY[instance.name] = instance
    return cls

def get_all_strategies() -> list[BaseStrategy]:
    return list(STRATEGY_REGISTRY.values())

def get_strategy(name: str) -> BaseStrategy | None:
    return STRATEGY_REGISTRY.get(name)
```

**Step 4: 提交**

```bash
git add pixiu/strategies/
git commit -m "feat: add volatility and kalman filter strategies"
```

---

### Task 5.2: 完善设置页面和配置持久化

**Files:**
- Create: `pixiu/utils/config_manager.py`
- Modify: `pixiu/pages/settings.py`

**Step 1: 创建 utils/config_manager.py**

```python
import json
from pathlib import Path
from typing import Any

CONFIG_FILE = Path(__file__).parent.parent.parent / "data" / "user_config.json"

DEFAULT_CONFIG = {
    "glm_api_key": "",
    "default_market": "A股",
    "backtest_config": {
        "initial_capital": 100000,
        "commission_rate": 0.0003,
        "slippage_rate": 0.0001,
    }
}

def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()

def save_config(config: dict):
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def get_config(key: str, default: Any = None) -> Any:
    config = load_config()
    return config.get(key, default)

def set_config(key: str, value: Any):
    config = load_config()
    config[key] = value
    save_config(config)
```

**Step 2: 更新 pages/settings.py**

```python
import reflex as rx
from pixiu.state import State
from pixiu.utils.config_manager import load_config, save_config

class SettingsState(rx.State):
    glm_api_key: str = ""
    api_key_saved: bool = False
    
    def on_load(self):
        config = load_config()
        self.glm_api_key = config.get("glm_api_key", "")
    
    def save_api_key(self):
        from pixiu.services.ai_service import ai_service
        ai_service.configure(self.glm_api_key)
        
        config = load_config()
        config["glm_api_key"] = self.glm_api_key
        save_config(config)
        
        self.api_key_saved = True

def settings_page() -> rx.Component:
    return rx.vstack(
        rx.heading("⚙️ 设置", size="lg"),
        
        rx.box(
            rx.heading("GLM API 配置", size="md"),
            rx.text("配置智谱AI GLM-5 API密钥，用于生成智能分析报告"),
            rx.input(
                placeholder="请输入GLM API Key",
                type="password",
                value=SettingsState.glm_api_key,
                on_change=SettingsState.set_glm_api_key,
                margin_top="1rem",
            ),
            rx.hstack(
                rx.button("保存", on_click=SettingsState.save_api_key),
                rx.text("✓ 已保存", color="green") if SettingsState.api_key_saved else rx.fragment(),
                margin_top="1rem",
            ),
            padding="1rem",
            bg="gray.50",
            border_radius="md",
        ),
        
        rx.box(
            rx.heading("回测参数", size="md"),
            rx.text("默认回测配置"),
            rx.vstack(
                rx.hstack(
                    rx.text("初始资金:"),
                    rx.text("100,000"),
                ),
                rx.hstack(
                    rx.text("手续费率:"),
                    rx.text("0.03%"),
                ),
            ),
            padding="1rem",
            bg="gray.50",
            border_radius="md",
            margin_top="1rem",
        ),
        
        rx.box(
            rx.heading("数据管理", size="md"),
            rx.vstack(
                rx.button("清除所有缓存", color_scheme="red", on_click=lambda: None),
                rx.button("重新下载全部数据", on_click=lambda: None),
            ),
            padding="1rem",
            bg="gray.50",
            border_radius="md",
            margin_top="1rem",
        ),
        
        spacing="4",
        padding="4",
        on_mount=SettingsState.on_load,
    )
```

**Step 3: 提交**

```bash
git add pixiu/utils/config_manager.py pixiu/pages/settings.py
git commit -m "feat: add config persistence and improved settings page"
```

---

### Task 5.3: 最终集成测试

**Step 1: 完整功能测试**

Run: `reflex run`

测试流程：
1. 打开首页，搜索"茅台"
2. 选择股票，等待数据加载
3. 进入分析页面，选择策略
4. 执行回测，查看结果
5. 生成AI报告

**Step 2: 修复发现的问题**

根据测试结果修复任何bug

**Step 3: 最终提交**

```bash
git add .
git commit -m "feat: complete Pixiu quantitative analysis software v0.1.0"
```

---

## 完成标志

- [ ] Reflex应用正常启动
- [ ] 股票搜索功能正常
- [ ] 数据获取和存储正常
- [ ] 至少2个策略可用
- [ ] 回测引擎输出正确指标
- [ ] 图表正常显示
- [ ] AI报告生成功能正常
- [ ] 设置可持久化

---

## 后续优化方向

1. **性能优化**: 大数据量下的内存管理
2. **策略扩展**: 添加更多量化策略
3. **实时数据**: 支持实时行情推送
4. **打包发布**: 使用PyInstaller打包成可执行文件
5. **用户文档**: 编写使用说明
