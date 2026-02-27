# Pixiu 量化分析软件实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 构建一个基于Reflex的A股/港股/美股量化分析桌面软件，支持策略实验、回测分析和AI智能解读。

**Architecture:** 采用分层架构 - UI层(Reflex) → 服务层 → 策略层(插件式) → 数据层(SQLite+akshare)。策略通过装饰器自动注册，支持热扩展。

**Tech Stack:** Python 3.11+, Reflex, SQLite, Pandas, NumPy, SciPy, Plotly, akshare, GLM-5 API

---

## Phase 1: 项目初始化与数据层

### Task 1: 初始化Reflex项目

**Files:**

- Create: `requirements.txt`
- Create: `rxconfig.py`
- Create: `pixiu/__init__.py`
- Create: `pixiu/pixiu.py`

**Step 1: 创建requirements.txt**

```txt
reflex>=0.4.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
akshare>=1.12.0
plotly>=5.18.0
zhipuai>=2.0.0
aiosqlite>=0.19.0
python-dotenv>=1.0.0
```

**Step 2: 创建虚拟环境并安装依赖**

Run: `python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt`
Expected: 所有依赖安装成功

**Step 3: 初始化Reflex项目**

Run: `reflex init`
Expected: 创建Reflex项目结构

**Step 4: 配置rxconfig.py**

```python
import reflex as rx

config = rx.Config(
    app_name="pixiu",
    title="Pixiu 量化分析",
    description="A股/港股/美股量化分析软件",
)
```

**Step 5: 创建应用入口pixiu/pixiu.py**

```python
import reflex as rx

from pixiu.pages import home

app = rx.App()
app.add_page(home.page, route="/", title="Pixiu 量化分析")
```

**Step 6: 验证项目运行**

Run: `reflex run`
Expected: 浏览器打开 http://localhost:3000 显示空白页面

**Step 7: Commit**

```bash
git add .
git commit -m "feat: initialize Reflex project structure"
```

---

### Task 2: 创建配置管理模块

**Files:**

- Create: `pixiu/config.py`
- Create: `.env.example`
- Create: `tests/test_config.py`

**Step 1: 创建测试文件**

```python
# tests/test_config.py
import pytest
from pixiu.config import Config

def test_config_default_values():
    config = Config()
    assert config.database_path == "data/stocks.db"
    assert config.cache_dir == "data/cache"
    assert config.initial_capital == 100000

def test_config_from_env(monkeypatch):
    monkeypatch.setenv("GLM_API_KEY", "test_key_123")
    config = Config()
    assert config.glm_api_key == "test_key_123"
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_config.py -v`
Expected: FAIL - ModuleNotFoundError

**Step 3: 创建配置模块**

```python
# pixiu/config.py
import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass, field

load_dotenv()

@dataclass
class Config:
    database_path: str = field(default_factory=lambda: os.getenv("DATABASE_PATH", "data/stocks.db"))
    cache_dir: str = field(default_factory=lambda: os.getenv("CACHE_DIR", "data/cache"))
    glm_api_key: str = field(default_factory=lambda: os.getenv("GLM_API_KEY", ""))

    initial_capital: float = 100000.0
    commission_rate: float = 0.0003
    slippage_rate: float = 0.0001
    position_size: float = 0.95
    risk_free_rate: float = 0.03

    data_update_days: int = 30

    @property
    def base_dir(self) -> Path:
        return Path(__file__).parent.parent

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    def ensure_dirs(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

config = Config()
```

**Step 4: 创建.env.example**

```env
GLM_API_KEY=your_glm_api_key_here
DATABASE_PATH=data/stocks.db
CACHE_DIR=data/cache
```

**Step 5: 运行测试验证通过**

Run: `pytest tests/test_config.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add pixiu/config.py tests/test_config.py .env.example
git commit -m "feat: add configuration management module"
```

---

### Task 3: 创建数据模型

**Files:**

- Create: `pixiu/models/__init__.py`
- Create: `pixiu/models/stock.py`
- Create: `pixiu/models/backtest.py`
- Create: `pixiu/models/signal.py`
- Create: `tests/test_models.py`

**Step 1: 创建测试文件**

```python
# tests/test_models.py
import pytest
from datetime import datetime
from pixiu.models.stock import Stock
from pixiu.models.backtest import BacktestResult, Trade
from pixiu.models.signal import Signal

def test_stock_model():
    stock = Stock(
        code="600519.SH",
        name="贵州茅台",
        market="A股",
        industry="白酒"
    )
    assert stock.code == "600519.SH"
    assert stock.name == "贵州茅台"

def test_backtest_result():
    result = BacktestResult(
        total_return=0.25,
        annualized_return=0.15,
        max_drawdown=0.10,
        sharpe_ratio=1.5,
        win_rate=0.6,
        profit_loss_ratio=2.0,
        calmar_ratio=1.5,
        total_trades=50
    )
    assert result.total_return == 0.25
    assert result.win_rate == 0.6

def test_signal_model():
    signal = Signal(
        code="600519.SH",
        strategy_name="趋势强度",
        signal_date="2024-01-15",
        signal_type="BUY",
        confidence=0.85,
        price=1800.0
    )
    assert signal.signal_type == "BUY"
    assert signal.confidence == 0.85
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_models.py -v`
Expected: FAIL - ModuleNotFoundError

**Step 3: 创建Stock模型**

```python
# pixiu/models/stock.py
from dataclasses import dataclass
from datetime import date
from typing import Optional

@dataclass
class Stock:
    code: str
    name: str
    market: str
    industry: Optional[str] = None
    list_date: Optional[date] = None
    updated_at: Optional[str] = None

@dataclass  
class DailyQuote:
    code: str
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    turnover_rate: Optional[float] = None
```

**Step 4: 创建Backtest模型**

```python
# pixiu/models/backtest.py
from dataclasses import dataclass, field
from datetime import date
from typing import List

@dataclass
class Trade:
    trade_date: date
    signal_type: str
    shares: int
    price: float
    amount: float
    commission: float

@dataclass
class BacktestResult:
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_loss_ratio: float
    calmar_ratio: float
    total_trades: int
    start_date: str = ""
    end_date: str = ""
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
```

**Step 5: 创建Signal模型**

```python
# pixiu/models/signal.py
from dataclasses import dataclass
from datetime import date
from typing import Optional, Dict

@dataclass
class Signal:
    code: str
    strategy_name: str
    signal_date: str
    signal_type: str
    confidence: float
    price: float
    metadata: Optional[Dict] = None
```

**Step 6: 创建__init__.py**

```python
# pixiu/models/__init__.py
from .stock import Stock, DailyQuote
from .backtest import BacktestResult, Trade
from .signal import Signal

__all__ = ["Stock", "DailyQuote", "BacktestResult", "Trade", "Signal"]
```

**Step 7: 运行测试验证通过**

Run: `pytest tests/test_models.py -v`
Expected: PASS

**Step 8: Commit**

```bash
git add pixiu/models/ tests/test_models.py
git commit -m "feat: add data models for stock, backtest and signal"
```

---

### Task 4: 创建数据库服务

**Files:**

- Create: `pixiu/services/__init__.py`
- Create: `pixiu/services/database.py`
- Create: `tests/test_database.py`

**Step 1: 创建测试文件**

```python
# tests/test_database.py
import pytest
import asyncio
from pathlib import Path
from pixiu.services.database import Database
from pixiu.models.stock import Stock, DailyQuote

@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test.db"
    return Database(str(db_path))

def test_database_init(db):
    assert db is not None

@pytest.mark.asyncio
async def test_create_tables(db):
    await db.create_tables()
    assert Path(db.db_path).exists()

@pytest.mark.asyncio
async def test_insert_stock(db):
    await db.create_tables()
    stock = Stock(code="600519.SH", name="贵州茅台", market="A股")
    await db.insert_stock(stock)

    result = await db.get_stock("600519.SH")
    assert result is not None
    assert result.name == "贵州茅台"
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_database.py -v`
Expected: FAIL - ModuleNotFoundError

**Step 3: 创建数据库服务**

```python
# pixiu/services/database.py
import aiosqlite
from typing import Optional, List
from datetime import date
from pixiu.models.stock import Stock, DailyQuote

class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path

    async def create_tables(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS stocks (
                    code TEXT PRIMARY KEY,
                    name TEXT,
                    market TEXT,
                    industry TEXT,
                    list_date DATE,
                    updated_at TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS daily_quotes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code TEXT,
                    trade_date DATE,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    amount REAL,
                    turnover_rate REAL,
                    FOREIGN KEY (code) REFERENCES stocks(code),
                    UNIQUE(code, trade_date)
                );

                CREATE INDEX IF NOT EXISTS idx_quotes_code_date 
                ON daily_quotes(code, trade_date);

                CREATE TABLE IF NOT EXISTS strategy_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code TEXT,
                    strategy_name TEXT,
                    signal_date DATE,
                    signal_type TEXT,
                    confidence REAL,
                    price REAL,
                    metadata TEXT
                );

                CREATE TABLE IF NOT EXISTS update_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market TEXT,
                    last_update TIMESTAMP,
                    records_updated INTEGER
                );
            """)
            await db.commit()

    async def insert_stock(self, stock: Stock):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO stocks (code, name, market, industry, list_date, updated_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
            """, (stock.code, stock.name, stock.market, stock.industry, stock.list_date))
            await db.commit()

    async def get_stock(self, code: str) -> Optional[Stock]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM stocks WHERE code = ?", (code,))
            row = await cursor.fetchone()
            if row:
                return Stock(**dict(row))
            return None

    async def insert_quotes(self, quotes: List[DailyQuote]):
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany("""
                INSERT OR REPLACE INTO daily_quotes 
                (code, trade_date, open, high, low, close, volume, amount, turnover_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (q.code, q.trade_date, q.open, q.high, q.low, q.close, q.volume, q.amount, q.turnover_rate)
                for q in quotes
            ])
            await db.commit()

    async def get_quotes(self, code: str, start_date: str = None, end_date: str = None) -> List[DailyQuote]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            query = "SELECT * FROM daily_quotes WHERE code = ?"
            params = [code]

            if start_date:
                query += " AND trade_date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND trade_date <= ?"
                params.append(end_date)

            query += " ORDER BY trade_date"

            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            return [DailyQuote(**dict(row)) for row in rows]

    async def get_last_update(self, market: str) -> Optional[str]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT MAX(last_update) FROM update_logs WHERE market = ?",
                (market,)
            )
            result = await cursor.fetchone()
            return result[0] if result else None

    async def log_update(self, market: str, records_count: int):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO update_logs (market, last_update, records_updated)
                VALUES (?, datetime('now'), ?)
            """, (market, records_count))
            await db.commit()
```

**Step 4: 创建__init__.py**

```python
# pixiu/services/__init__.py
from .database import Database

__all__ = ["Database"]
```

**Step 5: 运行测试验证通过**

Run: `pytest tests/test_database.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add pixiu/services/ tests/test_database.py
git commit -m "feat: add SQLite database service with async support"
```

---

### Task 5: 创建数据获取服务

**Files:**

- Create: `pixiu/services/data_service.py`
- Create: `tests/test_data_service.py`

**Step 1: 创建测试文件**

```python
# tests/test_data_service.py
import pytest
from unittest.mock import patch, MagicMock
from pixiu.services.data_service import DataService
from pixiu.services.database import Database

@pytest.fixture
def data_service(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    return DataService(db)

def test_data_service_init(data_service):
    assert data_service is not None

@pytest.mark.asyncio
async def test_search_stocks(data_service):
    with patch('akshare.stock_zh_a_spot_em') as mock_ak:
        mock_ak.return_value = MagicMock()
        mock_ak.return_value.itertuples.return_value = []
        result = await data_service.search_stocks("茅台", "A股")
        assert result is not None
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_data_service.py -v`
Expected: FAIL - ModuleNotFoundError

**Step 3: 创建数据获取服务**

```python
# pixiu/services/data_service.py
import asyncio
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import akshare as ak
from loguru import logger

from pixiu.services.database import Database
from pixiu.models.stock import Stock, DailyQuote

class DataService:
    MARKET_MAP = {
        "A股": "stock_zh_a",
        "港股": "stock_hk", 
        "美股": "stock_us"
    }

    def __init__(self, db: Database):
        self.db = db

    async def search_stocks(self, keyword: str, market: str = "A股") -> List[Stock]:
        """搜索股票"""
        try:
            if market == "A股":
                df = await asyncio.to_thread(ak.stock_zh_a_spot_em)
                filtered = df[df['名称'].str.contains(keyword, na=False)]
                return [
                    Stock(code=row['代码'], name=row['名称'], market="A股")
                    for _, row in filtered.head(20).iterrows()
                ]
            elif market == "港股":
                df = await asyncio.to_thread(ak.stock_hk_spot_em)
                filtered = df[df['名称'].str.contains(keyword, na=False)]
                return [
                    Stock(code=row['代码'], name=row['名称'], market="港股")
                    for _, row in filtered.head(20).iterrows()
                ]
            elif market == "美股":
                df = await asyncio.to_thread(ak.stock_us_spot_em)
                filtered = df[df['名称'].str.contains(keyword, na=False)]
                return [
                    Stock(code=row['代码'], name=row['名称'], market="美股")
                    for _, row in filtered.head(20).iterrows()
                ]
        except Exception as e:
            logger.error(f"搜索股票失败: {e}")
            return []

    async def fetch_stock_history(
        self, 
        code: str, 
        market: str = "A股",
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """获取股票历史数据"""
        try:
            if not end_date:
                end_date = datetime.now().strftime("%Y%m%d")
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y%m%d")

            if market == "A股":
                df = await asyncio.to_thread(
                    ak.stock_zh_a_hist,
                    symbol=code.split('.')[0],
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"
                )
                df = df.rename(columns={
                    '日期': 'trade_date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume',
                    '成交额': 'amount',
                    '换手率': 'turnover_rate'
                })
            elif market == "港股":
                df = await asyncio.to_thread(
                    ak.stock_hk_daily_hist,
                    symbol=code,
                    adjust="qfq"
                )
            elif market == "美股":
                df = await asyncio.to_thread(
                    ak.stock_us_hist,
                    symbol=code,
                    adjust="qfq"
                )

            df['code'] = code
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            return df.sort_values('trade_date')

        except Exception as e:
            logger.error(f"获取股票历史数据失败: {e}")
            return pd.DataFrame()

    async def download_and_save(
        self, 
        code: str, 
        market: str = "A股",
        force_full: bool = False
    ) -> Tuple[bool, int]:
        """下载并保存股票数据"""
        try:
            last_update = await self.db.get_last_update(market)

            if last_update and not force_full:
                start_date = (datetime.fromisoformat(last_update) + timedelta(days=1)).strftime("%Y%m%d")
            else:
                start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y%m%d")

            df = await self.fetch_stock_history(code, market, start_date)

            if df.empty:
                return False, 0

            quotes = [
                DailyQuote(
                    code=row['code'],
                    trade_date=row['trade_date'].date(),
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume'],
                    amount=row.get('amount', 0),
                    turnover_rate=row.get('turnover_rate', 0)
                )
                for _, row in df.iterrows()
            ]

            await self.db.insert_quotes(quotes)
            await self.db.log_update(market, len(quotes))

            return True, len(quotes)

        except Exception as e:
            logger.error(f"下载保存数据失败: {e}")
            return False, 0

    async def get_cached_data(self, code: str) -> pd.DataFrame:
        """获取缓存的数据"""
        quotes = await self.db.get_quotes(code)
        if not quotes:
            return pd.DataFrame()

        df = pd.DataFrame([
            {
                'trade_date': q.trade_date,
                'open': q.open,
                'high': q.high,
                'low': q.low,
                'close': q.close,
                'volume': q.volume,
                'amount': q.amount,
                'turnover_rate': q.turnover_rate
            }
            for q in quotes
        ])
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        return df.set_index('trade_date')
```

**Step 4: 更新requirements.txt添加loguru**

```txt
loguru>=0.7.0
```

**Step 5: 运行测试验证通过**

Run: `pytest tests/test_data_service.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add pixiu/services/data_service.py tests/test_data_service.py requirements.txt
git commit -m "feat: add data service with akshare integration"
```

---

## Phase 2: 策略层

### Task 6: 创建策略基类和注册机制

**Files:**

- Create: `pixiu/strategies/__init__.py`
- Create: `pixiu/strategies/base.py`
- Create: `tests/test_strategy_base.py`

**Step 1: 创建测试文件**

```python
# tests/test_strategy_base.py
import pytest
from pixiu.strategies.base import BaseStrategy
from pixiu.strategies import register_strategy, get_all_strategies, STRATEGY_REGISTRY

class MockStrategy(BaseStrategy):
    name = "测试策略"
    description = "用于测试的策略"

    def generate_signals(self, df):
        df['signal'] = 0
        return df

    def get_required_data(self):
        return ['close']

def test_strategy_base():
    strategy = MockStrategy()
    assert strategy.name == "测试策略"

def test_register_strategy():
    registry_before = len(STRATEGY_REGISTRY)

    @register_strategy
    class TestRegStrategy(BaseStrategy):
        name = "注册测试"

        def generate_signals(self, df):
            df['signal'] = 0
            return df

        def get_required_data(self):
            return []

    assert "注册测试" in STRATEGY_REGISTRY
    assert len(STRATEGY_REGISTRY) == registry_before + 1
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_strategy_base.py -v`
Expected: FAIL - ModuleNotFoundError

**Step 3: 创建策略基类**

```python
# pixiu/strategies/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import pandas as pd

class BaseStrategy(ABC):
    """所有策略必须继承此类"""

    name: str = ""
    description: str = ""
    params: Dict[str, Any] = {}

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """输入行情数据，输出带信号的DataFrame

        Args:
            df: 包含 open, high, low, close, volume 等列的DataFrame

        Returns:
            添加 signal 列的DataFrame
            signal: 1=买入, -1=卖出, 0=持有
        """
        pass

    @abstractmethod
    def get_required_data(self) -> List[str]:
        """返回需要的数据列"""
        pass

    def validate_params(self) -> bool:
        """参数校验"""
        return True

    def get_documentation(self) -> str:
        """返回策略的数学原理说明（Markdown格式）"""
        return f"## {self.name}\n\n{self.description}"

    def get_param_schema(self) -> Dict[str, Any]:
        """返回参数schema用于UI生成"""
        return {
            "type": "object",
            "properties": {
                k: {"type": "number", "default": v}
                for k, v in self.params.items()
            }
        }
```

**Step 4: 创建策略注册机制**

```python
# pixiu/strategies/__init__.py
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseStrategy

STRATEGY_REGISTRY: Dict[str, "BaseStrategy"] = {}

def register_strategy(cls):
    """装饰器：自动注册策略"""
    instance = cls()
    STRATEGY_REGISTRY[instance.name] = instance
    return cls

def get_all_strategies() -> List["BaseStrategy"]:
    """获取所有已注册策略"""
    return list(STRATEGY_REGISTRY.values())

def get_strategy(name: str) -> "BaseStrategy | None":
    """按名称获取策略"""
    return STRATEGY_REGISTRY.get(name)

from .base import BaseStrategy
```

**Step 5: 运行测试验证通过**

Run: `pytest tests/test_strategy_base.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add pixiu/strategies/ tests/test_strategy_base.py
git commit -m "feat: add strategy base class and registration mechanism"
```

---

### Task 7: 实现趋势强度策略

**Files:**

- Create: `pixiu/strategies/trend_strength.py`
- Create: `tests/test_trend_strength.py`

**Step 1: 创建测试文件**

```python
# tests/test_trend_strength.py
import pytest
import pandas as pd
import numpy as np
from pixiu.strategies.trend_strength import TrendStrengthStrategy
from pixiu.strategies import STRATEGY_REGISTRY

def test_trend_strength_registered():
    assert "趋势强度策略" in STRATEGY_REGISTRY

def test_trend_strength_basic():
    strategy = TrendStrengthStrategy()

    dates = pd.date_range('2024-01-01', periods=100)
    prices = 100 + np.cumsum(np.random.randn(100) * 2)

    df = pd.DataFrame({
        'trade_date': dates,
        'close': prices,
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'volume': np.random.randint(1000000, 2000000, 100)
    })

    result = strategy.generate_signals(df)

    assert 'signal' in result.columns
    assert 'trend_strength' in result.columns
    assert set(result['signal'].unique()).issubset({-1, 0, 1})

def test_trend_strength_uptrend():
    strategy = TrendStrengthStrategy()

    dates = pd.date_range('2024-01-01', periods=100)
    prices = 100 + np.arange(100) * 0.5

    df = pd.DataFrame({
        'trade_date': dates,
        'close': prices,
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'volume': np.ones(100) * 1000000
    })

    result = strategy.generate_signals(df)

    assert result['signal'].iloc[-1] == 1
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_trend_strength.py -v`
Expected: FAIL - ModuleNotFoundError

**Step 3: 创建趋势强度策略**

```python
# pixiu/strategies/trend_strength.py
import numpy as np
import pandas as pd
from typing import List
from pixiu.strategies.base import BaseStrategy
from pixiu.strategies import register_strategy

@register_strategy
class TrendStrengthStrategy(BaseStrategy):
    """
    基于价格导数判断趋势强度

    数学原理：
    - 一阶导数 f'(t)：价格变化率，判断趋势方向
    - 二阶导数 f''(t)：变化加速度，判断趋势强度

    信号生成：
    - f'(t) > 0 且 f''(t) > 0 → 强买 (1)
    - f'(t) < 0 且 f''(t) < 0 → 强卖 (-1)
    - 其他 → 持有 (0)
    """

    name = "趋势强度策略"
    description = "基于微积分导数分析价格趋势方向和强度"
    params = {
        "window": 20,
        "strength_threshold": 0.02
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        window = self.params.get("window", 20)

        df['price_derivative'] = np.gradient(df['close'])
        df['price_acceleration'] = np.gradient(df['price_derivative'])

        rolling_std = df['close'].rolling(window=window).std()
        df['trend_strength'] = df['price_derivative'] / rolling_std

        df['signal'] = np.where(
            (df['price_derivative'] > 0) & (df['price_acceleration'] > 0),
            1,
            np.where(
                (df['price_derivative'] < 0) & (df['price_acceleration'] < 0),
                -1,
                0
            )
        )

        df.loc[:window, 'signal'] = 0

        return df

    def get_required_data(self) -> List[str]:
        return ['close']

    def get_documentation(self) -> str:
        return """
## 趋势强度策略

### 数学原理

本策略基于微积分中的导数概念：

**一阶导数（价格变化率）**
$$f'(t) = \\frac{dP}{dt} \\approx \\frac{P(t) - P(t-1)}{\\Delta t}$$

- f'(t) > 0：价格上升
- f'(t) < 0：价格下降

**二阶导数（加速度）**
$$f''(t) = \\frac{d^2P}{dt^2}$$

- f''(t) > 0：上升趋势加强
- f''(t) < 0：下降趋势加强

### 交易规则

| 条件 | 信号 | 说明 |
|------|------|------|
| f'(t) > 0 且 f''(t) > 0 | 买入 | 趋势向上且加速 |
| f'(t) < 0 且 f''(t) < 0 | 卖出 | 趋势向下且加速 |
| 其他 | 持有 | 趋势不明确 |
"""
```

**Step 4: 运行测试验证通过**

Run: `pytest tests/test_trend_strength.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pixiu/strategies/trend_strength.py tests/test_trend_strength.py
git commit -m "feat: implement trend strength strategy with calculus"
```

---

### Task 8: 实现波动率套利策略

**Files:**

- Create: `pixiu/strategies/volatility.py`
- Create: `tests/test_volatility.py`

**Step 1: 创建测试文件**

```python
# tests/test_volatility.py
import pytest
import pandas as pd
import numpy as np
from pixiu.strategies.volatility import VolatilityStrategy
from pixiu.strategies import STRATEGY_REGISTRY

def test_volatility_registered():
    assert "波动率套利策略" in STRATEGY_REGISTRY

def test_volatility_basic():
    strategy = VolatilityStrategy()

    dates = pd.date_range('2024-01-01', periods=100)
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100))

    df = pd.DataFrame({
        'trade_date': dates,
        'close': prices,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'volume': np.random.randint(1000000, 2000000, 100)
    })

    result = strategy.generate_signals(df)

    assert 'signal' in result.columns
    assert 'volatility' in result.columns
    assert 'volatility_integral' in result.columns
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_volatility.py -v`
Expected: FAIL

**Step 3: 创建波动率套利策略**

```python
# pixiu/strategies/volatility.py
import numpy as np
import pandas as pd
from scipy import integrate
from typing import List
from pixiu.strategies.base import BaseStrategy
from pixiu.strategies import register_strategy

@register_strategy
class VolatilityStrategy(BaseStrategy):
    """
    基于波动率积分的均值回归策略

    数学原理：
    - 波动率 σ：价格标准差
    - 波动率积分 ∫σ dt：累积波动能量
    - 高累积波动后往往出现回归

    信号生成：
    - 波动率积分超过阈值 → 预期回归 → 反向操作
    """

    name = "波动率套利策略"
    description = "基于波动率积分分析，捕捉均值回归机会"
    params = {
        "window": 20,
        "entry_threshold": 2.0,
        "exit_threshold": 0.5
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        window = self.params.get("window", 20)
        entry_threshold = self.params.get("entry_threshold", 2.0)

        returns = df['close'].pct_change()
        df['volatility'] = returns.rolling(window=window).std() * np.sqrt(252)

        volatility_values = df['volatility'].fillna(0).values
        df['volatility_integral'] = 0.0

        for i in range(1, len(df)):
            x = np.arange(i)
            y = volatility_values[:i]
            mask = ~np.isnan(y)
            if mask.sum() > 1:
                df.loc[df.index[i], 'volatility_integral'] = integrate.trapezoid(y[mask], x[mask])

        vol_integral_normalized = df['volatility_integral'] / df['volatility_integral'].rolling(window*2).mean()

        df['signal'] = 0
        df.loc[vol_integral_normalized > entry_threshold, 'signal'] = -1
        df.loc[vol_integral_normalized < -entry_threshold, 'signal'] = 1

        df.loc[:window*2, 'signal'] = 0

        return df

    def get_required_data(self) -> List[str]:
        return ['close']

    def get_documentation(self) -> str:
        return """
## 波动率套利策略

### 数学原理

**波动率计算**
$$\\sigma_t = \\sqrt{\\frac{1}{N}\\sum_{i=0}^{N-1}(r_{t-i} - \\bar{r})^2}$$

**波动率积分（累积波动能量）**
$$E(t) = \\int_0^t \\sigma(\\tau) d\\tau$$

### 交易逻辑

高波动率累积后，市场往往出现均值回归：

- 积分值异常高 → 市场过度波动 → 预期回归 → 卖出
- 积分值异常低 → 市场过度平静 → 预期突破 → 买入
"""
```

**Step 4: 运行测试验证通过**

Run: `pytest tests/test_volatility.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pixiu/strategies/volatility.py tests/test_volatility.py
git commit -m "feat: implement volatility arbitrage strategy"
```

---

### Task 9: 实现卡尔曼滤波策略

**Files:**

- Create: `pixiu/strategies/kalman_filter.py`
- Create: `tests/test_kalman_filter.py`

**Step 1: 创建测试文件**

```python
# tests/test_kalman_filter.py
import pytest
import pandas as pd
import numpy as np
from pixiu.strategies.kalman_filter import KalmanFilterStrategy
from pixiu.strategies import STRATEGY_REGISTRY

def test_kalman_registered():
    assert "卡尔曼滤波策略" in STRATEGY_REGISTRY

def test_kalman_basic():
    strategy = KalmanFilterStrategy()

    dates = pd.date_range('2024-01-01', periods=100)
    np.random.seed(42)
    true_price = 100 + np.cumsum(np.random.randn(100) * 0.5)
    noise = np.random.randn(100) * 2
    observed = true_price + noise

    df = pd.DataFrame({
        'trade_date': dates,
        'close': observed,
        'high': observed * 1.02,
        'low': observed * 0.98,
        'volume': np.ones(100) * 1000000
    })

    result = strategy.generate_signals(df)

    assert 'signal' in result.columns
    assert 'kalman_estimate' in result.columns
    assert 'kalman_variance' in result.columns
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_kalman_filter.py -v`
Expected: FAIL

**Step 3: 创建卡尔曼滤波策略**

```python
# pixiu/strategies/kalman_filter.py
import numpy as np
import pandas as pd
from typing import List
from pixiu.strategies.base import BaseStrategy
from pixiu.strategies import register_strategy

@register_strategy
class KalmanFilterStrategy(BaseStrategy):
    """
    基于卡尔曼滤波的价格估计策略

    数学原理：
    状态空间模型：
    - 状态方程：x_t = A*x_{t-1} + w_t
    - 观测方程：y_t = H*x_t + v_t

    卡尔曼滤波递归估计真实价格，过滤噪声

    信号生成：
    - 观测价格 > 估计价格 + kσ → 卖出（价格偏高）
    - 观测价格 < 估计价格 - kσ → 买入（价格偏低）
    """

    name = "卡尔曼滤波策略"
    description = "使用卡尔曼滤波估计真实价格，捕捉价格偏离"
    params = {
        "process_variance": 1e-5,
        "measurement_variance": 1e-2,
        "signal_threshold": 1.5
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        Q = self.params.get("process_variance", 1e-5)
        R = self.params.get("measurement_variance", 1e-2)
        threshold = self.params.get("signal_threshold", 1.5)

        n = len(df)
        x = np.zeros(n)
        P = np.zeros(n)

        x[0] = df['close'].iloc[0]
        P[0] = 1.0

        for i in range(1, n):
            x_pred = x[i-1]
            P_pred = P[i-1] + Q

            K = P_pred / (P_pred + R)

            x[i] = x_pred + K * (df['close'].iloc[i] - x_pred)
            P[i] = (1 - K) * P_pred

        df['kalman_estimate'] = x
        df['kalman_variance'] = P
        df['kalman_std'] = np.sqrt(P)

        deviation = (df['close'] - df['kalman_estimate']) / df['kalman_std']

        df['signal'] = 0
        df.loc[deviation < -threshold, 'signal'] = 1
        df.loc[deviation > threshold, 'signal'] = -1

        return df

    def get_required_data(self) -> List[str]:
        return ['close']

    def get_documentation(self) -> str:
        return """
## 卡尔曼滤波策略

### 数学原理

卡尔曼滤波是一种最优递归估计算法，基于状态空间模型：

**预测步骤**
$$\\hat{x}_{t|t-1} = A\\hat{x}_{t-1}$$
$$P_{t|t-1} = AP_{t-1}A^T + Q$$

**更新步骤**
$$K_t = P_{t|t-1}H^T(HP_{t|t-1}H^T + R)^{-1}$$
$$\\hat{x}_t = \\hat{x}_{t|t-1} + K_t(y_t - H\\hat{x}_{t|t-1})$$

### 交易逻辑

- 观测价格显著低于滤波估计 → 价格被低估 → 买入
- 观测价格显著高于滤波估计 → 价格被高估 → 卖出
"""
```

**Step 4: 运行测试验证通过**

Run: `pytest tests/test_kalman_filter.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pixiu/strategies/kalman_filter.py tests/test_kalman_filter.py
git commit -m "feat: implement Kalman filter strategy"
```

---

## Phase 3: 回测引擎

### Task 10: 实现回测引擎

**Files:**

- Create: `pixiu/services/backtest_service.py`
- Create: `tests/test_backtest_service.py`

**Step 1: 创建测试文件**

```python
# tests/test_backtest_service.py
import pytest
import pandas as pd
import numpy as np
from pixiu.services.backtest_service import BacktestEngine, BacktestConfig
from pixiu.models.backtest import BacktestResult

@pytest.fixture
def sample_data():
    dates = pd.date_range('2024-01-01', periods=100)
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 2)

    return pd.DataFrame({
        'trade_date': dates,
        'close': prices,
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'volume': np.ones(100) * 1000000,
        'signal': np.random.choice([-1, 0, 1], 100)
    })

def test_backtest_engine_init():
    config = BacktestConfig(initial_capital=50000)
    engine = BacktestEngine(config)
    assert engine.config.initial_capital == 50000

def test_backtest_run(sample_data):
    engine = BacktestEngine()
    result = engine.run(sample_data)

    assert isinstance(result, BacktestResult)
    assert result.total_trades >= 0

def test_backtest_calculate_metrics(sample_data):
    engine = BacktestEngine()
    result = engine.run(sample_data)

    assert -1 <= result.max_drawdown <= 0
    assert result.sharpe_ratio != 0 or result.total_trades == 0
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_backtest_service.py -v`
Expected: FAIL

**Step 3: 创建回测引擎**

```python
# pixiu/services/backtest_service.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List
from datetime import date

from pixiu.models.backtest import BacktestResult, Trade
from pixiu.config import config

@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0
    commission_rate: float = 0.0003
    slippage_rate: float = 0.0001
    position_size: float = 0.95
    risk_free_rate: float = 0.03

class Portfolio:
    def __init__(self, initial_capital: float):
        self.cash = initial_capital
        self.shares = 0
        self.initial_capital = initial_capital
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []

    def buy(self, shares: int, price: float, date, commission_rate: float, slippage_rate: float):
        actual_price = price * (1 + slippage_rate)
        commission = shares * actual_price * commission_rate
        cost = shares * actual_price + commission

        if cost <= self.cash:
            self.cash -= cost
            self.shares += shares
            self.trades.append(Trade(
                trade_date=date,
                signal_type="BUY",
                shares=shares,
                price=actual_price,
                amount=shares * actual_price,
                commission=commission
            ))
            return True
        return False

    def sell(self, shares: int, price: float, date, commission_rate: float, slippage_rate: float):
        if shares > self.shares:
            shares = self.shares

        actual_price = price * (1 - slippage_rate)
        commission = shares * actual_price * commission_rate
        proceeds = shares * actual_price - commission

        self.cash += proceeds
        self.shares -= shares
        self.trades.append(Trade(
            trade_date=date,
            signal_type="SELL",
            shares=shares,
            price=actual_price,
            amount=shares * actual_price,
            commission=commission
        ))
        return True

    def get_value(self, current_price: float) -> float:
        return self.cash + self.shares * current_price

class BacktestEngine:
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig(
            initial_capital=config.initial_capital,
            commission_rate=config.commission_rate,
            slippage_rate=config.slippage_rate,
            position_size=config.position_size,
            risk_free_rate=config.risk_free_rate
        )

    def run(self, df: pd.DataFrame, signals: pd.Series = None) -> BacktestResult:
        if signals is not None:
            df = df.copy()
            df['signal'] = signals

        portfolio = Portfolio(self.config.initial_capital)
        equity_curve = []

        for i, row in df.iterrows():
            signal = row['signal']
            price = row['close']
            trade_date = row.get('trade_date', i)

            if signal == 1 and portfolio.cash > 0:
                shares = int(portfolio.cash * self.config.position_size / price)
                if shares > 0:
                    portfolio.buy(
                        shares, price, trade_date,
                        self.config.commission_rate,
                        self.config.slippage_rate
                    )

            elif signal == -1 and portfolio.shares > 0:
                portfolio.sell(
                    portfolio.shares, price, trade_date,
                    self.config.commission_rate,
                    self.config.slippage_rate
                )

            equity_curve.append(portfolio.get_value(price))

        return self._calculate_metrics(df, portfolio, equity_curve)

    def _calculate_metrics(
        self, 
        df: pd.DataFrame, 
        portfolio: Portfolio, 
        equity_curve: List[float]
    ) -> BacktestResult:
        equity = np.array(equity_curve)

        total_return = (equity[-1] - equity[0]) / equity[0]

        days = len(equity)
        annualized_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0

        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_drawdown = drawdown.min()

        returns = np.diff(equity) / equity[:-1]
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = (np.mean(returns) * 252 - self.config.risk_free_rate) / (np.std(returns) * np.sqrt(252))
        else:
            sharpe_ratio = 0

        trades = portfolio.trades
        if len(trades) >= 2:
            trade_returns = []
            for i in range(0, len(trades) - 1, 2):
                if i + 1 < len(trades):
                    buy_trade = trades[i]
                    sell_trade = trades[i + 1]
                    if buy_trade.signal_type == "BUY" and sell_trade.signal_type == "SELL":
                        trade_return = (sell_trade.price - buy_trade.price) / buy_trade.price
                        trade_returns.append(trade_return)

            if trade_returns:
                win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
                wins = [r for r in trade_returns if r > 0]
                losses = [r for r in trade_returns if r < 0]
                profit_loss_ratio = np.mean(wins) / abs(np.mean(losses)) if losses else 0
            else:
                win_rate = 0
                profit_loss_ratio = 0
        else:
            win_rate = 0
            profit_loss_ratio = 0

        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            profit_loss_ratio=profit_loss_ratio,
            calmar_ratio=calmar_ratio,
            total_trades=len(trades),
            start_date=str(df.iloc[0].get('trade_date', '')),
            end_date=str(df.iloc[-1].get('trade_date', '')),
            trades=trades,
            equity_curve=equity.tolist(),
            drawdown_curve=drawdown.tolist()
        )
```

**Step 4: 运行测试验证通过**

Run: `pytest tests/test_backtest_service.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pixiu/services/backtest_service.py tests/test_backtest_service.py
git commit -m "feat: implement backtest engine with full metrics"
```

---

## Phase 4: UI界面

### Task 11: 创建基础UI组件

**Files:**

- Create: `pixiu/components/__init__.py`
- Create: `pixiu/components/stock_selector.py`
- Create: `pixiu/components/chart_panel.py`

**Step 1: 创建股票选择器组件**

```python
# pixiu/components/stock_selector.py
import reflex as rx

def stock_selector(
    stocks: list,
    selected_code: str,
    on_select,
    on_search,
    search_value: str
) -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.select(
                ["A股", "港股", "美股"],
                default_value="A股",
                width="100px",
            ),
            rx.input(
                placeholder="搜索股票...",
                value=search_value,
                on_change=on_search,
                width="200px",
            ),
            width="100%",
        ),
        rx.box(
            rx.foreach(
                stocks[:10],
                lambda stock: rx.box(
                    rx.text(f"{stock['code']} - {stock['name']}"),
                    padding="0.5rem",
                    cursor="pointer",
                    on_click=lambda: on_select(stock['code']),
                    _hover={"bg": "gray.100"},
                ),
            ),
            max_height="200px",
            overflow_y="auto",
            margin_top="0.5rem",
        ),
        width="100%",
    )
```

**Step 2: 创建图表面板组件**

```python
# pixiu/components/chart_panel.py
import reflex as rx

def chart_panel(
    title: str,
    figure_data: dict,
    height: str = "400px"
) -> rx.Component:
    return rx.box(
        rx.heading(title, size="md", margin_bottom="1rem"),
        rx.plotly(
            data=figure_data.get("data", []),
            layout=figure_data.get("layout", {}),
            use_resize_handler=True,
            style={"width": "100%", "height": height},
        ),
        width="100%",
        padding="1rem",
        border_radius="md",
        bg="white",
        shadow="sm",
    )
```

**Step 3: 创建__init__.py**

```python
# pixiu/components/__init__.py
from .stock_selector import stock_selector
from .chart_panel import chart_panel

__all__ = ["stock_selector", "chart_panel"]
```

**Step 4: Commit**

```bash
git add pixiu/components/
git commit -m "feat: add basic UI components"
```

---

### Task 12: 创建应用状态管理

**Files:**

- Create: `pixiu/state.py`

**Step 1: 创建全局状态**

```python
# pixiu/state.py
import reflex as rx
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime

from pixiu.services.database import Database
from pixiu.services.data_service import DataService
from pixiu.services.backtest_service import BacktestEngine, BacktestConfig
from pixiu.strategies import get_all_strategies, get_strategy
from pixiu.config import config as app_config

class AppState(rx.State):
    is_loading: bool = False
    loading_message: str = ""
    progress: float = 0.0

    current_market: str = "A股"
    search_keyword: str = ""
    search_results: List[Dict] = []
    selected_stock: str = ""
    selected_stock_name: str = ""

    stock_data: Dict = {}
    chart_data: Dict = {}

    available_strategies: List[Dict] = []
    selected_strategies: List[str] = []
    strategy_params: Dict[str, Dict] = {}

    backtest_result: Dict = {}
    backtest_config: Dict = {
        "initial_capital": 100000,
        "commission_rate": 0.0003,
        "position_size": 0.95
    }

    ai_report: str = ""
    ai_generating: bool = False

    def __init__(self):
        super().__init__()
        self.db = Database(str(app_config.data_dir / "stocks.db"))
        self.data_service = DataService(self.db)
        self._load_strategies()

    def _load_strategies(self):
        strategies = get_all_strategies()
        self.available_strategies = [
            {
                "name": s.name,
                "description": s.description,
                "params": s.params
            }
            for s in strategies
        ]

    @rx.background
    async def search_stocks(self):
        async with self:
            self.is_loading = True
            self.loading_message = "搜索中..."

        results = await self.data_service.search_stocks(
            self.search_keyword,
            self.current_market
        )

        async with self:
            self.search_results = [
                {"code": s.code, "name": s.name, "market": s.market}
                for s in results
            ]
            self.is_loading = False

    @rx.background
    async def select_stock(self, code: str):
        async with self:
            self.is_loading = True
            self.loading_message = "加载数据..."
            self.selected_stock = code

        df = await self.data_service.get_cached_data(code)

        if df.empty:
            success, count = await self.data_service.download_and_save(
                code, self.current_market
            )
            if success:
                df = await self.data_service.get_cached_data(code)

        async with self:
            if not df.empty:
                self.stock_data = {
                    "dates": df.index.strftime("%Y-%m-%d").tolist(),
                    "close": df["close"].tolist(),
                    "volume": df["volume"].tolist()
                }
                self._update_chart_data(df)
            self.is_loading = False

    def _update_chart_data(self, df: pd.DataFrame):
        self.chart_data = {
            "data": [{
                "x": df.index.strftime("%Y-%m-%d").tolist(),
                "y": df["close"].tolist(),
                "type": "scatter",
                "mode": "lines",
                "name": "收盘价"
            }],
            "layout": {
                "title": f"{self.selected_stock_name} 价格走势",
                "xaxis": {"title": "日期"},
                "yaxis": {"title": "价格"},
                "showlegend": True
            }
        }

    def toggle_strategy(self, strategy_name: str):
        if strategy_name in self.selected_strategies:
            self.selected_strategies.remove(strategy_name)
        else:
            self.selected_strategies.append(strategy_name)

    @rx.background
    async def run_backtest(self):
        async with self:
            self.is_loading = True
            self.loading_message = "执行回测..."
            self.progress = 0

        df = pd.DataFrame(self.stock_data)
        df["trade_date"] = pd.to_datetime(df["dates"])
        df = df.set_index("trade_date")

        results = []
        total = len(self.selected_strategies)

        for i, strategy_name in enumerate(self.selected_strategies):
            strategy = get_strategy(strategy_name)
            if strategy:
                df_with_signals = strategy.generate_signals(df.reset_index())

                engine = BacktestEngine(BacktestConfig(**self.backtest_config))
                result = engine.run(df_with_signals)

                results.append({
                    "strategy": strategy_name,
                    "result": result
                })

            async with self:
                self.progress = (i + 1) / total * 100

        async with self:
            self.backtest_result = {
                "results": [
                    {
                        "strategy": r["strategy"],
                        "total_return": r["result"].total_return,
                        "annualized_return": r["result"].annualized_return,
                        "max_drawdown": r["result"].max_drawdown,
                        "sharpe_ratio": r["result"].sharpe_ratio,
                        "win_rate": r["result"].win_rate,
                    }
                    for r in results
                ]
            }
            self.is_loading = False

    def set_market(self, market: str):
        self.current_market = market

    def set_search_keyword(self, keyword: str):
        self.search_keyword = keyword
```

**Step 2: Commit**

```bash
git add pixiu/state.py
git commit -m "feat: add global state management"
```

---

### Task 13: 创建首页

**Files:**

- Modify: `pixiu/pages/home.py`

**Step 1: 创建首页组件**

```python
# pixiu/pages/home.py
import reflex as rx
from pixiu.state import AppState
from pixiu.components.stock_selector import stock_selector
from pixiu.components.chart_panel import chart_panel

def strategy_card(strategy: dict, selected: bool, on_toggle) -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.checkbox(
                checked=selected,
                on_change=lambda: on_toggle(strategy["name"])
            ),
            rx.vstack(
                rx.text(strategy["name"], font_weight="bold"),
                rx.text(strategy["description"], font_size="sm", color="gray.500"),
                align_items="start",
            ),
            width="100%",
        ),
        padding="1rem",
        border="1px solid",
        border_color="gray.200" if not selected else "blue.400",
        border_radius="md",
        cursor="pointer",
        on_click=lambda: on_toggle(strategy["name"]),
        _hover={"border_color": "blue.300"},
    )

def page() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.heading("📊 Pixiu 量化分析", size="lg"),
                rx.spacer(),
                rx.badge("v0.1.0"),
                width="100%",
                margin_bottom="1rem",
            ),

            rx.hstack(
                rx.text("市场:"),
                rx.button_group(
                    rx.button(
                        "A股",
                        variant="solid" if AppState.current_market == "A股" else "outline",
                        on_click=lambda: AppState.set_market("A股"),
                    ),
                    rx.button(
                        "港股",
                        variant="solid" if AppState.current_market == "港股" else "outline",
                        on_click=lambda: AppState.set_market("港股"),
                    ),
                    rx.button(
                        "美股",
                        variant="solid" if AppState.current_market == "美股" else "outline",
                        on_click=lambda: AppState.set_market("美股"),
                    ),
                ),
                width="100%",
                margin_bottom="1rem",
            ),

            rx.input(
                placeholder="输入股票代码或名称搜索...",
                value=AppState.search_keyword,
                on_change=AppState.set_search_keyword,
                on_key_down=lambda e: AppState.search_stocks() if e.key == "Enter" else None,
                width="100%",
                margin_bottom="0.5rem",
            ),

            rx.box(
                rx.foreach(
                    AppState.search_results,
                    lambda stock: rx.box(
                        rx.text(f"{stock['code']} - {stock['name']}"),
                        padding="0.5rem",
                        cursor="pointer",
                        on_click=lambda: AppState.select_stock(stock['code']),
                        _hover={"bg": "gray.100"},
                        border_bottom="1px solid",
                        border_color="gray.100",
                    ),
                ),
                max_height="150px",
                overflow_y="auto",
                width="100%",
            ),

            rx.cond(
                AppState.is_loading,
                rx.hstack(
                    rx.spinner(),
                    rx.text(AppState.loading_message),
                    margin_y="1rem",
                ),
            ),

            rx.cond(
                AppState.selected_stock != "",
                rx.box(
                    rx.heading(f"📈 {AppState.selected_stock}", size="md", margin_bottom="1rem"),

                    rx.plotly(
                        data=AppState.chart_data.get("data", []),
                        layout=AppState.chart_data.get("layout", {}),
                        use_resize_handler=True,
                        style={"width": "100%", "height": "400px"},
                    ),

                    rx.divider(margin_y="1rem"),

                    rx.heading("策略选择", size="md", margin_bottom="1rem"),

                    rx.grid(
                        rx.foreach(
                            AppState.available_strategies,
                            lambda s: strategy_card(
                                s,
                                AppState.selected_strategies.contains(s["name"]),
                                AppState.toggle_strategy
                            )
                        ),
                        columns="2",
                        spacing="1rem",
                        margin_bottom="1rem",
                    ),

                    rx.hstack(
                        rx.button(
                            "▶ 开始分析",
                            on_click=AppState.run_backtest,
                            is_disabled=len(AppState.selected_strategies) == 0,
                            color_scheme="blue",
                        ),
                        rx.spacer(),
                        rx.progress(
                            value=AppState.progress,
                            width="200px",
                        ),
                    ),

                    margin_top="1rem",
                ),
            ),

            rx.spacer(),

            rx.hstack(
                rx.text("Pixiu © 2024", color="gray.400"),
                rx.spacer(),
                rx.link("设置", href="/settings"),
            ),

            width="100%",
            max_width="1200px",
            margin="0 auto",
            padding="2rem",
        ),
        min_height="100vh",
        bg="gray.50",
    )
```

**Step 2: Commit**

```bash
git add pixiu/pages/home.py
git commit -m "feat: create home page with stock selector and strategy cards"
```

---

### Task 14: 创建回测结果页面

**Files:**

- Create: `pixiu/pages/backtest.py`

**Step 1: 创建回测页面**

```python
# pixiu/pages/backtest.py
import reflex as rx
from pixiu.state import AppState

def metric_card(title: str, value: str, color: str = "black") -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.text(title, font_size="sm", color="gray.500"),
            rx.text(value, font_size="xl", font_weight="bold", color=color),
            align_items="center",
        ),
        padding="1rem",
        bg="white",
        border_radius="md",
        shadow="sm",
        text_align="center",
    )

def backtest_result_item(result: dict) -> rx.Component:
    total_return = result.get("total_return", 0)
    color = "green.500" if total_return > 0 else "red.500"

    return rx.box(
        rx.heading(result.get("strategy", ""), size="md", margin_bottom="1rem"),

        rx.grid(
            metric_card("总收益率", f"{total_return:.2%}", color),
            metric_card("年化收益", f"{result.get('annualized_return', 0):.2%}"),
            metric_card("最大回撤", f"{result.get('max_drawdown', 0):.2%}"),
            metric_card("夏普比率", f"{result.get('sharpe_ratio', 0):.2f}"),
            metric_card("胜率", f"{result.get('win_rate', 0):.2%}"),
            columns="5",
            spacing="1rem",
            margin_bottom="1rem",
        ),

        rx.divider(),

        padding="1.5rem",
        bg="white",
        border_radius="md",
        shadow="md",
        margin_bottom="1rem",
    )

def page() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.heading("📋 回测报告", size="lg"),
                rx.spacer(),
                rx.button("返回", on_click=rx.redirect("/")),
                width="100%",
                margin_bottom="1rem",
            ),

            rx.cond(
                len(AppState.backtest_result.get("results", [])) > 0,
                rx.box(
                    rx.foreach(
                        AppState.backtest_result.get("results", []),
                        backtest_result_item
                    ),
                ),
                rx.box(
                    rx.text("暂无回测结果，请先选择股票和策略进行分析。"),
                    padding="2rem",
                    text_align="center",
                    color="gray.500",
                ),
            ),

            rx.spacer(),

            width="100%",
            max_width="1200px",
            margin="0 auto",
            padding="2rem",
        ),
        min_height="100vh",
        bg="gray.50",
    )
```

**Step 2: 更新pixiu.py添加页面**

```python
# pixiu/pixiu.py
import reflex as rx

from pixiu.pages import home, backtest

app = rx.App()
app.add_page(home.page, route="/", title="Pixiu 量化分析")
app.add_page(backtest.page, route="/backtest", title="回测报告 - Pixiu")
```

**Step 3: Commit**

```bash
git add pixiu/pages/backtest.py pixiu/pixiu.py
git commit -m "feat: add backtest results page"
```

---

### Task 15: 创建设置页面

**Files:**

- Create: `pixiu/pages/settings.py`

**Step 1: 创建设置页面**

```python
# pixiu/pages/settings.py
import reflex as rx
from pixiu.state import AppState
from pixiu.config import config

class SettingsState(rx.State):
    glm_api_key: str = ""
    initial_capital: str = "100000"
    commission_rate: str = "0.0003"

    def load_settings(self):
        self.glm_api_key = config.glm_api_key
        self.initial_capital = str(config.initial_capital)
        self.commission_rate = str(config.commission_rate)

    def save_settings(self):
        import os
        os.environ["GLM_API_KEY"] = self.glm_api_key
        config.glm_api_key = self.glm_api_key
        config.initial_capital = float(self.initial_capital)
        config.commission_rate = float(self.commission_rate)

def page() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.heading("⚙️ 设置", size="lg"),
                rx.spacer(),
                rx.button("返回", on_click=rx.redirect("/")),
                width="100%",
                margin_bottom="2rem",
            ),

            rx.box(
                rx.heading("GLM API 配置", size="md", margin_bottom="1rem"),
                rx.input(
                    placeholder="输入 GLM API Key",
                    value=SettingsState.glm_api_key,
                    on_change=SettingsState.set_glm_api_key,
                    type="password",
                    width="100%",
                ),
                rx.text("用于生成AI智能分析报告", font_size="sm", color="gray.500", margin_top="0.5rem"),
                padding="1.5rem",
                bg="white",
                border_radius="md",
                shadow="sm",
                margin_bottom="1rem",
            ),

            rx.box(
                rx.heading("回测参数", size="md", margin_bottom="1rem"),

                rx.vstack(
                    rx.hstack(
                        rx.text("初始资金:", width="100px"),
                        rx.input(
                            value=SettingsState.initial_capital,
                            on_change=SettingsState.set_initial_capital,
                            width="200px",
                        ),
                    ),
                    rx.hstack(
                        rx.text("手续费率:", width="100px"),
                        rx.input(
                            value=SettingsState.commission_rate,
                            on_change=SettingsState.set_commission_rate,
                            width="200px",
                        ),
                    ),
                    align_items="start",
                    spacing="1rem",
                ),

                padding="1.5rem",
                bg="white",
                border_radius="md",
                shadow="sm",
                margin_bottom="1rem",
            ),

            rx.button(
                "保存设置",
                on_click=[SettingsState.save_settings, lambda: rx.window.alert("设置已保存")],
                color_scheme="blue",
                margin_top="1rem",
            ),

            rx.spacer(),

            width="100%",
            max_width="600px",
            margin="0 auto",
            padding="2rem",
        ),
        min_height="100vh",
        bg="gray.50",
        on_mount=SettingsState.load_settings,
    )
```

**Step 2: 更新pixiu.py**

```python
# pixiu/pixiu.py
import reflex as rx

from pixiu.pages import home, backtest, settings

app = rx.App()
app.add_page(home.page, route="/", title="Pixiu 量化分析")
app.add_page(backtest.page, route="/backtest", title="回测报告 - Pixiu")
app.add_page(settings.page, route="/settings", title="设置 - Pixiu")
```

**Step 3: Commit**

```bash
git add pixiu/pages/settings.py pixiu/pixiu.py
git commit -m "feat: add settings page for API and backtest config"
```

---

## Phase 5: AI集成

### Task 16: 实现GLM AI服务

**Files:**

- Create: `pixiu/services/ai_service.py`
- Create: `tests/test_ai_service.py`

**Step 1: 创建测试文件**

```python
# tests/test_ai_service.py
import pytest
from unittest.mock import Mock, patch
from pixiu.services.ai_service import AIReportService

def test_ai_service_init():
    service = AIReportService("test_key")
    assert service.api_key == "test_key"

def test_build_prompt():
    service = AIReportService("test_key")

    result = {
        "total_return": 0.25,
        "annualized_return": 0.15,
        "max_drawdown": -0.10,
        "sharpe_ratio": 1.5,
        "win_rate": 0.6,
        "total_trades": 50
    }

    stock = {"name": "贵州茅台", "code": "600519.SH"}

    prompt = service._build_prompt(result, stock, "趋势强度策略")

    assert "贵州茅台" in prompt
    assert "趋势强度策略" in prompt
    assert "25%" in prompt
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_ai_service.py -v`
Expected: FAIL

**Step 3: 创建AI服务**

```python
# pixiu/services/ai_service.py
from typing import Dict, Optional
from zhipuai import ZhipuAI
from loguru import logger

class AIReportService:
    SYSTEM_PROMPT = """你是一位专业的量化分析师。
你的任务是将回测数据转化为易懂的投资建议和分析报告。

请用中文回答，报告应包含：
1. **策略表现总结** - 用简洁语言总结策略的整体表现
2. **优势分析** - 策略在哪些方面表现良好
3. **风险分析** - 需要注意的风险点
4. **改进建议** - 针对该策略的优化建议
5. **适用场景** - 该策略适合什么样的市场环境

请保持客观、专业的分析风格。"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = ZhipuAI(api_key=api_key) if api_key else None

    async def generate_analysis(
        self,
        backtest_result: Dict,
        stock_info: Dict,
        strategy_name: str
    ) -> str:
        """生成自然语言分析报告"""

        if not self.client:
            return "请先在设置页面配置 GLM API Key"

        try:
            prompt = self._build_prompt(backtest_result, stock_info, strategy_name)

            response = self.client.chat.completions.create(
                model="glm-5",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"AI分析生成失败: {e}")
            return f"AI分析生成失败: {str(e)}"

    def _build_prompt(
        self,
        result: Dict,
        stock: Dict,
        strategy: str
    ) -> str:
        return f"""请分析以下回测结果：

## 基本信息
- **股票**: {stock.get('name', '')} ({stock.get('code', '')})
- **策略**: {strategy}
- **回测周期**: {result.get('start_date', '')} 至 {result.get('end_date', '')}

## 核心指标

| 指标 | 数值 |
|------|------|
| 总收益率 | {result.get('total_return', 0):.2%} |
| 年化收益率 | {result.get('annualized_return', 0):.2%} |
| 最大回撤 | {result.get('max_drawdown', 0):.2%} |
| 夏普比率 | {result.get('sharpe_ratio', 0):.2f} |
| 胜率 | {result.get('win_rate', 0):.2%} |
| 盈亏比 | {result.get('profit_loss_ratio', 0):.2f} |
| 卡玛比率 | {result.get('calmar_ratio', 0):.2f} |
| 总交易次数 | {result.get('total_trades', 0)} |

请给出详细的分析报告。
"""

    async def generate_comparison(
        self,
        results: list[Dict],
        stock_info: Dict
    ) -> str:
        """生成多策略对比分析"""

        if not self.client:
            return "请先配置 GLM API Key"

        strategies_info = "\n".join([
            f"- {r['strategy']}: 年化收益 {r['result']['annualized_return']:.2%}, "
            f"夏普比率 {r['result']['sharpe_ratio']:.2f}, "
            f"最大回撤 {r['result']['max_drawdown']:.2%}"
            for r in results
        ])

        prompt = f"""请对比分析以下多个策略在股票 {stock_info.get('name', '')} 上的表现：

{strategies_info}

请给出：
1. 各策略的优劣势对比
2. 最推荐的策略及理由
3. 组合使用建议
"""

        try:
            response = self.client.chat.completions.create(
                model="glm-5",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"AI对比分析失败: {e}")
            return f"分析失败: {str(e)}"
```

**Step 4: 运行测试验证通过**

Run: `pytest tests/test_ai_service.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pixiu/services/ai_service.py tests/test_ai_service.py
git commit -m "feat: implement GLM-5 AI analysis service"
```

---

### Task 17: 集成AI到状态和UI

**Files:**

- Modify: `pixiu/state.py`
- Modify: `pixiu/pages/backtest.py`

**Step 1: 更新state.py添加AI功能**

```python
# 在 AppState 类中添加

from pixiu.services.ai_service import AIReportService

class AppState(rx.State):
    # ... 现有属性 ...

    ai_report: str = ""
    ai_generating: bool = False
    ai_service: AIReportService = None

    def __init__(self):
        super().__init__()
        # ... 现有初始化 ...
        self._init_ai_service()

    def _init_ai_service(self):
        from pixiu.config import config
        if config.glm_api_key:
            self.ai_service = AIReportService(config.glm_api_key)

    @rx.background
    async def generate_ai_report(self):
        if not self.ai_service:
            async with self:
                self.ai_report = "请先在设置页面配置 GLM API Key"
            return

        async with self:
            self.ai_generating = True
            self.ai_report = ""

        results = self.backtest_result.get("results", [])
        if not results:
            async with self:
                self.ai_report = "暂无回测结果"
                self.ai_generating = False
            return

        first_result = results[0]
        stock_info = {
            "name": self.selected_stock_name,
            "code": self.selected_stock
        }

        report = await self.ai_service.generate_analysis(
            first_result,
            stock_info,
            first_result.get("strategy", "")
        )

        async with self:
            self.ai_report = report
            self.ai_generating = False
```

**Step 2: 更新backtest.py添加AI报告区域**

```python
# 在 backtest.py 中添加

def ai_report_section() -> rx.Component:
    return rx.box(
        rx.heading("🤖 AI 智能分析", size="md", margin_bottom="1rem"),

        rx.cond(
            AppState.ai_generating,
            rx.hstack(
                rx.spinner(),
                rx.text("AI正在分析中..."),
                spacing="0.5rem",
            ),
            rx.cond(
                AppState.ai_report != "",
                rx.box(
                    rx.markdown(AppState.ai_report),
                    border_left="4px solid",
                    border_color="blue.500",
                    padding_left="1rem",
                    bg="blue.50",
                    padding="1rem",
                    border_radius="md",
                ),
                rx.box(
                    rx.text("点击下方按钮生成AI分析报告"),
                    color="gray.500",
                ),
            ),
        ),

        rx.button(
            "生成AI报告",
            on_click=AppState.generate_ai_report,
            color_scheme="blue",
            margin_top="1rem",
        ),

        padding="1.5rem",
        bg="white",
        border_radius="md",
        shadow="md",
        margin_top="1rem",
    )

# 在 page 函数末尾添加 ai_report_section()
```

**Step 3: Commit**

```bash
git add pixiu/state.py pixiu/pages/backtest.py
git commit -m "feat: integrate AI analysis into backtest results page"
```

---

## Phase 6: 测试与打包

### Task 18: 添加集成测试

**Files:**

- Create: `tests/test_integration.py`

**Step 1: 创建集成测试**

```python
# tests/test_integration.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from pixiu.services.database import Database
from pixiu.services.backtest_service import BacktestEngine, BacktestConfig
from pixiu.strategies.trend_strength import TrendStrengthStrategy
from pixiu.strategies import get_all_strategies

def test_full_backtest_workflow():
    dates = pd.date_range('2024-01-01', periods=100)
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 2)

    df = pd.DataFrame({
        'trade_date': dates,
        'close': prices,
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'volume': np.ones(100) * 1000000,
    })

    strategy = TrendStrengthStrategy()
    df_with_signals = strategy.generate_signals(df)

    assert 'signal' in df_with_signals.columns

    engine = BacktestEngine(BacktestConfig(initial_capital=100000))
    result = engine.run(df_with_signals)

    assert result.total_return is not None
    assert result.max_drawdown <= 0

def test_all_strategies_run():
    strategies = get_all_strategies()

    dates = pd.date_range('2024-01-01', periods=100)
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 2)

    df = pd.DataFrame({
        'trade_date': dates,
        'close': prices,
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'volume': np.ones(100) * 1000000,
    })

    for strategy in strategies:
        result_df = strategy.generate_signals(df.copy())
        assert 'signal' in result_df.columns, f"{strategy.name} failed"
```

**Step 2: 运行集成测试**

Run: `pytest tests/test_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for full workflow"
```

---

### Task 19: 运行全部测试

**Step 1: 运行全部测试**

Run: `pytest tests/ -v --cov=pixiu`
Expected: All tests pass

**Step 2: 生成覆盖率报告**

Run: `pytest --cov=pixiu --cov-report=html`
Expected: Coverage report generated

---

### Task 20: 项目打包

**Files:**

- Update: `rxconfig.py`

**Step 1: 更新配置**

```python
import reflex as rx

config = rx.Config(
    app_name="pixiu",
    title="Pixiu 量化分析",
    description="A股/港股/美股量化分析软件",
    frontend_port=3000,
    backend_port=8000,
)
```

**Step 2: 构建生产版本**

Run: `reflex export`
Expected: Build completed

**Step 3: Commit**

```bash
git add .
git commit -m "chore: prepare for production build"
```

---

## 最终验证清单

- [ ] 所有测试通过
- [ ] Reflex开发服务器正常启动
- [ ] 股票搜索功能正常
- [ ] 策略选择和回测功能正常
- [ ] AI报告生成功能正常（需API Key）
- [ ] 设置页面保存功能正常
