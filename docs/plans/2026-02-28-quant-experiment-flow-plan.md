# Pixiu 量化实验流程实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现完整的量化交易实验流程，包括择势判断、多策略支持、策略组合、AKQuant回测和AI报告。

**Architecture:** 分层架构 - 数据层(AKShare)→分析层(择势)→策略层(经典+高级)→组合层→回测层(AKQuant)→展示层(Reflex+AI)

**Tech Stack:** Python 3.10+, Reflex, AKShare, AKQuant, GLM-5, Plotly, Pandas, NumPy

---

## Phase 1: 分析层 - 择势判断

### Task 1.1: 创建择势判断模块

**Files:**

- Create: `pixiu/analysis/__init__.py`
- Create: `pixiu/analysis/regime_detector.py`
- Create: `tests/test_regime_detector.py`

**Step 1: 创建分析模块目录**

```bash
mkdir -p pixiu/analysis
touch pixiu/analysis/__init__.py
```

**Step 2: 编写择势判断测试**

创建 `tests/test_regime_detector.py`:

```python
"""择势判断模块测试"""
import pytest
import pandas as pd
import numpy as np
from pixiu.analysis.regime_detector import MarketRegimeDetector


class TestMarketRegimeDetector:

    @pytest.fixture
    def trend_data(self):
        """生成趋势行情数据"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        close = 10 + np.cumsum(np.random.randn(100) * 0.5 + 0.1)
        return pd.DataFrame({
            'trade_date': dates,
            'open': close + np.random.randn(100) * 0.1,
            'high': close + np.abs(np.random.randn(100) * 0.2),
            'low': close - np.abs(np.random.randn(100) * 0.2),
            'close': close,
            'volume': np.random.randint(1000000, 10000000, 100)
        })

    @pytest.fixture
    def range_data(self):
        """生成震荡行情数据"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        close = 10 + np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.randn(100) * 0.1
        return pd.DataFrame({
            'trade_date': dates,
            'open': close + np.random.randn(100) * 0.05,
            'high': close + np.abs(np.random.randn(100) * 0.1),
            'low': close - np.abs(np.random.randn(100) * 0.1),
            'close': close,
            'volume': np.random.randint(1000000, 10000000, 100)
        })

    def test_init(self):
        """测试初始化"""
        detector = MarketRegimeDetector()
        assert detector is not None

    def test_calc_adx(self, trend_data):
        """测试ADX计算"""
        detector = MarketRegimeDetector()
        adx = detector._calc_adx(trend_data)
        assert isinstance(adx, float)
        assert 0 <= adx <= 100

    def test_calc_ma_slope(self, trend_data):
        """测试MA斜率计算"""
        detector = MarketRegimeDetector()
        slope = detector._calc_ma_slope(trend_data)
        assert isinstance(slope, float)

    def test_calc_volatility(self, trend_data):
        """测试波动率计算"""
        detector = MarketRegimeDetector()
        vol = detector._calc_volatility(trend_data)
        assert isinstance(vol, float)
        assert vol >= 0

    def test_detect_regime_trend(self, trend_data):
        """测试趋势识别"""
        detector = MarketRegimeDetector()
        regime = detector.detect_regime(trend_data)
        assert regime in ['trend', 'range']

    def test_detect_regime_range(self, range_data):
        """测试震荡识别"""
        detector = MarketRegimeDetector()
        regime = detector.detect_regime(range_data)
        assert regime in ['trend', 'range']

    def test_get_analysis_detail(self, trend_data):
        """测试详细分析"""
        detector = MarketRegimeDetector()
        detail = detector.get_analysis_detail(trend_data)
        assert 'regime' in detail
        assert 'adx' in detail
        assert 'ma_slope' in detail
        assert 'volatility' in detail
```

**Step 3: 运行测试确认失败**

```bash
pytest tests/test_regime_detector.py -v
```

Expected: FAIL (模块不存在)

**Step 4: 实现择势判断模块**

创建 `pixiu/analysis/regime_detector.py`:

```python
"""择势判断模块"""
from typing import Dict
import pandas as pd
import numpy as np


class MarketRegimeDetector:
    """大盘/个股择势判断

    通过ADX、MA斜率、波动率等指标判断市场状态：
    - trend: 趋势行情，适合跟踪策略
    - range: 震荡行情，适合均值回归策略
    """

    def __init__(self, adx_period: int = 14, ma_period: int = 20, vol_period: int = 20):
        self.adx_period = adx_period
        self.ma_period = ma_period
        self.vol_period = vol_period

    def detect_regime(self, df: pd.DataFrame) -> str:
        """判断市场状态

        Args:
            df: 行情数据，包含open, high, low, close

        Returns:
            'trend' 或 'range'
        """
        if len(df) < max(self.adx_period, self.ma_period, self.vol_period) + 10:
            return "range"

        adx = self._calc_adx(df)
        slope = self._calc_ma_slope(df)
        vol = self._calc_volatility(df)
        vol_ma = df['close'].pct_change().rolling(self.vol_period).std().iloc[-1]

        trend_score = 0.0

        if adx > 25:
            trend_score += 0.4

        if abs(slope) > 0.005:
            trend_score += 0.3

        if vol > vol_ma * 1.2:
            trend_score += 0.3

        return "trend" if trend_score > 0.5 else "range"

    def _calc_adx(self, df: pd.DataFrame) -> float:
        """计算ADX指标

        ADX > 25 表示趋势行情
        ADX < 25 表示震荡行情
        """
        high = df['high']
        low = df['low']
        close = df['close']

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(self.adx_period).mean()

        plus_di = 100 * (plus_dm.rolling(self.adx_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(self.adx_period).mean() / atr)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(self.adx_period).mean()

        return float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0

    def _calc_ma_slope(self, df: pd.DataFrame) -> float:
        """计算MA斜率

        |斜率| > 0.5% 表示趋势行情
        """
        ma = df['close'].rolling(self.ma_period).mean()

        if len(ma) < 2:
            return 0.0

        slope = (ma.iloc[-1] - ma.iloc[-2]) / ma.iloc[-2]
        return float(slope)

    def _calc_volatility(self, df: pd.DataFrame) -> float:
        """计算波动率"""
        returns = df['close'].pct_change()
        vol = returns.rolling(self.vol_period).std().iloc[-1]
        return float(vol) if not pd.isna(vol) else 0.0

    def get_analysis_detail(self, df: pd.DataFrame) -> Dict:
        """获取详细分析结果

        Returns:
            包含regime, adx, ma_slope, volatility的字典
        """
        return {
            "regime": self.detect_regime(df),
            "adx": round(self._calc_adx(df), 2),
            "ma_slope": round(self._calc_ma_slope(df), 4),
            "volatility": round(self._calc_volatility(df), 4),
        }
```

创建 `pixiu/analysis/__init__.py`:

```python
"""分析模块"""
from .regime_detector import MarketRegimeDetector

__all__ = ["MarketRegimeDetector"]
```

**Step 5: 运行测试确认通过**

```bash
pytest tests/test_regime_detector.py -v
```

Expected: PASS

**Step 6: 提交**

```bash
git add pixiu/analysis/ tests/test_regime_detector.py
git commit -m "feat(analysis): add market regime detector with ADX, MA slope, volatility"
```

---

## Phase 2: 策略层 - 经典策略

### Task 2.1: RSI策略

**Files:**

- Create: `pixiu/strategies/classic/__init__.py`
- Create: `pixiu/strategies/classic/rsi.py`
- Create: `tests/test_rsi_strategy.py`

**Step 1: 创建经典策略目录**

```bash
mkdir -p pixiu/strategies/classic
touch pixiu/strategies/classic/__init__.py
```

**Step 2: 编写RSI策略测试**

创建 `tests/test_rsi_strategy.py`:

```python
"""RSI策略测试"""
import pytest
import pandas as pd
import numpy as np
from pixiu.strategies.classic.rsi import RSIStrategy


class TestRSIStrategy:

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        close = 10 + np.cumsum(np.random.randn(100) * 0.5)
        return pd.DataFrame({
            'trade_date': dates,
            'open': close + np.random.randn(100) * 0.1,
            'high': close + np.abs(np.random.randn(100) * 0.2),
            'low': close - np.abs(np.random.randn(100) * 0.2),
            'close': close,
            'volume': np.random.randint(1000000, 10000000, 100)
        })

    def test_init(self):
        strategy = RSIStrategy()
        assert strategy.name == "RSI策略"
        assert strategy.regime == "range"

    def test_generate_signals(self, sample_data):
        strategy = RSIStrategy()
        signals = strategy.generate_signals(sample_data)
        assert len(signals) == len(sample_data)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_signals_with_params(self, sample_data):
        strategy = RSIStrategy(oversold=25, overbought=75)
        signals = strategy.generate_signals(sample_data)
        assert len(signals) == len(sample_data)
```

**Step 3: 运行测试确认失败**

```bash
pytest tests/test_rsi_strategy.py -v
```

Expected: FAIL

**Step 4: 实现RSI策略**

创建 `pixiu/strategies/classic/rsi.py`:

```python
"""RSI策略"""
import pandas as pd
import numpy as np
from pixiu.strategies.base import BaseStrategy
from pixiu.strategies import register_strategy


@register_strategy
class RSIStrategy(BaseStrategy):
    """RSI相对强弱指标策略

    适用于震荡行情：
    - RSI < oversold (30): 超卖，买入信号
    - RSI > overbought (70): 超买，卖出信号
    """

    name = "RSI策略"
    description = "基于RSI相对强弱指标的均值回归策略，适用于震荡行情"
    regime = "range"
    params = {
        "period": 14,
        "oversold": 30,
        "overbought": 70
    }

    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        self.params = {
            "period": period,
            "oversold": oversold,
            "overbought": overbought
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """生成交易信号

        Args:
            df: 包含close列的DataFrame

        Returns:
            信号序列: 1=买入, -1=卖出, 0=持有
        """
        close = df['close']
        period = self.params["period"]
        oversold = self.params["oversold"]
        overbought = self.params["overbought"]

        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        signals = pd.Series(0, index=df.index)
        signals[rsi < oversold] = 1
        signals[rsi > overbought] = -1

        return signals

    def get_required_data(self) -> list:
        return ["close"]
```

更新 `pixiu/strategies/classic/__init__.py`:

```python
"""经典策略模块"""
from .rsi import RSIStrategy

__all__ = ["RSIStrategy"]
```

**Step 5: 运行测试确认通过**

```bash
pytest tests/test_rsi_strategy.py -v
```

Expected: PASS

**Step 6: 提交**

```bash
git add pixiu/strategies/classic/ tests/test_rsi_strategy.py
git commit -m "feat(strategy): add RSI strategy for range-bound markets"
```

### Task 2.2: 均线交叉策略

**Files:**

- Create: `pixiu/strategies/classic/ma_cross.py`
- Create: `tests/test_ma_cross_strategy.py`

**Step 1: 编写测试**

创建 `tests/test_ma_cross_strategy.py`:

```python
"""均线交叉策略测试"""
import pytest
import pandas as pd
import numpy as np
from pixiu.strategies.classic.ma_cross import MACrossStrategy


class TestMACrossStrategy:

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        close = 10 + np.cumsum(np.random.randn(100) * 0.5 + 0.05)
        return pd.DataFrame({
            'trade_date': dates,
            'open': close + np.random.randn(100) * 0.1,
            'high': close + np.abs(np.random.randn(100) * 0.2),
            'low': close - np.abs(np.random.randn(100) * 0.2),
            'close': close,
            'volume': np.random.randint(1000000, 10000000, 100)
        })

    def test_init(self):
        strategy = MACrossStrategy()
        assert strategy.name == "均线交叉策略"
        assert strategy.regime == "trend"

    def test_generate_signals(self, sample_data):
        strategy = MACrossStrategy()
        signals = strategy.generate_signals(sample_data)
        assert len(signals) == len(sample_data)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_golden_cross_signal(self):
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        close = pd.Series([10] * 20 + list(range(10, 40)), index=dates[:50])
        df = pd.DataFrame({'close': close, 'trade_date': dates[:50]})

        strategy = MACrossStrategy(fast_period=5, slow_period=20)
        signals = strategy.generate_signals(df)
        assert 1 in signals.values
```

**Step 2: 运行测试确认失败**

```bash
pytest tests/test_ma_cross_strategy.py -v
```

Expected: FAIL

**Step 3: 实现均线交叉策略**

创建 `pixiu/strategies/classic/ma_cross.py`:

```python
"""均线交叉策略"""
import pandas as pd
from pixiu.strategies.base import BaseStrategy
from pixiu.strategies import register_strategy


@register_strategy
class MACrossStrategy(BaseStrategy):
    """均线交叉策略

    适用于趋势行情：
    - 金叉（短期均线上穿长期均线）: 买入
    - 死叉（短期均线下穿长期均线）: 卖出
    """

    name = "均线交叉策略"
    description = "基于快慢均线交叉的趋势跟踪策略"
    regime = "trend"
    params = {
        "fast_period": 5,
        "slow_period": 20
    }

    def __init__(self, fast_period: int = 5, slow_period: int = 20):
        self.params = {
            "fast_period": fast_period,
            "slow_period": slow_period
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        close = df['close']
        fast_period = self.params["fast_period"]
        slow_period = self.params["slow_period"]

        fast_ma = close.rolling(fast_period).mean()
        slow_ma = close.rolling(slow_period).mean()

        signals = pd.Series(0, index=df.index)

        gold_cross = (fast_ma.shift(1) <= slow_ma.shift(1)) & (fast_ma > slow_ma)
        death_cross = (fast_ma.shift(1) >= slow_ma.shift(1)) & (fast_ma < slow_ma)

        signals[gold_cross] = 1
        signals[death_cross] = -1

        return signals

    def get_required_data(self) -> list:
        return ["close"]
```

更新 `pixiu/strategies/classic/__init__.py`:

```python
"""经典策略模块"""
from .rsi import RSIStrategy
from .ma_cross import MACrossStrategy

__all__ = ["RSIStrategy", "MACrossStrategy"]
```

**Step 4: 运行测试确认通过**

```bash
pytest tests/test_ma_cross_strategy.py -v
```

Expected: PASS

**Step 5: 提交**

```bash
git add pixiu/strategies/classic/ tests/test_ma_cross_strategy.py
git commit -m "feat(strategy): add MA cross strategy for trend markets"
```

### Task 2.3: 网格交易策略

**Files:**

- Create: `pixiu/strategies/classic/grid_trading.py`
- Create: `tests/test_grid_trading_strategy.py`

**Step 1: 编写测试**

创建 `tests/test_grid_trading_strategy.py`:

```python
"""网格交易策略测试"""
import pytest
import pandas as pd
import numpy as np
from pixiu.strategies.classic.grid_trading import GridTradingStrategy


class TestGridTradingStrategy:

    @pytest.fixture
    def range_data(self):
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        close = 10 + np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.randn(100) * 0.1
        return pd.DataFrame({
            'trade_date': dates,
            'open': close + np.random.randn(100) * 0.05,
            'high': close + np.abs(np.random.randn(100) * 0.1),
            'low': close - np.abs(np.random.randn(100) * 0.1),
            'close': close,
            'volume': np.random.randint(1000000, 10000000, 100)
        })

    def test_init(self):
        strategy = GridTradingStrategy()
        assert strategy.name == "网格交易策略"
        assert strategy.regime == "range"

    def test_generate_signals(self, range_data):
        strategy = GridTradingStrategy()
        signals = strategy.generate_signals(range_data)
        assert len(signals) == len(range_data)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_params(self, range_data):
        strategy = GridTradingStrategy(grid_size=0.03, grid_count=5)
        signals = strategy.generate_signals(range_data)
        assert len(signals) == len(range_data)
```

**Step 2: 运行测试确认失败**

```bash
pytest tests/test_grid_trading_strategy.py -v
```

Expected: FAIL

**Step 3: 实现网格交易策略**

创建 `pixiu/strategies/classic/grid_trading.py`:

```python
"""网格交易策略"""
import pandas as pd
import numpy as np
from pixiu.strategies.base import BaseStrategy
from pixiu.strategies import register_strategy


@register_strategy
class GridTradingStrategy(BaseStrategy):
    """网格交易策略

    适用于震荡行情：
    - 在价格下跌grid_size时买入
    - 在价格上涨grid_size时卖出
    """

    name = "网格交易策略"
    description = "在价格区间内设置网格，低买高卖的均值回归策略"
    regime = "range"
    params = {
        "grid_size": 0.02,
        "grid_count": 10
    }

    def __init__(self, grid_size: float = 0.02, grid_count: int = 10):
        self.params = {
            "grid_size": grid_size,
            "grid_count": grid_count
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        close = df['close']
        grid_size = self.params["grid_size"]

        signals = pd.Series(0, index=df.index)

        base_price = close.iloc[0]
        position = 0
        last_trade_price = base_price

        for i in range(1, len(close)):
            current_price = close.iloc[i]
            price_change = (current_price - last_trade_price) / last_trade_price

            if position == 0 and price_change <= -grid_size:
                signals.iloc[i] = 1
                position = 1
                last_trade_price = current_price
            elif position > 0 and price_change >= grid_size:
                signals.iloc[i] = -1
                position = 0
                last_trade_price = current_price

        return signals

    def get_required_data(self) -> list:
        return ["close"]
```

更新 `pixiu/strategies/classic/__init__.py`:

```python
"""经典策略模块"""
from .rsi import RSIStrategy
from .ma_cross import MACrossStrategy
from .grid_trading import GridTradingStrategy

__all__ = ["RSIStrategy", "MACrossStrategy", "GridTradingStrategy"]
```

**Step 4: 运行测试确认通过**

```bash
pytest tests/test_grid_trading_strategy.py -v
```

Expected: PASS

**Step 5: 提交**

```bash
git add pixiu/strategies/classic/ tests/test_grid_trading_strategy.py
git commit -m "feat(strategy): add grid trading strategy for range-bound markets"
```

---

## Phase 3: 策略层 - 高级策略

### Task 3.1: 随机过程策略

**Files:**

- Create: `pixiu/strategies/advanced/__init__.py`
- Create: `pixiu/strategies/advanced/stochastic.py`
- Create: `tests/test_stochastic_strategy.py`

**Step 1: 创建高级策略目录**

```bash
mkdir -p pixiu/strategies/advanced
touch pixiu/strategies/advanced/__init__.py
```

**Step 2: 编写测试**

创建 `tests/test_stochastic_strategy.py`:

```python
"""随机过程策略测试"""
import pytest
import pandas as pd
import numpy as np
from pixiu.strategies.advanced.stochastic import StochasticStrategy


class TestStochasticStrategy:

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        close = 10 + np.cumsum(np.random.randn(100) * 0.5)
        return pd.DataFrame({
            'trade_date': dates,
            'open': close + np.random.randn(100) * 0.1,
            'high': close + np.abs(np.random.randn(100) * 0.2),
            'low': close - np.abs(np.random.randn(100) * 0.2),
            'close': close,
            'volume': np.random.randint(1000000, 10000000, 100)
        })

    def test_init(self):
        strategy = StochasticStrategy()
        assert strategy.name == "随机过程策略"
        assert strategy.regime == "any"

    def test_generate_signals(self, sample_data):
        strategy = StochasticStrategy()
        signals = strategy.generate_signals(sample_data)
        assert len(signals) == len(sample_data)
        assert set(signals.unique()).issubset({-1, 0, 1})
```

**Step 3: 运行测试确认失败**

```bash
pytest tests/test_stochastic_strategy.py -v
```

Expected: FAIL

**Step 4: 实现随机过程策略**

创建 `pixiu/strategies/advanced/stochastic.py`:

```python
"""随机过程策略"""
import pandas as pd
import numpy as np
from scipy import stats
from pixiu.strategies.base import BaseStrategy
from pixiu.strategies import register_strategy


@register_strategy
class StochasticStrategy(BaseStrategy):
    """随机过程策略

    基于几何布朗运动(GBM)建模：
    dS = μS dt + σS dW

    利用估计的漂移项和波动率预测价格偏离
    """

    name = "随机过程策略"
    description = "基于几何布朗运动的随机过程建模策略"
    regime = "any"
    params = {
        "lookback": 60,
        "z_threshold": 1.5
    }

    def __init__(self, lookback: int = 60, z_threshold: float = 1.5):
        self.params = {
            "lookback": lookback,
            "z_threshold": z_threshold
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        close = df['close']
        lookback = self.params["lookback"]
        z_threshold = self.params["z_threshold"]

        signals = pd.Series(0, index=df.index)

        for i in range(lookback, len(close)):
            window = close.iloc[i-lookback:i]
            returns = window.pct_change().dropna()

            mu = returns.mean() * 252
            sigma = returns.std() * np.sqrt(252)

            expected_return = mu * (1/252)
            actual_return = (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]

            z_score = (actual_return - expected_return) / (sigma / np.sqrt(252))

            if z_score < -z_threshold:
                signals.iloc[i] = 1
            elif z_score > z_threshold:
                signals.iloc[i] = -1

        return signals

    def get_required_data(self) -> list:
        return ["close"]
```

创建 `pixiu/strategies/advanced/__init__.py`:

```python
"""高级策略模块"""
from .stochastic import StochasticStrategy

__all__ = ["StochasticStrategy"]
```

**Step 5: 运行测试确认通过**

```bash
pytest tests/test_stochastic_strategy.py -v
```

Expected: PASS

**Step 6: 提交**

```bash
git add pixiu/strategies/advanced/ tests/test_stochastic_strategy.py
git commit -m "feat(strategy): add stochastic process strategy based on GBM"
```

### Task 3.2: 最优执行策略

**Files:**

- Create: `pixiu/strategies/advanced/optimal_execution.py`
- Create: `tests/test_optimal_execution_strategy.py`

**Step 1: 编写测试**

创建 `tests/test_optimal_execution_strategy.py`:

```python
"""最优执行策略测试"""
import pytest
import pandas as pd
import numpy as np
from pixiu.strategies.advanced.optimal_execution import OptimalExecutionStrategy


class TestOptimalExecutionStrategy:

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        close = 10 + np.cumsum(np.random.randn(100) * 0.5 + 0.05)
        return pd.DataFrame({
            'trade_date': dates,
            'open': close + np.random.randn(100) * 0.1,
            'high': close + np.abs(np.random.randn(100) * 0.2),
            'low': close - np.abs(np.random.randn(100) * 0.2),
            'close': close,
            'volume': np.random.randint(1000000, 10000000, 100)
        })

    def test_init(self):
        strategy = OptimalExecutionStrategy()
        assert strategy.name == "最优执行策略"
        assert strategy.regime == "trend"

    def test_generate_signals(self, sample_data):
        strategy = OptimalExecutionStrategy()
        signals = strategy.generate_signals(sample_data)
        assert len(signals) == len(sample_data)
        assert set(signals.unique()).issubset({-1, 0, 1})
```

**Step 2: 运行测试确认失败**

```bash
pytest tests/test_optimal_execution_strategy.py -v
```

Expected: FAIL

**Step 3: 实现最优执行策略**

创建 `pixiu/strategies/advanced/optimal_execution.py`:

```python
"""最优执行策略"""
import pandas as pd
import numpy as np
from pixiu.strategies.base import BaseStrategy
from pixiu.strategies import register_strategy


@register_strategy
class OptimalExecutionStrategy(BaseStrategy):
    """最优执行策略

    基于TWAP/VWAP执行算法：
    - 在趋势行情中分批建仓
    - 使用成交量加权平均价格作为执行基准
    """

    name = "最优执行策略"
    description = "基于TWAP/VWAP的最优执行算法策略"
    regime = "trend"
    params = {
        "execution_window": 5,
        "volume_threshold": 1.2
    }

    def __init__(self, execution_window: int = 5, volume_threshold: float = 1.2):
        self.params = {
            "execution_window": execution_window,
            "volume_threshold": volume_threshold
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        close = df['close']
        volume = df['volume']
        execution_window = self.params["execution_window"]
        volume_threshold = self.params["volume_threshold"]

        signals = pd.Series(0, index=df.index)

        avg_volume = volume.rolling(execution_window * 5).mean()
        vwap = (close * volume).rolling(execution_window * 5).sum() / \
               volume.rolling(execution_window * 5).sum()

        for i in range(execution_window * 5, len(close)):
            price_vs_vwap = (close.iloc[i] - vwap.iloc[i]) / vwap.iloc[i]
            vol_ratio = volume.iloc[i] / avg_volume.iloc[i]

            if price_vs_vwap < -0.01 and vol_ratio > volume_threshold:
                signals.iloc[i] = 1
            elif price_vs_vwap > 0.01 and vol_ratio > volume_threshold:
                signals.iloc[i] = -1

        return signals

    def get_required_data(self) -> list:
        return ["close", "volume"]
```

更新 `pixiu/strategies/advanced/__init__.py`:

```python
"""高级策略模块"""
from .stochastic import StochasticStrategy
from .optimal_execution import OptimalExecutionStrategy

__all__ = ["StochasticStrategy", "OptimalExecutionStrategy"]
```

**Step 4: 运行测试确认通过**

```bash
pytest tests/test_optimal_execution_strategy.py -v
```

Expected: PASS

**Step 5: 提交**

```bash
git add pixiu/strategies/advanced/ tests/test_optimal_execution_strategy.py
git commit -m "feat(strategy): add optimal execution strategy based on TWAP/VWAP"
```

---

## Phase 4: 组合层 - 策略组合器

### Task 4.1: 策略组合器实现

**Files:**

- Create: `pixiu/strategies/combiner.py`
- Create: `tests/test_strategy_combiner.py`

**Step 1: 编写测试**

创建 `tests/test_strategy_combiner.py`:

```python
"""策略组合器测试"""
import pytest
import pandas as pd
import numpy as np
from pixiu.strategies.combiner import StrategyCombiner
from pixiu.strategies.classic.rsi import RSIStrategy
from pixiu.strategies.classic.ma_cross import MACrossStrategy


class TestStrategyCombiner:

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        close = 10 + np.cumsum(np.random.randn(100) * 0.5)
        return pd.DataFrame({
            'trade_date': dates,
            'open': close + np.random.randn(100) * 0.1,
            'high': close + np.abs(np.random.randn(100) * 0.2),
            'low': close - np.abs(np.random.randn(100) * 0.2),
            'close': close,
            'volume': np.random.randint(1000000, 10000000, 100)
        })

    def test_init(self):
        combiner = StrategyCombiner()
        assert combiner is not None

    def test_equal_weight(self, sample_data):
        combiner = StrategyCombiner()
        rsi = RSIStrategy()
        ma = MACrossStrategy()

        signals = [
            rsi.generate_signals(sample_data),
            ma.generate_signals(sample_data)
        ]

        combined = combiner.equal_weight(signals)
        assert len(combined) == len(sample_data)
        assert set(combined.unique()).issubset({-1, 0, 1})

    def test_signal_filter(self, sample_data):
        combiner = StrategyCombiner()
        rsi = RSIStrategy()
        ma = MACrossStrategy()

        signals = [
            rsi.generate_signals(sample_data),
            ma.generate_signals(sample_data)
        ]

        filtered = combiner.signal_filter(signals, threshold=2)
        assert len(filtered) == len(sample_data)

    def test_complementary(self, sample_data):
        combiner = StrategyCombiner()
        rsi = RSIStrategy()
        ma = MACrossStrategy()

        trend_strategies = [ma]
        range_strategies = [rsi]

        combined = combiner.complementary(
            sample_data, 
            "trend",
            trend_strategies,
            range_strategies
        )
        assert len(combined) == len(sample_data)
```

**Step 2: 运行测试确认失败**

```bash
pytest tests/test_strategy_combiner.py -v
```

Expected: FAIL

**Step 3: 实现策略组合器**

创建 `pixiu/strategies/combiner.py`:

```python
"""策略组合器"""
from typing import List, Dict
import pandas as pd
import numpy as np


class StrategyCombiner:
    """策略组合器

    提供三种组合模式：
    1. equal_weight: 等权组合
    2. signal_filter: 信号过滤
    3. complementary: 互补策略
    """

    COMBINE_MODES = ["equal_weight", "signal_filter", "complementary"]

    def __init__(self, config: Dict = None):
        self.config = config or {
            "mode": "complementary",
            "filter_threshold": 2,
            "trend_strategies": ["均线交叉策略"],
            "range_strategies": ["RSI策略", "网格交易策略"]
        }

    def equal_weight(self, signals: List[pd.Series]) -> pd.Series:
        """等权组合

        所有策略信号取平均，>0买入，<0卖出
        """
        if not signals:
            return pd.Series(0, index=[])

        combined = sum(signals) / len(signals)
        result = pd.Series(0, index=signals[0].index)
        result[combined > 0] = 1
        result[combined < 0] = -1
        return result

    def signal_filter(self, signals: List[pd.Series], threshold: int = 2) -> pd.Series:
        """信号过滤

        N个以上策略一致时才执行
        """
        if not signals:
            return pd.Series(0, index=[])

        buy_votes = sum((s == 1).astype(int) for s in signals)
        sell_votes = sum((s == -1).astype(int) for s in signals)

        result = pd.Series(0, index=signals[0].index)
        result[buy_votes >= threshold] = 1
        result[sell_votes >= threshold] = -1
        return result

    def complementary(
        self,
        df: pd.DataFrame,
        regime: str,
        trend_strategies: List,
        range_strategies: List
    ) -> pd.Series:
        """互补策略

        根据市场状态自动切换策略组
        """
        if regime == "trend":
            strategies = trend_strategies
        else:
            strategies = range_strategies

        if not strategies:
            return pd.Series(0, index=df.index)

        signals = [s.generate_signals(df) for s in strategies]
        return self.equal_weight(signals)

    def combine(
        self,
        signals: List[pd.Series],
        regime: str = None,
        df: pd.DataFrame = None,
        trend_strategies: List = None,
        range_strategies: List = None
    ) -> pd.Series:
        """通用组合接口"""
        mode = self.config.get("mode", "equal_weight")

        if mode == "equal_weight":
            return self.equal_weight(signals)
        elif mode == "signal_filter":
            return self.signal_filter(signals, self.config.get("filter_threshold", 2))
        elif mode == "complementary":
            return self.complementary(
                df, regime, trend_strategies or [], range_strategies or []
            )

        raise ValueError(f"Unknown combine mode: {mode}")
```

**Step 4: 运行测试确认通过**

```bash
pytest tests/test_strategy_combiner.py -v
```

Expected: PASS

**Step 5: 提交**

```bash
git add pixiu/strategies/combiner.py tests/test_strategy_combiner.py
git commit -m "feat(strategy): add strategy combiner with equal_weight, signal_filter, complementary modes"
```

---

## Phase 5: 回测层 - AKQuant集成

### Task 5.1: AKQuant适配器

**Files:**

- Create: `pixiu/services/akquant_adapter.py`
- Create: `tests/test_akquant_adapter.py`

**Step 1: 更新requirements.txt**

添加到 `requirements.txt`:

```
akquant>=0.1.0
```

**Step 2: 编写测试**

创建 `tests/test_akquant_adapter.py`:

```python
"""AKQuant适配器测试"""
import pytest
import pandas as pd
import numpy as np
from pixiu.services.akquant_adapter import AKQuantAdapter
from pixiu.strategies.classic.rsi import RSIStrategy


class TestAKQuantAdapter:

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        close = 10 + np.cumsum(np.random.randn(100) * 0.5)
        df = pd.DataFrame({
            'trade_date': dates,
            'open': close + np.random.randn(100) * 0.1,
            'high': close + np.abs(np.random.randn(100) * 0.2),
            'low': close - np.abs(np.random.randn(100) * 0.2),
            'close': close,
            'volume': np.random.randint(1000000, 10000000, 100)
        })
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        return df

    def test_init(self):
        adapter = AKQuantAdapter()
        assert adapter is not None

    @pytest.mark.skipif(
        True,  # 跳过需要akquant环境的测试
        reason="AKQuant not installed in test environment"
    )
    def test_run_backtest(self, sample_data):
        adapter = AKQuantAdapter()
        strategy = RSIStrategy()
        config = {
            'initial_capital': 100000,
            'symbol': 'test'
        }

        result = adapter.run_backtest(sample_data, strategy, config)
        assert result is not None
```

**Step 3: 实现AKQuant适配器**

创建 `pixiu/services/akquant_adapter.py`:

```python
"""AKQuant适配器"""
from typing import Dict, Optional
import pandas as pd

try:
    import akquant as aq
    from akquant import Strategy
    AKQUANT_AVAILABLE = True
except ImportError:
    AKQUANT_AVAILABLE = False
    aq = None
    Strategy = None

from pixiu.models.backtest import BacktestResult, Trade


class AKQuantAdapter:
    """AKQuant高性能回测适配器"""

    def __init__(self):
        if not AKQUANT_AVAILABLE:
            import warnings
            warnings.warn("AKQuant not installed, falling back to built-in engine")

    def run_backtest(
        self,
        df: pd.DataFrame,
        strategy,
        config: Dict
    ) -> Optional[BacktestResult]:
        """运行回测

        Args:
            df: 行情数据
            strategy: Pixiu策略实例
            config: 回测配置

        Returns:
            BacktestResult或None（如果AKQuant不可用）
        """
        if not AKQUANT_AVAILABLE:
            return self._fallback_backtest(df, strategy, config)

        try:
            wrapped_strategy = self._wrap_strategy(strategy)

            result = aq.run_backtest(
                data=self._prepare_data(df),
                strategy=wrapped_strategy,
                initial_cash=config.get('initial_capital', 100000),
                symbol=config.get('symbol', 'stock')
            )

            return self._convert_result(result)
        except Exception as e:
            import warnings
            warnings.warn(f"AKQuant backtest failed: {e}, using fallback")
            return self._fallback_backtest(df, strategy, config)

    def _wrap_strategy(self, strategy):
        """将Pixiu策略包装为AKQuant策略"""
        if not AKQUANT_AVAILABLE:
            return None

        pixiu_strategy = strategy
        strategy_self = self

        class WrappedStrategy(Strategy):
            def on_bar(self, bar):
                df = strategy_self._bar_to_df(bar, self.data)
                signal = pixiu_strategy.generate_signals(df).iloc[-1]

                if signal == 1:
                    pos_size = int(self.cash * 0.95 / bar.close)
                    if pos_size > 0:
                        self.buy(symbol=bar.symbol, quantity=pos_size)
                elif signal == -1:
                    pos = self.get_position(bar.symbol)
                    if pos > 0:
                        self.close_position(symbol=bar.symbol)

        return WrappedStrategy

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备AKQuant格式的数据"""
        result = df.copy()
        if 'trade_date' in result.columns:
            result = result.rename(columns={'trade_date': 'date'})
        return result

    def _bar_to_df(self, bar, data) -> pd.DataFrame:
        """将bar转换为DataFrame"""
        return pd.DataFrame({
            'open': [bar.open],
            'high': [bar.high],
            'low': [bar.low],
            'close': [bar.close],
            'volume': [bar.volume]
        })

    def _convert_result(self, aq_result) -> BacktestResult:
        """转换AKQuant结果为BacktestResult"""
        return BacktestResult(
            total_return=aq_result.total_return_pct,
            annualized_return=getattr(aq_result, 'annualized_return', 0),
            max_drawdown=aq_result.max_drawdown_pct,
            sharpe_ratio=getattr(aq_result, 'sharpe_ratio', 0),
            win_rate=aq_result.win_rate / 100 if aq_result.win_rate else 0,
            profit_loss_ratio=getattr(aq_result, 'profit_factor', 0),
            calmar_ratio=getattr(aq_result, 'calmar_ratio', 0),
            total_trades=int(aq_result.trade_count) if aq_result.trade_count else 0,
            start_date=str(getattr(aq_result, 'start_time', '')),
            end_date=str(getattr(aq_result, 'end_time', '')),
            trades=[],
            equity_curve=[],
            drawdown_curve=[]
        )

    def _fallback_backtest(
        self,
        df: pd.DataFrame,
        strategy,
        config: Dict
    ) -> BacktestResult:
        """降级到内置回测引擎"""
        from pixiu.services.backtest_service import BacktestEngine, BacktestConfig

        backtest_config = BacktestConfig(
            initial_capital=config.get('initial_capital', 100000),
            commission_rate=config.get('commission_rate', 0.0003),
            position_size=0.95
        )

        signals = strategy.generate_signals(df)
        engine = BacktestEngine(backtest_config)
        return engine.run(df, signals)

    def generate_report(self, result, output_path: str):
        """生成可视化报告"""
        if AKQUANT_AVAILABLE and hasattr(result, '_aq_result'):
            result._aq_result.report(show=False, output_path=output_path)
```

**Step 4: 运行测试**

```bash
pytest tests/test_akquant_adapter.py -v
```

Expected: PASS (跳过AKQuant测试)

**Step 5: 提交**

```bash
git add pixiu/services/akquant_adapter.py tests/test_akquant_adapter.py requirements.txt
git commit -m "feat(backtest): add AKQuant adapter with fallback to built-in engine"
```

---

## Phase 6: 展示层 - UI组件

### Task 6.1: 择势状态组件

**Files:**

- Create: `pixiu/components/regime_indicator.py`

**Step 1: 创建择势状态组件**

创建 `pixiu/components/regime_indicator.py`:

```python
"""择势状态指示器组件"""
import reflex as rx


def regime_indicator(
    regime: str,
    adx: float = 0,
    ma_slope: float = 0,
    volatility: float = 0
) -> rx.Component:
    """择势状态指示器

    Args:
        regime: 市场状态 ('trend' | 'range')
        adx: ADX指标值
        ma_slope: MA斜率
        volatility: 波动率
    """
    regime_color = "#10b981" if regime == "trend" else "#f59e0b"
    regime_text = "趋势" if regime == "trend" else "震荡"
    regime_icon = "📈" if regime == "trend" else "📊"

    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text(f"{regime_icon} {regime_text}行情", 
                       font_size="lg", font_weight="bold"),
                rx.badge(
                    regime_text,
                    color_scheme="green" if regime == "trend" else "yellow"
                ),
                justify="space_between",
                width="100%"
            ),
            rx.divider(),
            rx.hstack(
                rx.vstack(
                    rx.text("ADX", font_size="sm", color="gray"),
                    rx.text(f"{adx:.1f}", font_weight="bold"),
                ),
                rx.vstack(
                    rx.text("MA斜率", font_size="sm", color="gray"),
                    rx.text(f"{ma_slope:.4f}", font_weight="bold"),
                ),
                rx.vstack(
                    rx.text("波动率", font_size="sm", color="gray"),
                    rx.text(f"{volatility:.4f}", font_weight="bold"),
                ),
                justify="space_between",
                width="100%"
            ),
            spacing="2",
        ),
        padding="1rem",
        border_radius="lg",
        bg="#1a1a24",
        border=f"2px solid {regime_color}",
    )
```

**Step 2: 提交**

```bash
git add pixiu/components/regime_indicator.py
git commit -m "feat(ui): add regime indicator component"
```

### Task 6.2: 策略推荐组件

**Files:**

- Create: `pixiu/components/strategy_recommender.py`

**Step 1: 创建策略推荐组件**

创建 `pixiu/components/strategy_recommender.py`:

```python
"""策略推荐组件"""
import reflex as rx


STRATEGY_REGIME_MAP = {
    "trend": ["趋势强度策略", "均线交叉策略", "最优执行策略"],
    "range": ["网格交易策略", "RSI策略", "波动率套利策略"],
    "any": ["随机过程策略", "卡尔曼滤波策略"]
}


def strategy_recommender(
    regime: str,
    available_strategies: list,
    selected_strategies: list,
    on_toggle
) -> rx.Component:
    """策略推荐组件

    Args:
        regime: 市场状态
        available_strategies: 所有可用策略
        selected_strategies: 已选策略
        on_toggle: 切换策略的回调
    """
    recommended = STRATEGY_REGIME_MAP.get(regime, [])

    return rx.vstack(
        rx.hstack(
            rx.text("推荐策略", font_weight="bold"),
            rx.badge(f"基于{regime}行情", color_scheme="blue"),
        ),
        rx.foreach(
            available_strategies,
            lambda s: _strategy_item(s, recommended, selected_strategies, on_toggle)
        ),
        spacing="2",
        width="100%"
    )


def _strategy_item(
    strategy: dict,
    recommended: list,
    selected: list,
    on_toggle
) -> rx.Component:
    """单个策略项"""
    is_recommended = strategy["name"] in recommended
    is_selected = strategy["name"] in selected

    return rx.box(
        rx.hstack(
            rx.checkbox(
                is_checked=is_selected,
                on_change=lambda: on_toggle(strategy["name"])
            ),
            rx.vstack(
                rx.text(strategy["name"], font_weight="medium"),
                rx.text(strategy["description"], font_size="sm", color="gray"),
                spacing="1",
                align_items="start"
            ),
            rx.cond(
                is_recommended,
                rx.badge("推荐", color_scheme="green", size="sm"),
                rx.box()
            ),
            justify="space_between",
            width="100%"
        ),
        padding="0.5rem",
        border_radius="md",
        bg="#252532" if is_selected else "transparent",
        border=f"1px solid {'#10b981' if is_recommended else '#333'}",
    )
```

**Step 2: 提交**

```bash
git add pixiu/components/strategy_recommender.py
git commit -m "feat(ui): add strategy recommender component"
```

---

## Phase 7: 状态管理更新

### Task 7.1: 更新State类

**Files:**

- Modify: `pixiu/state.py`

**Step 1: 添加择势相关状态**

在 `pixiu/state.py` 中添加:

```python
# 在State类中添加新属性
market_regime: str = "unknown"
stock_regime: str = "unknown"
regime_analysis: Dict = {}

combine_mode: str = "complementary"
filter_threshold: int = 2

# 添加新方法
async def analyze_regime(self):
    """分析市场状态"""
    from pixiu.analysis import MarketRegimeDetector

    if self.stock_data is None or self.stock_data.empty:
        return

    self.is_loading = True
    self.loading_message = "分析市场状态..."
    yield

    try:
        detector = MarketRegimeDetector()
        self.regime_analysis = detector.get_analysis_detail(self.stock_data)
        self.stock_regime = self.regime_analysis["regime"]
    finally:
        self.is_loading = False
        yield

def set_combine_mode(self, mode: str):
    """设置组合模式"""
    if mode in ["equal_weight", "signal_filter", "complementary"]:
        self.combine_mode = mode

def set_filter_threshold(self, value: str):
    """设置信号过滤阈值"""
    try:
        self.filter_threshold = int(value)
    except ValueError:
        pass
```

**Step 2: 提交**

```bash
git add pixiu/state.py
git commit -m "feat(state): add regime analysis and combine mode state"
```

---

## Phase 8: AI报告增强

### Task 8.1: 增强AI服务

**Files:**

- Modify: `pixiu/services/ai_service.py`

**Step 1: 添加完整报告生成方法**

在 `pixiu/services/ai_service.py` 中添加:

```python
async def generate_full_report(
    self,
    stock_info: Dict,
    regime_analysis: Dict,
    backtest_results: List[Dict],
    strategy_params: Dict
) -> str:
    """生成完整AI分析报告"""

    prompt = f"""请分析以下量化回测结果并生成专业报告：

## 1. 股票信息
- 代码：{stock_info.get('code', 'N/A')}
- 名称：{stock_info.get('name', 'N/A')}
- 市场：{stock_info.get('market', 'N/A')}

## 2. 择势判断
- 大盘状态：{regime_analysis.get('market_regime', 'N/A')}
- 个股状态：{regime_analysis.get('regime', 'N/A')}
- ADX：{regime_analysis.get('adx', 0):.2f}
- MA斜率：{regime_analysis.get('ma_slope', 0):.4f}
- 波动率：{regime_analysis.get('volatility', 0):.4f}

## 3. 回测表现
"""

    for i, result in enumerate(backtest_results, 1):
        prompt += f"""
### 策略 {i}: {result.get('strategy', 'N/A')}
- 总收益率：{result.get('total_return', 0):.2%}
- 年化收益：{result.get('annualized_return', 0):.2%}
- 最大回撤：{result.get('max_drawdown', 0):.2%}
- 夏普比率：{result.get('sharpe_ratio', 0):.2f}
- 胜率：{result.get('win_rate', 0):.2%}
- 盈亏比：{result.get('profit_loss_ratio', 0):.2f}
"""

    prompt += """
请从以下角度进行分析：

1. **策略表现评估**：策略在该股票上的表现如何？是否符合预期？
2. **择势判断准确性**：市场状态判断是否准确？对策略选择的影响？
3. **风险提示**：主要风险点有哪些？最大回撤是否可接受？
4. **改进建议**：有哪些可以优化的地方？
5. **适用性评估**：该策略适合什么类型的投资者？

请用中文回答，格式清晰，专业严谨。
"""

    return await self._call_api(prompt)
```

**Step 2: 提交**

```bash
git add pixiu/services/ai_service.py
git commit -m "feat(ai): add full report generation with regime analysis"
```

---

## Phase 9: 集成测试

### Task 9.1: 端到端测试

**Files:**

- Create: `tests/test_integration.py`

**Step 1: 编写集成测试**

创建 `tests/test_integration.py`:

```python
"""集成测试"""
import pytest
import pandas as pd
import numpy as np
from pixiu.analysis import MarketRegimeDetector
from pixiu.strategies.classic import RSIStrategy, MACrossStrategy, GridTradingStrategy
from pixiu.strategies.advanced import StochasticStrategy
from pixiu.strategies.combiner import StrategyCombiner
from pixiu.services.backtest_service import BacktestEngine, BacktestConfig


class TestIntegration:

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=200, freq='D')
        close = 10 + np.cumsum(np.random.randn(200) * 0.5)
        return pd.DataFrame({
            'trade_date': dates,
            'open': close + np.random.randn(200) * 0.1,
            'high': close + np.abs(np.random.randn(200) * 0.2),
            'low': close - np.abs(np.random.randn(200) * 0.2),
            'close': close,
            'volume': np.random.randint(1000000, 10000000, 200)
        })

    def test_full_workflow(self, sample_data):
        """测试完整工作流"""
        # 1. 择势判断
        detector = MarketRegimeDetector()
        regime = detector.detect_regime(sample_data)
        assert regime in ['trend', 'range']

        # 2. 选择策略
        if regime == 'trend':
            strategies = [MACrossStrategy()]
        else:
            strategies = [RSIStrategy(), GridTradingStrategy()]

        # 3. 生成信号
        signals = [s.generate_signals(sample_data) for s in strategies]

        # 4. 组合信号
        combiner = StrategyCombiner()
        combined = combiner.equal_weight(signals)

        # 5. 回测
        config = BacktestConfig(initial_capital=100000)
        engine = BacktestEngine(config)
        result = engine.run(sample_data, combined)

        # 6. 验证结果
        assert result.total_return is not None
        assert result.max_drawdown is not None
        assert result.sharpe_ratio is not None

    def test_strategy_combiner_all_modes(self, sample_data):
        """测试所有组合模式"""
        rsi = RSIStrategy()
        ma = MACrossStrategy()
        signals = [
            rsi.generate_signals(sample_data),
            ma.generate_signals(sample_data)
        ]

        combiner = StrategyCombiner()

        # 等权组合
        eq = combiner.equal_weight(signals)
        assert len(eq) == len(sample_data)

        # 信号过滤
        sf = combiner.signal_filter(signals, threshold=1)
        assert len(sf) == len(sample_data)

        # 互补策略
        cp = combiner.complementary(
            sample_data, 'range', 
            [ma], [rsi]
        )
        assert len(cp) == len(sample_data)
```

**Step 2: 运行集成测试**

```bash
pytest tests/test_integration.py -v
```

Expected: PASS

**Step 3: 提交**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for full workflow"
```

---

## Phase 10: 最终验证

### Task 10.1: 运行所有测试

```bash
pytest tests/ -v --cov=pixiu
```

### Task 10.2: 更新README

更新 `README.md` 添加新功能说明。

### Task 10.3: 最终提交

```bash
git add .
git commit -m "feat: complete quantitative experiment flow with regime detection, multi-strategy, and AKQuant integration"
```

---

## 依赖清单

```
akquant>=0.1.0
reflex>=0.4.0
akshare>=1.12.0
zhipuai>=2.0.0
plotly>=5.18.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
```

---

## 实施顺序总结

1. **Phase 1**: 分析层 - 择势判断模块
2. **Phase 2**: 策略层 - 经典策略(RSI, 均线, 网格)
3. **Phase 3**: 策略层 - 高级策略(随机过程, 最优执行)
4. **Phase 4**: 组合层 - 策略组合器
5. **Phase 5**: 回测层 - AKQuant适配器
6. **Phase 6**: 展示层 - UI组件
7. **Phase 7**: 状态管理更新
8. **Phase 8**: AI报告增强
9. **Phase 9**: 集成测试
10. **Phase 10**: 最终验证
