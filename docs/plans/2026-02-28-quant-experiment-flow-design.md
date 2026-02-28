# Pixiu 量化实验流程设计文档

**Date**: 2026-02-28
**Author**: AI Assistant
**Status**: Approved

---

## 1. 概述

本文档设计了一个完整的量化交易实验流程，包括：

- 大盘/股票择势判断（震荡/趋势自动识别）
- 多策略支持（经典+高级）
- 策略组合（等权/信号过滤/互补）
- AKQuant高性能回测
- AI智能报告

---

## 2. 整体架构

采用**分层架构**，各层职责清晰：

```
┌─────────────────────────────────────────────────────────────────┐
│                      展示层 (Presentation)                        │
│  Reflex UI + AI报告生成（GLM-5）                                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                      组合层 (Combination)                         │
│  等权组合 | 信号过滤 | 互补策略（趋势+震荡自动切换）                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                      回测层 (Backtest)                            │
│  AKQuant高性能回测引擎 + 风控模块 + 指标计算                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                      策略层 (Strategy)                            │
│  经典策略：RSI | 均线 | 网格交易                                    │
│  高级策略：趋势强度 | 波动率套利 | 随机过程 | 最优执行               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                      分析层 (Analysis)                            │
│  大盘择势：ADX/MA斜率/波动率 → 震荡/趋势                           │
│  股票择势：同上指标 → 自动适配策略                                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                      数据层 (Data)                                │
│  AKShare数据源 + AKQuant数据适配 + SQLite缓存                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 分析层 - 择势判断

### 3.1 市场状态分类

- **趋势 (Trend)**: 价格持续朝一个方向运动，适合跟踪策略
- **震荡 (Range)**: 价格在一定区间内波动，适合均值回归策略

### 3.2 判断指标

| 指标   | 趋势信号          | 震荡信号          | 权重  |
| ---- | ------------- | ------------- | --- |
| ADX  | > 25          | < 25          | 40% |
| MA斜率 | \|斜率\| > 0.5% | \|斜率\| < 0.5% | 30% |
| 波动率  | 持续扩张          | 收缩或稳定         | 30% |

### 3.3 核心类设计

```python
class MarketRegimeDetector:
    """大盘/个股择势判断"""

    def detect_regime(self, df: pd.DataFrame) -> str:
        """返回 'trend' 或 'range'"""
        adx_score = self._calc_adx(df)
        slope_score = self._calc_ma_slope(df)
        vol_score = self._calc_volatility(df)

        # 加权投票
        trend_votes = 0
        if adx_score > 25:
            trend_votes += 0.4
        if abs(slope_score) > 0.005:
            trend_votes += 0.3
        if vol_score > self.vol_threshold:
            trend_votes += 0.3

        return "trend" if trend_votes > 0.5 else "range"

    def _calc_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """计算ADX指标"""

    def _calc_ma_slope(self, df: pd.DataFrame, period: int = 20) -> float:
        """计算MA斜率"""

    def _calc_volatility(self, df: pd.DataFrame, period: int = 20) -> float:
        """计算波动率"""

    def get_analysis_detail(self, df: pd.DataFrame) -> Dict:
        """返回详细分析结果（用于UI展示）"""
        return {
            "regime": self.detect_regime(df),
            "adx": self._calc_adx(df),
            "ma_slope": self._calc_ma_slope(df),
            "volatility": self._calc_volatility(df),
        }
```

### 3.4 策略-市场适配

```python
STRATEGY_REGIME_MAP = {
    "trend": ["趋势强度策略", "均线交叉策略", "最优执行策略"],
    "range": ["网格交易策略", "RSI策略", "波动率套利策略"],
    "any": ["随机过程策略", "卡尔曼滤波策略"]
}
```

---

## 4. 策略层

### 4.1 策略分类

#### 经典策略

| 策略   | 适用场景 | 核心逻辑                | 参数                            |
| ---- | ---- | ------------------- | ----------------------------- |
| RSI  | 震荡   | 超卖买入(<30)，超买卖出(>70) | period=14                     |
| 均线交叉 | 趋势   | 金叉买入，死叉卖出           | fast=5, slow=20               |
| 网格交易 | 震荡   | 固定间隔挂单，低买高卖         | grid_size=0.02, grid_count=10 |

#### 高级策略

| 策略    | 适用场景 | 核心逻辑          | 参数                  |
| ----- | ---- | ------------- | ------------------- |
| 趋势强度  | 趋势   | 价格导数分析，捕捉加速度  | threshold=0.01      |
| 波动率套利 | 震荡   | 波动率均值回归       | vol_period=20       |
| 随机过程  | 任意   | 布朗运动+跳扩散建模    | mu, sigma, lambda   |
| 最优执行  | 趋势   | TWAP/VWAP执行算法 | execution_window=30 |

### 4.2 策略基类扩展

```python
class BaseStrategy(ABC):
    name: str
    description: str
    regime: str  # "trend" | "range" | "any" - 新增字段
    params: Dict[str, Any]

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """生成交易信号: 1=买入, -1=卖出, 0=持有"""
        pass

    def get_param_schema(self) -> Dict:
        """返回UI参数配置schema"""
        return {
            "type": "object",
            "properties": {
                k: {"type": "number", "default": v, "description": f"{k}参数"}
                for k, v in self.params.items()
            }
        }

    def validate_params(self) -> bool:
        """参数校验"""
        return True
```

### 4.3 策略示例：网格交易

```python
@register_strategy
class GridTradingStrategy(BaseStrategy):
    name = "网格交易策略"
    description = "在价格区间内设置网格，低买高卖"
    regime = "range"
    params = {"grid_size": 0.02, "grid_count": 10}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        close = df['close']

        # 计算网格线
        base_price = close.iloc[0]
        grid_size = self.params['grid_size']
        grid_count = self.params['grid_count']

        grid_levels = [base_price * (1 + i * grid_size) 
                       for i in range(-grid_count, grid_count + 1)]

        position = 0
        for i, price in enumerate(close):
            if position == 0 and price < grid_levels[grid_count]:
                signals.iloc[i] = 1  # 买入
                position = 1
            elif position > 0 and price > grid_levels[grid_count]:
                signals.iloc[i] = -1  # 卖出
                position = 0

        return signals
```

---

## 5. 组合层 - 策略组合器

### 5.1 三种组合模式

#### 等权组合 (Equal Weight)

所有策略信号取平均，>0买入，<0卖出。

```python
def equal_weight(self, signals: List[pd.Series]) -> pd.Series:
    combined = sum(signals) / len(signals)
    return (combined > 0).astype(int) - (combined < 0).astype(int)
```

#### 信号过滤 (Signal Filter)

N个以上策略一致时才执行。

```python
def signal_filter(self, signals: List[pd.Series], threshold: int = 2) -> pd.Series:
    buy_votes = sum(s == 1 for s in signals)
    sell_votes = sum(s == -1 for s in signals)
    result = pd.Series(0, index=signals[0].index)
    result[buy_votes >= threshold] = 1
    result[sell_votes >= threshold] = -1
    return result
```

#### 互补策略 (Complementary)

根据市场状态自动切换策略组。

```python
def complementary(self, df: pd.DataFrame, regime: str,
                  trend_strategies: List, range_strategies: List) -> pd.Series:
    if regime == "trend":
        strategies = trend_strategies
    else:
        strategies = range_strategies

    signals = [s.generate_signals(df) for s in strategies]
    return self.equal_weight(signals)
```

### 5.2 组合器核心类

```python
class StrategyCombiner:
    """策略组合器"""

    COMBINE_MODES = ["equal_weight", "signal_filter", "complementary"]

    def __init__(self, config: Dict = None):
        self.config = config or {
            "mode": "complementary",
            "filter_threshold": 2,
            "trend_strategies": ["趋势强度策略", "均线交叉策略"],
            "range_strategies": ["网格交易策略", "RSI策略", "波动率套利策略"]
        }

    def combine(self, signals: List[pd.Series], regime: str = None) -> pd.Series:
        mode = self.config["mode"]

        if mode == "equal_weight":
            return self.equal_weight(signals)
        elif mode == "signal_filter":
            return self.signal_filter(signals, self.config["filter_threshold"])
        elif mode == "complementary":
            return self.complementary(regime, signals)

        raise ValueError(f"Unknown combine mode: {mode}")
```

---

## 6. 回测层 - AKQuant集成

### 6.1 集成方式

AKQuant是基于Rust+Python的高性能回测框架，提供：

- 极致性能（比传统Python框架快X倍）
- 内置ML支持（Walk-forward Validation）
- 因子表达式引擎
- 完善的风控模块

### 6.2 适配器设计

```python
import akquant as aq
from akquant import Strategy

class AKQuantAdapter:
    """AKQuant适配器"""

    def run_backtest(self, df: pd.DataFrame, 
                     strategy: BaseStrategy,
                     config: Dict) -> BacktestResult:
        # 创建AKQuant策略包装器
        aq_strategy = self._wrap_strategy(strategy)

        # 运行回测
        result = aq.run_backtest(
            data=df,
            strategy=aq_strategy,
            initial_cash=config.get('initial_capital', 100000),
            symbol=config.get('symbol', 'stock')
        )

        return self._convert_result(result)

    def _wrap_strategy(self, strategy: BaseStrategy):
        """将Pixiu策略包装为AKQuant策略"""
        class WrappedStrategy(Strategy):
            def __init__(self):
                self.pixiu_strategy = strategy

            def on_bar(self, bar):
                df = self._bar_to_dataframe(bar)
                signal = self.pixiu_strategy.generate_signals(df).iloc[-1]

                if signal == 1:
                    self.buy(symbol=bar.symbol, quantity=self.calc_position())
                elif signal == -1:
                    self.close_position(symbol=bar.symbol)

        return WrappedStrategy

    def generate_report(self, result, output_path: str):
        """生成可视化报告"""
        result.report(show=False, output_path=output_path)
```

### 6.3 回测指标

AKQuant内置指标：

| 类别   | 指标                              |
| ---- | ------------------------------- |
| 收益   | 总收益率、年化收益、累计收益                  |
| 风险   | 最大回撤、波动率、VaR、CVaR               |
| 风险调整 | 夏普比率、索提诺比率、卡玛比率                 |
| 交易质量 | 胜率、盈亏比、平均持仓时间                   |
| 其他   | Ulcer Index、SQN、Kelly Criterion |

---

## 7. 展示层 - 实验流程UI

### 7.1 标准6步实验流程

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: 选择股票                                                │
│  ├── 搜索股票（代码/名称）                                        │
│  └── 显示基本信息                                                │
├─────────────────────────────────────────────────────────────────┤
│  STEP 2: 择势判断                                                │
│  ├── 自动识别大盘状态 → 趋势/震荡                                 │
│  ├── 自动识别个股状态 → 趋势/震荡                                 │
│  └── 显示判断依据（ADX、MA斜率、波动率数值）                      │
├─────────────────────────────────────────────────────────────────┤
│  STEP 3: 选择策略                                                │
│  ├── 系统根据择势结果推荐适合的策略                               │
│  ├── 用户可选择多个策略                                          │
│  └── 可调整策略参数                                              │
├─────────────────────────────────────────────────────────────────┤
│  STEP 4: 策略组合（可选）                                        │
│  ├── 等权组合                                                    │
│  ├── 信号过滤                                                    │
│  └── 互补切换（自动根据市场状态选择策略组）                       │
├─────────────────────────────────────────────────────────────────┤
│  STEP 5: 执行回测                                                │
│  ├── 配置回测参数（资金、手续费、仓位）                           │
│  └── 运行AKQuant回测                                             │
├─────────────────────────────────────────────────────────────────┤
│  STEP 6: 查看报告                                                │
│  ├── 回测指标（收益、风险、风险调整）                             │
│  ├── 资金曲线图（Plotly交互式）                                   │
│  ├── 交易记录表                                                  │
│  └── AI完整分析报告                                              │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 AI报告增强

```python
class AIReportService:
    """增强的AI报告服务"""

    async def generate_full_report(self,
        stock_info: Dict,
        regime_analysis: Dict,
        backtest_results: List[Dict],
        strategy_params: Dict
    ) -> str:
        prompt = f"""
        请分析以下量化回测结果并生成专业报告：

        ## 1. 股票信息
        - 代码：{stock_info['code']}
        - 名称：{stock_info['name']}
        - 市场：{stock_info['market']}

        ## 2. 择势判断
        - 大盘状态：{regime_analysis['market_regime']}
        - 个股状态：{regime_analysis['stock_regime']}
        - ADX：{regime_analysis['adx']:.2f}
        - MA斜率：{regime_analysis['ma_slope']:.4f}
        - 波动率：{regime_analysis['volatility']:.4f}

        ## 3. 回测表现
        - 策略：{backtest_results['strategy']}
        - 总收益率：{backtest_results['total_return']:.2%}
        - 年化收益：{backtest_results['annualized_return']:.2%}
        - 最大回撤：{backtest_results['max_drawdown']:.2%}
        - 夏普比率：{backtest_results['sharpe_ratio']:.2f}
        - 胜率：{backtest_results['win_rate']:.2%}

        请从以下角度进行分析：
        1. **策略表现评估**：策略在该股票上的表现如何？是否符合预期？
        2. **择势判断准确性**：市场状态判断是否准确？对策略选择的影响？
        3. **风险提示**：主要风险点有哪些？最大回撤是否可接受？
        4. **改进建议**：有哪些可以优化的地方？
        5. **适用性评估**：该策略适合什么类型的投资者？
        """

        return await self._call_glm_api(prompt)
```

---

## 8. 文件结构

```
pixiu/
├── pixiu/
│   ├── analysis/              # 新增：分析层
│   │   ├── __init__.py
│   │   └── regime_detector.py # 择势判断
│   │
│   ├── strategies/            # 扩展：策略层
│   │   ├── __init__.py
│   │   ├── base.py            # 策略基类（扩展regime字段）
│   │   ├── classic/           # 新增：经典策略
│   │   │   ├── __init__.py
│   │   │   ├── rsi.py
│   │   │   ├── ma_cross.py
│   │   │   └── grid_trading.py
│   │   ├── advanced/          # 高级策略
│   │   │   ├── __init__.py
│   │   │   ├── trend_strength.py  # 已有
│   │   │   ├── volatility.py      # 已有
│   │   │   ├── kalman_filter.py   # 已有
│   │   │   ├── stochastic.py      # 新增
│   │   │   └── optimal_execution.py # 新增
│   │   └── combiner.py        # 新增：策略组合器
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── data_service.py    # 已有
│   │   ├── backtest_service.py # 改造：AKQuant适配器
│   │   ├── akquant_adapter.py # 新增：AKQuant适配
│   │   └── ai_service.py      # 增强：完整报告
│   │
│   ├── pages/
│   │   ├── __init__.py
│   │   ├── home.py            # 改造：6步实验流程
│   │   ├── backtest.py        # 已有
│   │   └── settings.py        # 已有
│   │
│   ├── components/            # 新增组件
│   │   ├── __init__.py
│   │   ├── metric_card.py     # 已有
│   │   ├── stock_card.py      # 已有
│   │   ├── strategy_card.py   # 已有
│   │   ├── regime_indicator.py # 新增：择势状态显示
│   │   ├── strategy_recommender.py # 新增：策略推荐
│   │   └── experiment_flow.py  # 新增：实验流程向导
│   │
│   ├── models/
│   │   └── ...
│   ├── config.py
│   ├── state.py               # 改造：支持新功能
│   └── pixiu.py
│
└── docs/plans/
    └── 2026-02-28-quant-experiment-flow-design.md  # 本文档
```

---

## 9. 依赖

```txt
# requirements.txt 更新
akquant>=0.1.0          # 高性能回测框架
ta-lib>=0.4.28          # 技术指标库（可选，ADX等）

# 已有依赖保持不变
reflex>=0.4.0
akshare>=1.12.0
zhipuai>=2.0.0
plotly>=5.18.0
pandas>=2.0.0
numpy>=1.24.0
```

---

## 10. 实施阶段

### Phase 1: 分析层

- [ ] 实现 `MarketRegimeDetector` 类
- [ ] 集成ADX、MA斜率、波动率计算
- [ ] 添加择势状态UI组件

### Phase 2: 策略层

- [ ] 扩展 `BaseStrategy` 添加 `regime` 字段
- [ ] 实现经典策略：RSI、均线交叉、网格交易
- [ ] 实现高级策略：随机过程、最优执行

### Phase 3: 组合层

- [ ] 实现 `StrategyCombiner` 类
- [ ] 实现三种组合模式
- [ ] 添加组合配置UI

### Phase 4: 回测层

- [ ] 实现 `AKQuantAdapter`
- [ ] 改造现有回测服务
- [ ] 集成AKQuant可视化报告

### Phase 5: 展示层

- [ ] 改造首页为6步实验流程
- [ ] 实现择势判断UI
- [ ] 实现策略推荐UI
- [ ] 增强AI报告功能

---

## 11. 成功标准

1. **择势判断准确** - ADX/MA斜率/波动率组合判断市场状态
2. **策略分类清晰** - 经典/高级策略，适用场景明确
3. **组合功能完整** - 等权/过滤/互补三种模式可用
4. **回测性能提升** - AKQuant集成后回测速度显著提升
5. **实验流程清晰** - 6步流程引导用户完成完整分析
6. **AI报告专业** - 包含择势分析、策略评估、风险提示、改进建议
