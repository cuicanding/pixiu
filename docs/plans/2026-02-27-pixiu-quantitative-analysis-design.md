# Pixiu 量化分析软件设计文档

## 项目概述

**Pixiu (貔貅)** 是一款基于Python的A股/港股/美股量化分析桌面软件，提供策略实验、回测分析和AI智能解读功能。

### 目标用户
- 有Python基础但对量化分析不太熟悉的投资者
- 希望通过实验学习量化策略的用户
- 需要验证策略效果后用于实际交易的用户

### 核心价值
- **学习**: 通过可视化实验理解量化策略原理
- **验证**: 回测功能验证策略历史表现
- **决策**: AI解读辅助投资决策

---

## 技术栈

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| 框架 | Reflex | 纯Python全栈框架，支持桌面打包 |
| 数据源 | akshare | 支持A股/港股/美股数据获取 |
| 存储 | SQLite | 轻量级本地数据库 |
| 数据处理 | Pandas + NumPy | 数据清洗和计算 |
| 科学计算 | SciPy | 微积分、滤波等数学运算 |
| 可视化 | Plotly | 交互式图表 |
| AI | GLM-5 API | 智能分析报告生成 |

---

## 系统架构

### 目录结构

```
pixiu/
├── pixiu/                    # 主应用包
│   ├── __init__.py
│   ├── pixiu.py              # Reflex应用入口
│   ├── config.py             # 配置管理
│   │
│   ├── pages/                # 页面组件
│   │   ├── home.py           # 首页/股票选择
│   │   ├── analysis.py       # 策略分析页面
│   │   ├── backtest.py       # 回测结果页面
│   │   └── settings.py       # 设置页面
│   │
│   ├── components/           # 可复用UI组件
│   │   ├── stock_selector.py # 股票选择器
│   │   ├── chart_panel.py    # 图表面板
│   │   └── report_panel.py   # 报告面板
│   │
│   ├── services/             # 业务逻辑层
│   │   ├── data_service.py   # 数据获取与管理
│   │   ├── strategy_service.py # 策略执行引擎
│   │   ├── backtest_service.py # 回测引擎
│   │   └── ai_service.py     # GLM API集成
│   │
│   ├── strategies/           # 策略插件目录
│   │   ├── base.py           # 策略基类
│   │   ├── trend_strength.py # 趋势强度策略
│   │   ├── volatility.py     # 波动率套利策略
│   │   ├── kalman_filter.py  # 卡尔曼滤波策略
│   │   └── random_walk.py    # 随机游走策略
│   │
│   ├── models/               # 数据模型
│   │   ├── stock.py          # 股票数据模型
│   │   ├── backtest.py       # 回测结果模型
│   │   └── signal.py         # 交易信号模型
│   │
│   └── utils/                # 工具函数
│       ├── calculus.py       # 微积分工具
│       ├── indicators.py     # 技术指标计算
│       └── visualization.py  # 可视化工具
│
├── data/                     # 数据目录
│   ├── stocks.db             # SQLite数据库
│   └── cache/                # 缓存目录
│
├── assets/                   # 静态资源
│   └── styles.css            # 自定义样式
│
├── requirements.txt
└── rxconfig.py               # Reflex配置
```

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        UI Layer (Reflex)                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │  Home   │  │Analysis │  │Backtest │  │Settings │        │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │
└───────┼────────────┼────────────┼────────────┼──────────────┘
        │            │            │            │
        └────────────┴─────┬──────┴────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                     Service Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ DataService  │  │StrategyService│  │BacktestEngine│       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                  │               │
│         └─────────────────┼──────────────────┘               │
│                           │                                  │
│                  ┌────────▼────────┐                         │
│                  │  AIService      │                         │
│                  │  (GLM-5 API)    │                         │
│                  └─────────────────┘                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    Strategy Layer                            │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌───────────┐ │
│  │TrendStrength│ │ Volatility │ │KalmanFilter│ │RandomWalk │ │
│  └────────────┘ └────────────┘ └────────────┘ └───────────┘ │
│           所有策略继承 BaseStrategy，插件式注册               │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                     Data Layer                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │   SQLite    │    │   akshare   │    │   Cache     │      │
│  │ (本地存储)   │    │ (数据源)    │    │  (缓存)     │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## 数据层设计

### 数据库表结构

```sql
-- 股票基础信息
CREATE TABLE stocks (
    code TEXT PRIMARY KEY,      -- 股票代码 (如 600519.SH, 00700.HK, AAPL.US)
    name TEXT,                  -- 股票名称
    market TEXT,                -- 市场 (A股/港股/美股)
    industry TEXT,              -- 所属行业
    list_date DATE,             -- 上市日期
    updated_at TIMESTAMP        -- 数据更新时间
);

-- 日线行情数据
CREATE TABLE daily_quotes (
    id INTEGER PRIMARY KEY,
    code TEXT,                  -- 股票代码
    trade_date DATE,            -- 交易日期
    open REAL,                  -- 开盘价
    high REAL,                  -- 最高价
    low REAL,                   -- 最低价
    close REAL,                 -- 收盘价
    volume REAL,                -- 成交量
    amount REAL,                -- 成交额
    turnover_rate REAL,         -- 换手率
    FOREIGN KEY (code) REFERENCES stocks(code),
    UNIQUE(code, trade_date)
);

CREATE INDEX idx_quotes_code_date ON daily_quotes(code, trade_date);

-- 策略信号记录
CREATE TABLE strategy_signals (
    id INTEGER PRIMARY KEY,
    code TEXT,
    strategy_name TEXT,         -- 策略名称
    signal_date DATE,           -- 信号日期
    signal_type TEXT,           -- BUY/SELL/HOLD
    confidence REAL,            -- 信号置信度 0-1
    price REAL,                 -- 信号时价格
    metadata TEXT               -- 额外参数 (JSON)
);

-- 数据更新记录
CREATE TABLE update_logs (
    id INTEGER PRIMARY KEY,
    market TEXT,
    last_update TIMESTAMP,
    records_updated INTEGER
);
```

### 数据更新策略

```
┌─────────────────┐
│   应用启动      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     是      ┌─────────────────┐
│  首次使用？      │──────────▶ │ 提示下载全量数据  │
└────────┬────────┘             │ A股:3年 其他:2年 │
         │ 否                   └─────────────────┘
         ▼
┌─────────────────┐
│ 检查最后更新时间 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  增量更新缺失数据 │
└─────────────────┘
```

---

## 策略层设计

### 策略基类

```python
from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    """所有策略必须继承此类"""
    
    name: str = ""
    description: str = ""
    params: dict = {}
    
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
    def get_required_data(self) -> list[str]:
        """返回需要的数据列"""
        pass
    
    def validate_params(self) -> bool:
        """参数校验"""
        return True
    
    def get_documentation(self) -> str:
        """返回策略的数学原理说明（Markdown格式）"""
        return ""
```

### 策略注册机制

```python
# strategies/__init__.py
STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {}

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

### 初始策略列表

| 策略名称 | 数学原理 | 说明 |
|----------|----------|------|
| 趋势强度策略 | 导数应用 | f'(t)>0上升，f''(t)判断加速度 |
| 波动率套利策略 | 积分/导数 | 波动率积分判断超买超卖 |
| 卡尔曼滤波策略 | 微分方程 | 状态估计与噪声过滤 |
| 随机游走策略 | 布朗运动 | 均值回归交易 |

---

## 回测引擎设计

### 回测配置

```python
@dataclass
class BacktestConfig:
    initial_capital: float = 100000   # 初始资金
    commission_rate: float = 0.0003    # 手续费率 0.03%
    slippage_rate: float = 0.0001      # 滑点率
    position_size: float = 0.95        # 仓位比例
    risk_free_rate: float = 0.03       # 无风险利率（用于夏普比率）
```

### 回测输出指标

| 指标 | 公式 | 说明 |
|------|------|------|
| 总收益率 | (期末-期初)/期初 | 整体盈亏 |
| 年化收益率 | 总收益率^(365/天数)-1 | 标准化比较 |
| 最大回撤 | max((峰值-谷值)/峰值) | 风险度量 |
| 夏普比率 | (收益-无风险利率)/波动率 | 风险调整收益 |
| 胜率 | 盈利次数/总次数 | 成功率 |
| 盈亏比 | 平均盈利/平均亏损 | 收益质量 |
| 卡玛比率 | 年化收益/最大回撤 | 回撤效率 |
| 总交易次数 | - | 换手频率 |

---

## UI界面设计

### 页面结构

1. **首页 (Home)**
   - 市场选择（A股/港股/美股）
   - 股票搜索与选择
   - K线图展示
   - 策略选择面板

2. **分析页 (Analysis)**
   - 单策略详细分析
   - 参数调节面板
   - 信号可视化

3. **回测页 (Backtest)**
   - 核心指标卡片
   - 收益曲线对比
   - 回撤分析图
   - 交易记录表
   - AI智能报告

4. **设置页 (Settings)**
   - GLM API Key 配置
   - 数据更新设置
   - 策略参数默认值

### 异步处理

所有耗时操作使用 `@rx.background` 装饰器，确保UI不阻塞：

```python
class AppState(rx.State):
    is_loading: bool = False
    loading_message: str = ""
    progress: float = 0.0
    
    @rx.background
    async def run_analysis(self):
        async with self:
            self.is_loading = True
            self.loading_message = "正在分析..."
        
        # 耗时操作
        result = await self._execute_strategies()
        
        async with self:
            self.backtest_result = result
            self.is_loading = False
```

---

## AI集成设计

### GLM-5 服务

```python
from zhipuai import ZhipuAI

class AIReportService:
    def __init__(self, api_key: str):
        self.client = ZhipuAI(api_key=api_key)
    
    async def generate_analysis(
        self, 
        backtest_result: dict,
        stock_info: dict,
        strategy_desc: str
    ) -> str:
        """生成自然语言分析报告"""
        
        response = await self.client.chat.completions.acreate(
            model="glm-5",
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": self._build_prompt(...)}
            ],
            temperature=0.7,
        )
        
        return response.choices[0].message.content
```

### 报告缓存

为避免重复调用API，对相同回测结果的AI报告进行本地缓存。

---

## 实现计划

### Phase 1: 框架搭建
- Reflex项目初始化
- SQLite数据库设计
- 数据获取服务（akshare集成）
- 基础UI框架

### Phase 2: 核心功能
- 趋势强度策略实现
- 回测引擎开发
- 回测结果可视化

### Phase 3: 策略扩展
- 波动率套利策略
- 卡尔曼滤波策略
- 随机游走策略
- 多策略对比功能

### Phase 4: AI集成
- GLM-5 API集成
- 自然语言报告生成
- 报告缓存优化

### Phase 5: 打磨发布
- 界面美化
- 错误处理完善
- 打包测试
- 用户文档

---

## 风险与约束

1. **数据可靠性**: akshare为免费数据源，可能存在延迟或中断
2. **API成本**: GLM-5调用有费用，需控制调用频率
3. **策略局限**: 所有策略仅供参考，不构成投资建议
4. **性能考量**: 大量历史数据处理需注意内存管理

---

## 后续扩展

- 支持更多数据源（Tushare Pro等）
- 实时行情推送
- 策略组合优化
- 风险预警功能
- 交易信号提醒
