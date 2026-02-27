README.md

# Pixiu 量化分析软件

基于 Python + Reflex 的 A股/港股/美股量化分析桌面软件。

## 功能特性

- 📊 **多市场支持**: A股、港股、美股
- 🧠 **智能策略**: 趋势强度、波动率套利、卡尔曼滤波
- 📈 **回测分析**: 完整的回测指标和可视化
- 🤖 **AI分析**: GLM-5 智能分析报告

## 快速开始

### 安装依赖

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 配置

复制 `.env.example` 为 `.env` 并填写你的 GLM API Key:

```bash
cp .env.example .env
```

编辑 `.env`:
```
GLM_API_KEY=your_api_key_here
DATABASE_PATH=data/stocks.db
CACHE_DIR=data/cache
```

### 运行

```bash
reflex run
```

浏览器将自动打开 http://localhost:3000

## 项目结构

```
pixiu/
├── pixiu/              # 主应用
│   ├── pages/          # 页面组件
│   │   ├── home.py     # 首页
│   │   ├── backtest.py # 回测报告页
│   │   └── settings.py # 设置页
│   ├── services/       # 业务逻辑
│   │   ├── database.py # 数据库服务
│   │   ├── data_service.py # 数据获取
│   │   ├── backtest_service.py # 回测引擎
│   │   └── ai_service.py # AI分析
│   ├── strategies/     # 量化策略
│   │   ├── base.py      # 策略基类
│   │   ├── trend_strength.py # 趋势强度
│   │   ├── volatility.py # 波动率套利
│   │   └── kalman_filter.py # 卡尔曼滤波
│   ├── models/         # 数据模型
│   ├── config.py       # 配置管理
│   ├── state.py        # 全局状态
│   └── pixiu.py        # 应用入口
├── data/               # 数据存储
├── tests/              # 测试文件
├── requirements.txt    # 依赖列表
└── README.md           # 本文件
```

## 策略说明

### 1. 趋势强度策略 (Trend Strength)

基于微积分导数分析价格趋势：
- **一阶导数**: 判断价格变化方向
- **二阶导数**: 判断趋势加速度

**信号规则**:
- 买入: 价格上升且加速 (f'(t) > 0 且 f''(t) > 0)
- 卖出: 价格下跌且加速 (f'(t) < 0 且 f''(t) < 0)

### 2. 波动率套利策略 (Volatility Arbitrage)

基于波动率积分的均值回归：
- 计算历史波动率
- 积分波动率能量
- 在极端波动时反向操作

### 3. 卡尔曼滤波策略 (Kalman Filter)

使用卡尔曼滤波估计真实价格：
- 过滤市场噪声
- 捕捉价格偏离
- 在价格显著偏离估计时交易

## 回测指标

| 指标 | 说明 |
|------|------|
| 总收益率 | 整体盈亏比例 |
| 年化收益 | 标准化年化回报 |
| 最大回撤 | 最大亏损幅度 |
| 夏普比率 | 风险调整收益 |
| 胜率 | 盈利交易占比 |
| 盈亏比 | 平均盈利/平均亏损 |
| 卡玛比率 | 收益/最大回撤 |

## 使用流程

1. **选择市场**: A股/港股/美股
2. **搜索股票**: 输入股票代码或名称
3. **选择策略**: 勾选要使用的策略
4. **开始分析**: 点击按钮执行回测
5. **查看报告**: 在回测页面查看详细结果
6. **AI分析**: 生成智能分析报告（需配置API Key）

## 开发

### 运行测试

```bash
pytest tests/
```

### 打包

```bash
reflex export
```

## 技术栈

- **前端**: Reflex (React + Chakra UI)
- **后端**: Python + FastAPI
- **数据库**: SQLite
- **数据处理**: Pandas + NumPy
- **量化计算**: SciPy
- **可视化**: Plotly
- **数据源**: akshare
- **AI**: GLM-5 API

## License

MIT

## 贡献

欢迎提交 Issue 和 Pull Request!
