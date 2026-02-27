"""快速测试脚本 - 验证核心功能"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Pixiu 量化分析软件 - 核心功能测试")
print("=" * 60)

# 测试1: 导入基础模块
print("\n[1/5] 测试模块导入...")
try:
    from pixiu.config import Config
    from pixiu.models.stock import Stock, DailyQuote
    from pixiu.models.backtest import BacktestResult, Trade
    print("✓ 配置和数据模型导入成功")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

# 测试2: 策略模块
print("\n[2/5] 测试策略模块...")
try:
    from pixiu.strategies import get_all_strategies, get_strategy, STRATEGY_REGISTRY
    from pixiu.strategies.base import BaseStrategy
    from pixiu.strategies.trend_strength import TrendStrengthStrategy
    from pixiu.strategies.volatility import VolatilityStrategy
    from pixiu.strategies.kalman_filter import KalmanFilterStrategy
    
    strategies = get_all_strategies()
    print(f"✓ 已注册 {len(strategies)} 个策略:")
    for s in strategies:
        print(f"  - {s.name}: {s.description}")
except Exception as e:
    print(f"✗ 策略模块测试失败: {e}")
    sys.exit(1)

# 测试3: 数据处理
print("\n[3/5] 测试策略信号生成...")
try:
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # 创建模拟数据
    dates = pd.date_range('2024-01-01', periods=100)
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    
    df = pd.DataFrame({
        'trade_date': dates,
        'close': prices,
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'volume': np.random.randint(1000000, 2000000, 100),
    })
    
    # 测试趋势强度策略
    strategy = TrendStrengthStrategy()
    result_df = strategy.generate_signals(df)
    
    assert 'signal' in result_df.columns, "缺少signal列"
    assert 'trend_strength' in result_df.columns, "缺少trend_strength列"
    
    signal_counts = result_df['signal'].value_counts().to_dict()
    print(f"✓ 趋势强度策略信号生成成功")
    print(f"  信号分布: 买入={signal_counts.get(1, 0)}, 卖出={signal_counts.get(-1, 0)}, 持有={signal_counts.get(0, 0)}")
    
except Exception as e:
    print(f"✗ 策略信号生成测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试4: 回测引擎
print("\n[4/5] 测试回测引擎...")
try:
    from pixiu.services.backtest_service import BacktestEngine, BacktestConfig
    
    engine = BacktestEngine(BacktestConfig(initial_capital=100000))
    result = engine.run(result_df)
    
    print(f"✓ 回测引擎运行成功")
    print(f"  总收益: {result.total_return:.2%}")
    print(f"  年化收益: {result.annualized_return:.2%}")
    print(f"  最大回撤: {result.max_drawdown:.2%}")
    print(f"  夏普比率: {result.sharpe_ratio:.2f}")
    print(f"  总交易次数: {result.total_trades}")
    
except Exception as e:
    print(f"✗ 回测引擎测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试5: 配置和数据模型
print("\n[5/5] 测试配置和数据模型...")
try:
    # 配置测试
    config = Config()
    assert config.initial_capital == 100000.0
    assert config.commission_rate == 0.0003
    
    # 股票模型测试
    stock = Stock(code="600519.SH", name="贵州茅台", market="A股", industry="白酒")
    assert stock.code == "600519.SH"
    assert stock.name == "贵州茅台"
    
    # 回测结果测试
    result_obj = BacktestResult(
        total_return=0.25,
        annualized_return=0.15,
        max_drawdown=-0.10,
        sharpe_ratio=1.5,
        win_rate=0.6,
        profit_loss_ratio=2.0,
        calmar_ratio=1.5,
        total_trades=50
    )
    assert result_obj.total_return == 0.25
    
    print("✓ 配置和数据模型测试通过")
    
except Exception as e:
    print(f"✗ 配置模型测试失败: {e}")
    sys.exit(1)

# 总结
print("\n" + "=" * 60)
print("✅ 所有核心功能测试通过!")
print("=" * 60)
print("\n下一步:")
print("1. 安装依赖: pip install -r requirements.txt")
print("2. 初始化Reflex: reflex init")
print("3. 运行应用: reflex run")
print("\n注意: LSP错误是因为依赖未安装，不影响运行")
