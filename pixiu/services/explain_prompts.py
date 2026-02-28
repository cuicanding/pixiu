"""AI explanation prompt templates for quantitative metrics."""

EXPLAIN_PROMPTS = {
    "total_return": """用简单易懂的中文解释：
1. 什么是总收益率？
2. 数值 {value} 代表什么水平？
3. 对普通投资者意味着什么？""",

    "annualized_return": """用简单易懂的中文解释：
1. 什么是年化收益率？
2. 数值 {value} 代表什么水平？
3. 与总收益率有什么区别？""",

    "sharpe_ratio": """用简单易懂的中文解释：
1. 什么是夏普比率？
2. 数值 {value} 代表什么水平？（>1优秀，>2很好，>3卓越）
3. 为什么它比单纯看收益率更重要？""",

    "max_drawdown": """用简单易懂的中文解释：
1. 什么是最大回撤？
2. 数值 {value} 代表什么风险水平？
3. 投资者应该如何理解和应对？""",

    "win_rate": """用简单易懂的中文解释：
1. 什么是胜率？
2. 数值 {value} 代表什么水平？
3. 胜率高一定赚钱吗？""",

    "profit_loss_ratio": """用简单易懂的中文解释：
1. 什么是盈亏比？
2. 数值 {value} 代表什么水平？
3. 为什么说盈亏比比胜率更重要？""",

    "calmar_ratio": """用简单易懂的中文解释：
1. 什么是卡玛比率？
2. 数值 {value} 代表什么水平？
3. 它与夏普比率有什么区别？""",

    "volatility": """用简单易懂的中文解释：
1. 什么是波动率？
2. 数值 {value} 代表什么水平？
3. 高波动和低波动各有什么优缺点？""",

    "adx": """用简单易懂的中文解释：
1. 什么是ADX（平均趋向指数）？
2. 数值 {value} 代表什么趋势强度？
3. 投资者如何利用ADX判断市场状态？""",

    "regime_trend": """用简单易懂的中文解释：
1. 什么是"趋势行情"？
2. 当前市场状态：{regime}
3. 趋势行情下应该使用什么策略？""",

    "regime_range": """用简单易懂的中文解释：
1. 什么是"震荡行情"？
2. 当前市场状态：{regime}
3. 震荡行情下应该使用什么策略？""",

    "strategy_recommend": """基于以下市场状态推荐策略：
- 大盘状态：{market_regime}
- 个股状态：{stock_regime}

请解释：
1. 为什么在这种市场状态下推荐这些策略？
2. 这些策略的原理是什么？
3. 使用时需要注意什么？""",

    "backtest_summary": """用简单易懂的中文总结这次回测：
- 策略：{strategy}
- 总收益率：{total_return}
- 夏普比率：{sharpe_ratio}
- 最大回撤：{max_drawdown}
- 胜率：{win_rate}

请给出：
1. 整体表现评价
2. 主要优缺点
3. 改进建议""",
}


def get_prompt(concept: str, **kwargs) -> str:
    """Get formatted prompt for a concept.
    
    Args:
        concept: The concept key (e.g., 'total_return', 'sharpe_ratio')
        **kwargs: Values to format into the prompt template
        
    Returns:
        Formatted prompt string
    """
    template = EXPLAIN_PROMPTS.get(concept, "请解释 {concept}")
    try:
        return template.format(**kwargs, concept=concept)
    except KeyError as e:
        return template.format(concept=concept, **{k: v for k, v in kwargs.items() if k in str(e)})
