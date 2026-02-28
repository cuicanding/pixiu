"""GLM-5 AI分析服务"""

from typing import Dict, Optional, List

try:
    from zhipuai import ZhipuAI
except ImportError:
    ZhipuAI = None

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class AIReportService:
    """GLM-5 AI分析服务"""
    
    SYSTEM_PROMPT = """你是一位专业的量化分析师。
你的任务是将回测数据转化为易懂的投资建议和分析报告。

请用中文回答，报告应包含：
1. **策略表现总结** - 用简洁语言总结策略的整体表现
2. **优势分析** - 策略在哪些方面表现良好
3. **风险分析** - 需要注意的风险点
4. **改进建议** - 针对该策略的优化建议
5. **适用场景** - 该策略适合什么样的市场环境

请保持客观、专业的分析风格。"""
    
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        if ZhipuAI and api_key:
            self.client = ZhipuAI(api_key=api_key)
        else:
            self.client = None
    
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
        results: List[Dict],
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
    
    async def generate_full_report(self, stock_info: Dict, regime_analysis: Dict, backtest_results: List[Dict], strategy_params: Dict) -> str:
        """生成完整的分析报告，包含择势判断和多策略结果"""
        
        if not self.client:
            return "请先配置 GLM API Key"
        
        prompt = f"""请分析以下量化回测结果并生成专业报告：

## 1. 股票信息
- 代码：{stock_info.get('code', 'N/A')}
- 名称：{stock_info.get('name', 'N/A')}
- 市场：{stock_info.get('market', 'N/A')}

## 2. 择势判断
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
"""
        prompt += """
请分析：
1. 策略表现评估
2. 择势判断准确性
3. 风险提示
4. 改进建议
"""
        return await self._call_api(prompt)
    
    async def _call_api(self, prompt: str) -> str:
        """调用GLM API"""
        if not self.client:
            return "请先配置 GLM API Key"
        
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
            logger.error(f"AI分析生成失败: {e}")
            return f"AI分析生成失败: {str(e)}"
