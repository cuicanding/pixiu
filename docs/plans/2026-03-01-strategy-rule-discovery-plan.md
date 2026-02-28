# 策略规则发现系统 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 构建自动化的策略规则发现系统，根据大盘+个股择势组合，发现并保存最佳策略规则

**Architecture:** 单页流程式，输入股票后自动匹配大盘，进行双时间线择势分析，AI推荐策略，批量回测发现规则，验证集验证

**Tech Stack:** Reflex, ECharts, SQLite, GLM AI, Pandas

---

## Task 1: 大盘指数智能匹配

**Files:**
- Create: `pixiu/analysis/regime_matcher.py`
- Test: `tests/test_regime_matcher.py`

**Step 1: Write the failing test**

```python
"""测试大盘指数匹配"""
import pytest
from pixiu.analysis.regime_matcher import match_index, INDEX_MAPPING


def test_match_kcb_stock():
    """科创板股票匹配科创50"""
    assert match_index("688001") == "sh000688"
    assert match_index("688999") == "sh000688"


def test_match_cyb_stock():
    """创业板股票匹配创业板指"""
    assert match_index("300001") == "sz399006"
    assert match_index("300999") == "sz399006"


def test_match_sh_main_board():
    """上证主板匹配上证指数"""
    assert match_index("600000") == "sh000001"
    assert match_index("603999") == "sh000001"


def test_match_sz_main_board():
    """深证主板匹配深证成指"""
    assert match_index("000001") == "sz399001"
    assert match_index("001999") == "sz399001"


def test_match_sme_board():
    """中小板匹配中小板指"""
    assert match_index("002001") == "sz399005"
    assert match_index("002999") == "sz399005"


def test_match_unknown_returns_default():
    """未知代码返回上证指数作为默认"""
    assert match_index("999999") == "sh000001"


def test_get_index_name():
    """获取指数中文名"""
    from pixiu.analysis.regime_matcher import get_index_name
    assert get_index_name("sh000688") == "科创50"
    assert get_index_name("sz399006") == "创业板指"
    assert get_index_name("sh000001") == "上证指数"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_regime_matcher.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'pixiu.analysis.regime_matcher'"

**Step 3: Write minimal implementation**

```python
"""大盘指数智能匹配"""
from typing import Dict

INDEX_MAPPING: Dict[str, str] = {
    "688": "sh000688",  # 科创板 → 科创50
    "300": "sz399006",  # 创业板 → 创业板指
    "60": "sh000001",   # 上证主板 → 上证指数
    "000": "sz399001",  # 深证主板 → 深证成指
    "001": "sz399001",  # 深证主板 → 深证成指
    "002": "sz399005",  # 中小板 → 中小板指
}

INDEX_NAMES: Dict[str, str] = {
    "sh000688": "科创50",
    "sz399006": "创业板指",
    "sh000001": "上证指数",
    "sz399001": "深证成指",
    "sz399005": "中小板指",
    "sh000300": "沪深300",
}


def match_index(stock_code: str) -> str:
    """根据股票代码智能匹配大盘指数
    
    Args:
        stock_code: 股票代码，如 "688001"
        
    Returns:
        大盘指数代码，如 "sh000688"
    """
    for prefix, index_code in INDEX_MAPPING.items():
        if stock_code.startswith(prefix):
            return index_code
    return "sh000001"


def get_index_name(index_code: str) -> str:
    """获取指数中文名
    
    Args:
        index_code: 指数代码，如 "sh000688"
        
    Returns:
        指数中文名，如 "科创50"
    """
    return INDEX_NAMES.get(index_code, index_code)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_regime_matcher.py -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add pixiu/analysis/regime_matcher.py tests/test_regime_matcher.py
git commit -m "feat: 添加大盘指数智能匹配功能"
```

---

## Task 2: 规则数据模型

**Files:**
- Create: `pixiu/models/rules.py`
- Test: `tests/test_rules_model.py`

**Step 1: Write the failing test**

```python
"""测试规则数据模型"""
import pytest
from datetime import datetime
from pixiu.models.rules import Rule, RuleCreate, BacktestRecord


def test_rule_creation():
    """测试规则创建"""
    rule = Rule(
        id=1,
        market_regime="range",
        stock_regime="trend_up",
        best_strategy="布朗运动策略",
        train_return=0.15,
        valid_return=0.12,
        baseline_return=0.05,
        confidence=0.82,
        train_period="2024-01-01~2024-10-31",
        stock_code="300001",
        created_at=datetime.now(),
    )
    assert rule.market_regime == "range"
    assert rule.stock_regime == "trend_up"
    assert rule.best_strategy == "布朗运动策略"
    assert rule.excess_return == 0.07  # valid_return - baseline_return


def test_rule_create():
    """测试规则创建DTO"""
    rule_create = RuleCreate(
        market_regime="range",
        stock_regime="trend_up",
        best_strategy="布朗运动策略",
        train_return=0.15,
        valid_return=0.12,
        baseline_return=0.05,
        confidence=0.82,
        train_period="2024-01-01~2024-10-31",
        stock_code="300001",
    )
    assert rule_create.market_regime == "range"


def test_backtest_record():
    """测试回测记录"""
    record = BacktestRecord(
        rule_id=1,
        strategy_name="布朗运动策略",
        total_return=0.15,
        sharpe=1.2,
        max_drawdown=0.08,
    )
    assert record.rule_id == 1
    assert record.sharpe == 1.2


def test_rule_regime_combo():
    """测试择势组合属性"""
    rule = Rule(
        id=1,
        market_regime="range",
        stock_regime="trend_up",
        best_strategy="test",
        train_return=0.1,
        valid_return=0.1,
        baseline_return=0.05,
        confidence=0.8,
        train_period="2024-01~2024-10",
        stock_code="300001",
        created_at=datetime.now(),
    )
    assert rule.regime_combo == "range+trend_up"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rules_model.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
"""规则数据模型"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class BacktestRecord:
    """回测记录"""
    rule_id: int
    strategy_name: str
    total_return: float
    sharpe: float
    max_drawdown: float
    id: Optional[int] = None


@dataclass
class RuleCreate:
    """规则创建DTO"""
    market_regime: str
    stock_regime: str
    best_strategy: str
    train_return: float
    valid_return: float
    baseline_return: float
    confidence: float
    train_period: str
    stock_code: str


@dataclass
class Rule:
    """策略规则"""
    id: int
    market_regime: str
    stock_regime: str
    best_strategy: str
    train_return: float
    valid_return: float
    baseline_return: float
    confidence: float
    train_period: str
    stock_code: str
    created_at: datetime
    backtest_records: list = field(default_factory=list)
    
    @property
    def regime_combo(self) -> str:
        """择势组合字符串"""
        return f"{self.market_regime}+{self.stock_regime}"
    
    @property
    def excess_return(self) -> float:
        """超额收益"""
        return self.valid_return - self.baseline_return
    
    @classmethod
    def from_create(cls, rule_create: RuleCreate, rule_id: int) -> "Rule":
        """从DTO创建Rule"""
        return cls(
            id=rule_id,
            market_regime=rule_create.market_regime,
            stock_regime=rule_create.stock_regime,
            best_strategy=rule_create.best_strategy,
            train_return=rule_create.train_return,
            valid_return=rule_create.valid_return,
            baseline_return=rule_create.baseline_return,
            confidence=rule_create.confidence,
            train_period=rule_create.train_period,
            stock_code=rule_create.stock_code,
            created_at=datetime.now(),
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_rules_model.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add pixiu/models/rules.py tests/test_rules_model.py
git commit -m "feat: 添加规则数据模型"
```

---

## Task 3: SQLite规则存储

**Files:**
- Create: `pixiu/services/rule_storage.py`
- Test: `tests/test_rule_storage.py`

**Step 1: Write the failing test**

```python
"""测试规则存储"""
import pytest
import tempfile
import os
from datetime import datetime
from pixiu.services.rule_storage import RuleStorage
from pixiu.models.rules import Rule, RuleCreate


@pytest.fixture
def temp_db():
    """临时数据库"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    storage = RuleStorage(path)
    storage.init_db()
    yield storage
    os.unlink(path)


def test_init_db(temp_db):
    """测试数据库初始化"""
    assert temp_db is not None


def test_save_rule(temp_db):
    """测试保存规则"""
    rule_create = RuleCreate(
        market_regime="range",
        stock_regime="trend_up",
        best_strategy="布朗运动策略",
        train_return=0.15,
        valid_return=0.12,
        baseline_return=0.05,
        confidence=0.82,
        train_period="2024-01-01~2024-10-31",
        stock_code="300001",
    )
    rule_id = temp_db.save_rule(rule_create)
    assert rule_id == 1


def test_get_best_strategy(temp_db):
    """测试获取最佳策略"""
    rule_create = RuleCreate(
        market_regime="range",
        stock_regime="trend_up",
        best_strategy="布朗运动策略",
        train_return=0.15,
        valid_return=0.12,
        baseline_return=0.05,
        confidence=0.82,
        train_period="2024-01-01~2024-10-31",
        stock_code="300001",
    )
    temp_db.save_rule(rule_create)
    
    best = temp_db.get_best_strategy("range", "trend_up")
    assert best == "布朗运动策略"


def test_get_best_strategy_not_found(temp_db):
    """测试获取不存在的规则"""
    best = temp_db.get_best_strategy("unknown", "unknown")
    assert best is None


def test_list_rules(temp_db):
    """测试列出所有规则"""
    rule_create1 = RuleCreate(
        market_regime="range",
        stock_regime="trend_up",
        best_strategy="策略1",
        train_return=0.15,
        valid_return=0.12,
        baseline_return=0.05,
        confidence=0.82,
        train_period="2024-01~2024-10",
        stock_code="300001",
    )
    rule_create2 = RuleCreate(
        market_regime="trend_up",
        stock_regime="trend_up",
        best_strategy="策略2",
        train_return=0.20,
        valid_return=0.18,
        baseline_return=0.05,
        confidence=0.88,
        train_period="2024-01~2024-10",
        stock_code="600000",
    )
    temp_db.save_rule(rule_create1)
    temp_db.save_rule(rule_create2)
    
    rules = temp_db.list_rules()
    assert len(rules) == 2


def test_export_rules_json(temp_db):
    """测试导出规则为JSON"""
    rule_create = RuleCreate(
        market_regime="range",
        stock_regime="trend_up",
        best_strategy="布朗运动策略",
        train_return=0.15,
        valid_return=0.12,
        baseline_return=0.05,
        confidence=0.82,
        train_period="2024-01~2024-10",
        stock_code="300001",
    )
    temp_db.save_rule(rule_create)
    
    json_str = temp_db.export_rules("json")
    assert "布朗运动策略" in json_str
    assert "range" in json_str


def test_update_rule_higher_confidence(temp_db):
    """测试更高置信度规则覆盖"""
    rule_create1 = RuleCreate(
        market_regime="range",
        stock_regime="trend_up",
        best_strategy="策略1",
        train_return=0.10,
        valid_return=0.08,
        baseline_return=0.05,
        confidence=0.70,
        train_period="2024-01~2024-06",
        stock_code="300001",
    )
    temp_db.save_rule(rule_create1)
    
    rule_create2 = RuleCreate(
        market_regime="range",
        stock_regime="trend_up",
        best_strategy="策略2",
        train_return=0.15,
        valid_return=0.12,
        baseline_return=0.05,
        confidence=0.85,
        train_period="2024-01~2024-10",
        stock_code="300002",
    )
    temp_db.save_rule(rule_create2)
    
    best = temp_db.get_best_strategy("range", "trend_up")
    assert best == "策略2"
    rules = temp_db.list_rules()
    assert len(rules) == 1  # 应该覆盖，不是新增
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rule_storage.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
"""SQLite规则存储"""
import json
import sqlite3
from typing import Optional, List
from datetime import datetime
from contextlib import contextmanager

from pixiu.models.rules import Rule, RuleCreate, BacktestRecord


class RuleStorage:
    """规则存储服务"""
    
    def __init__(self, db_path: str = "data/pixiu_rules.db"):
        self.db_path = db_path
    
    @contextmanager
    def _get_conn(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def init_db(self):
        """初始化数据库表"""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS strategy_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_regime TEXT NOT NULL,
                    stock_regime TEXT NOT NULL,
                    best_strategy TEXT NOT NULL,
                    train_return REAL NOT NULL,
                    valid_return REAL NOT NULL,
                    baseline_return REAL NOT NULL,
                    confidence REAL NOT NULL,
                    train_period TEXT NOT NULL,
                    stock_code TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(market_regime, stock_regime)
                );
                
                CREATE TABLE IF NOT EXISTS backtest_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id INTEGER NOT NULL,
                    strategy_name TEXT NOT NULL,
                    total_return REAL NOT NULL,
                    sharpe REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    FOREIGN KEY (rule_id) REFERENCES strategy_rules(id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_regime_combo 
                ON strategy_rules(market_regime, stock_regime);
            """)
            conn.commit()
    
    def save_rule(self, rule_create: RuleCreate) -> int:
        """保存规则，如果存在则更新"""
        with self._get_conn() as conn:
            cursor = conn.execute("""
                INSERT INTO strategy_rules 
                (market_regime, stock_regime, best_strategy, train_return, 
                 valid_return, baseline_return, confidence, train_period, stock_code)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(market_regime, stock_regime) 
                DO UPDATE SET
                    best_strategy = excluded.best_strategy,
                    train_return = excluded.train_return,
                    valid_return = excluded.valid_return,
                    baseline_return = excluded.baseline_return,
                    confidence = excluded.confidence,
                    train_period = excluded.train_period,
                    stock_code = excluded.stock_code,
                    created_at = CURRENT_TIMESTAMP
                WHERE excluded.confidence > confidence
                   OR (excluded.confidence = confidence AND excluded.valid_return > valid_return)
            """, (
                rule_create.market_regime,
                rule_create.stock_regime,
                rule_create.best_strategy,
                rule_create.train_return,
                rule_create.valid_return,
                rule_create.baseline_return,
                rule_create.confidence,
                rule_create.train_period,
                rule_create.stock_code,
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_best_strategy(self, market_regime: str, stock_regime: str) -> Optional[str]:
        """获取指定择势组合的最佳策略"""
        with self._get_conn() as conn:
            row = conn.execute("""
                SELECT best_strategy FROM strategy_rules
                WHERE market_regime = ? AND stock_regime = ?
            """, (market_regime, stock_regime)).fetchone()
            return row["best_strategy"] if row else None
    
    def list_rules(self) -> List[Rule]:
        """列出所有规则"""
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT * FROM strategy_rules ORDER BY created_at DESC
            """).fetchall()
            return [self._row_to_rule(row) for row in rows]
    
    def export_rules(self, format: str = "json") -> str:
        """导出规则"""
        rules = self.list_rules()
        if format == "json":
            data = [
                {
                    "market_regime": r.market_regime,
                    "stock_regime": r.stock_regime,
                    "best_strategy": r.best_strategy,
                    "train_return": r.train_return,
                    "valid_return": r.valid_return,
                    "baseline_return": r.baseline_return,
                    "confidence": r.confidence,
                    "train_period": r.train_period,
                    "stock_code": r.stock_code,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rules
            ]
            return json.dumps(data, ensure_ascii=False, indent=2)
        elif format == "csv":
            lines = ["market_regime,stock_regime,best_strategy,train_return,valid_return,confidence"]
            for r in rules:
                lines.append(f"{r.market_regime},{r.stock_regime},{r.best_strategy},{r.train_return},{r.valid_return},{r.confidence}")
            return "\n".join(lines)
        return ""
    
    def _row_to_rule(self, row: sqlite3.Row) -> Rule:
        """数据库行转Rule对象"""
        return Rule(
            id=row["id"],
            market_regime=row["market_regime"],
            stock_regime=row["stock_regime"],
            best_strategy=row["best_strategy"],
            train_return=row["train_return"],
            valid_return=row["valid_return"],
            baseline_return=row["baseline_return"],
            confidence=row["confidence"],
            train_period=row["train_period"],
            stock_code=row["stock_code"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(),
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_rule_storage.py -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add pixiu/services/rule_storage.py tests/test_rule_storage.py
git commit -m "feat: 添加SQLite规则存储服务"
```

---

## Task 4: AI策略推荐服务

**Files:**
- Create: `pixiu/services/strategy_recommender.py`
- Test: `tests/test_strategy_recommender.py`

**Step 1: Write the failing test**

```python
"""测试AI策略推荐"""
import pytest
from unittest.mock import Mock, patch
from pixiu.services.strategy_recommender import StrategyRecommender


@pytest.fixture
def recommender():
    """策略推荐器"""
    return StrategyRecommender()


def test_build_prompt(recommender):
    """测试构建Prompt"""
    prompt = recommender._build_prompt("range", "trend_up")
    assert "震荡" in prompt or "range" in prompt
    assert "趋势" in prompt or "trend" in prompt


def test_parse_response(recommender):
    """测试解析响应"""
    response = '''
    {
        "recommended": ["网格交易策略", "RSI策略", "布朗运动策略"],
        "reasons": {
            "网格交易策略": "适合震荡市场",
            "RSI策略": "超买超卖信号明确"
        },
        "warnings": ["注意波动风险"]
    }
    '''
    result = recommender._parse_response(response)
    assert len(result["recommended"]) == 3
    assert "网格交易策略" in result["recommended"]


def test_parse_response_fallback(recommender):
    """测试解析失败时返回默认策略"""
    response = "无效的JSON"
    result = recommender._parse_response(response)
    assert len(result["recommended"]) > 0


def test_get_cache_key(recommender):
    """测试缓存键"""
    key = recommender._get_cache_key("range", "trend_up")
    assert key == "range:trend_up"


def test_recommend_with_cache(recommender):
    """测试缓存生效"""
    recommender._cache["range:trend_up"] = {
        "recommended": ["策略A"],
        "reasons": {},
        "warnings": [],
    }
    
    result = recommender.recommend("range", "trend_up")
    assert result["recommended"] == ["策略A"]


@patch("pixiu.services.strategy_recommender.ai_service")
def test_recommend_calls_ai(mock_ai, recommender):
    """测试调用AI服务"""
    mock_ai.chat.return_value = '{"recommended": ["策略A"], "reasons": {}, "warnings": []}'
    
    result = recommender.recommend("trend_up", "trend_up")
    assert "recommended" in result
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_strategy_recommender.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
"""AI策略推荐服务"""
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class StrategyRecommender:
    """策略推荐器"""
    
    DEFAULT_STRATEGIES = [
        "网格交易策略",
        "RSI策略",
        "均线交叉策略",
        "趋势强度策略",
        "波动率策略",
        "卡尔曼滤波策略",
        "随机指标策略",
        "最优执行策略",
    ]
    
    REGIME_NAMES = {
        "trend_up": "趋势上涨",
        "trend_down": "趋势下跌",
        "range": "震荡",
    }
    
    def __init__(self, cache_hours: int = 24):
        self._cache: Dict[str, Dict] = {}
        self._cache_hours = cache_hours
    
    def recommend(self, market_regime: str, stock_regime: str) -> Dict:
        """推荐策略
        
        Args:
            market_regime: 大盘状态
            stock_regime: 个股状态
            
        Returns:
            {
                "recommended": ["策略1", "策略2", ...],
                "reasons": {"策略1": "原因", ...},
                "warnings": ["风险点1", ...]
            }
        """
        cache_key = self._get_cache_key(market_regime, stock_regime)
        
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if self._is_cache_valid(cached):
                return cached["data"]
        
        prompt = self._build_prompt(market_regime, stock_regime)
        
        try:
            from pixiu.services.ai_service import ai_service
            response = ai_service.chat(prompt)
            result = self._parse_response(response)
        except Exception:
            result = self._get_fallback_strategies(market_regime, stock_regime)
        
        self._cache[cache_key] = {
            "data": result,
            "timestamp": datetime.now(),
        }
        
        return result
    
    def _build_prompt(self, market_regime: str, stock_regime: str) -> str:
        """构建Prompt"""
        market_name = self.REGIME_NAMES.get(market_regime, market_regime)
        stock_name = self.REGIME_NAMES.get(stock_regime, stock_regime)
        
        return f"""你是量化策略专家。根据以下市场择势状态，推荐最合适的交易策略。

当前择势状态：
- 大盘: {market_name}
- 个股: {stock_name}

可用策略列表：
{', '.join(self.DEFAULT_STRATEGIES)}

请回答：
1. 推荐哪些策略（3-5个），按优先级排序
2. 每个策略为什么适合当前择势组合
3. 预期风险点和注意事项

以JSON格式返回：
{{
  "recommended": ["策略1", "策略2", ...],
  "reasons": {{"策略1": "原因", ...}},
  "warnings": ["风险点1", ...]
}}"""
    
    def _parse_response(self, response: str) -> Dict:
        """解析AI响应"""
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        return self._get_fallback_strategies("range", "range")
    
    def _get_fallback_strategies(self, market_regime: str, stock_regime: str) -> Dict:
        """获取后备策略（当AI调用失败时）"""
        if "trend" in market_regime and "trend" in stock_regime:
            return {
                "recommended": ["趋势强度策略", "均线交叉策略", "最优执行策略"],
                "reasons": {"趋势强度策略": "趋势行情适合趋势跟踪"},
                "warnings": ["注意趋势反转风险"],
            }
        elif "range" in market_regime or "range" in stock_regime:
            return {
                "recommended": ["网格交易策略", "RSI策略", "波动率策略"],
                "reasons": {"网格交易策略": "震荡行情适合网格交易"},
                "warnings": ["注意突破风险"],
            }
        else:
            return {
                "recommended": ["卡尔曼滤波策略", "随机指标策略"],
                "reasons": {},
                "warnings": [],
            }
    
    def _get_cache_key(self, market_regime: str, stock_regime: str) -> str:
        """获取缓存键"""
        return f"{market_regime}:{stock_regime}"
    
    def _is_cache_valid(self, cached: Dict) -> bool:
        """检查缓存是否有效"""
        timestamp = cached.get("timestamp")
        if not timestamp:
            return False
        return datetime.now() - timestamp < timedelta(hours=self._cache_hours)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_strategy_recommender.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add pixiu/services/strategy_recommender.py tests/test_strategy_recommender.py
git commit -m "feat: 添加AI策略推荐服务"
```

---

## Task 5: 规则发现服务

**Files:**
- Create: `pixiu/services/rule_discovery_service.py`
- Test: `tests/test_rule_discovery_service.py`

**Step 1: Write the failing test**

```python
"""测试规则发现服务"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pixiu.services.rule_discovery_service import RuleDiscoveryService


@pytest.fixture
def sample_stock_df():
    """生成测试股票数据"""
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    np.random.seed(42)
    close = 10 + np.cumsum(np.random.randn(200) * 0.02)
    return pd.DataFrame({
        "trade_date": dates,
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": np.random.randint(1000000, 5000000, 200),
    })


@pytest.fixture
def sample_index_df():
    """生成测试指数数据"""
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    np.random.seed(43)
    close = 3000 + np.cumsum(np.random.randn(200) * 10)
    return pd.DataFrame({
        "trade_date": dates,
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": np.random.randint(10000000, 50000000, 200),
    })


def test_split_data(sample_stock_df):
    """测试数据划分"""
    service = RuleDiscoveryService(train_ratio=0.8)
    train_df, valid_df = service.split_data(sample_stock_df)
    
    assert len(train_df) == 160  # 80%
    assert len(valid_df) == 40   # 20%


def test_group_by_regime_combo(sample_stock_df, sample_index_df):
    """测试按择势组合分组"""
    service = RuleDiscoveryService()
    
    from pixiu.analysis.regime_timeline import RegimeTimelineAnalyzer
    analyzer = RegimeTimelineAnalyzer(window=30)
    
    stock_timeline = analyzer.analyze_timeline(sample_stock_df)
    index_timeline = analyzer.analyze_timeline(sample_index_df)
    
    combos = service.group_by_regime_combo(stock_timeline, index_timeline)
    
    assert isinstance(combos, list)
    if len(combos) > 0:
        assert "date" in combos[0]
        assert "market_regime" in combos[0]
        assert "stock_regime" in combos[0]


def test_calc_baseline_return(sample_stock_df):
    """测试基准收益计算"""
    service = RuleDiscoveryService()
    train_df, valid_df = service.split_data(sample_stock_df)
    
    baseline = service.calc_baseline_return(valid_df)
    
    assert isinstance(baseline, float)


def test_calc_confidence(sample_stock_df):
    """测试置信度计算"""
    service = RuleDiscoveryService()
    
    confidence = service.calc_confidence(
        valid_return=0.15,
        baseline_return=0.05,
    )
    
    assert 0 <= confidence <= 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rule_discovery_service.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
"""规则发现服务"""
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from pixiu.analysis.regime_timeline import RegimeTimelineAnalyzer
from pixiu.services.strategy_recommender import StrategyRecommender
from pixiu.services.backtest_service import BacktestService
from pixiu.models.rules import RuleCreate


@dataclass
class RegimeComboSegment:
    """择势组合时间段"""
    start_date: str
    end_date: str
    market_regime: str
    stock_regime: str
    data_slice: pd.DataFrame


@dataclass
class DiscoveryResult:
    """发现结果"""
    rules: List[RuleCreate]
    train_period: str
    valid_period: str
    total_combos: int
    discovered_rules: int


class RuleDiscoveryService:
    """规则发现服务"""
    
    def __init__(self, train_ratio: float = 0.8, window: int = 60):
        self.train_ratio = train_ratio
        self.window = window
        self.timeline_analyzer = RegimeTimelineAnalyzer(window=window)
        self.recommender = StrategyRecommender()
        self.backtest_service = BacktestService()
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """划分训练集和验证集"""
        split_idx = int(len(df) * self.train_ratio)
        return df.iloc[:split_idx], df.iloc[split_idx:]
    
    def group_by_regime_combo(
        self, 
        stock_timeline: Dict, 
        index_timeline: Dict
    ) -> List[Dict]:
        """按择势组合分组
        
        Returns:
            [{"date": ..., "market_regime": ..., "stock_regime": ...}, ...]
        """
        combos = []
        
        stock_segments = stock_timeline.get("segments", [])
        index_segments = index_timeline.get("segments", [])
        
        for stock_seg in stock_segments:
            for index_seg in index_segments:
                start = max(stock_seg["start"], index_seg["start"])
                end = min(stock_seg["end"], index_seg["end"])
                
                if start <= end:
                    combos.append({
                        "date": start,
                        "end_date": end,
                        "market_regime": index_seg["regime"],
                        "stock_regime": stock_seg["regime"],
                    })
        
        return combos
    
    def discover_rules(
        self,
        stock_df: pd.DataFrame,
        index_df: pd.DataFrame,
        stock_code: str,
    ) -> DiscoveryResult:
        """发现规则"""
        train_df, valid_df = self.split_data(stock_df)
        train_index_df, valid_index_df = self.split_data(index_df)
        
        stock_timeline = self.timeline_analyzer.analyze_timeline(train_df)
        index_timeline = self.timeline_analyzer.analyze_timeline(train_index_df)
        
        combos = self.group_by_regime_combo(stock_timeline, index_timeline)
        
        rules: List[RuleCreate] = []
        combo_strategy_map: Dict[str, str] = {}
        
        for combo in combos:
            combo_key = f"{combo['market_regime']}+{combo['stock_regime']}"
            
            if combo_key in combo_strategy_map:
                continue
            
            recommended = self.recommender.recommend(
                combo["market_regime"],
                combo["stock_regime"],
            )
            
            best_strategy = None
            best_return = float("-inf")
            
            for strategy_name in recommended.get("recommended", []):
                try:
                    result = self.backtest_service.run_backtest(
                        train_df,
                        strategy_name=strategy_name,
                    )
                    if result and result.get("total_return", 0) > best_return:
                        best_return = result["total_return"]
                        best_strategy = strategy_name
                except Exception:
                    continue
            
            if best_strategy:
                combo_strategy_map[combo_key] = best_strategy
                
                baseline = self.calc_baseline_return(train_df)
                confidence = self.calc_confidence(best_return, baseline)
                
                rules.append(RuleCreate(
                    market_regime=combo["market_regime"],
                    stock_regime=combo["stock_regime"],
                    best_strategy=best_strategy,
                    train_return=best_return,
                    valid_return=0.0,
                    baseline_return=baseline,
                    confidence=confidence,
                    train_period=f"{train_df.iloc[0]['trade_date']}~{train_df.iloc[-1]['trade_date']}",
                    stock_code=stock_code,
                ))
        
        train_period = f"{train_df.iloc[0]['trade_date']}~{train_df.iloc[-1]['trade_date']}"
        valid_period = f"{valid_df.iloc[0]['trade_date']}~{valid_df.iloc[-1]['trade_date']}"
        
        return DiscoveryResult(
            rules=rules,
            train_period=train_period,
            valid_period=valid_period,
            total_combos=len(combos),
            discovered_rules=len(rules),
        )
    
    def calc_baseline_return(self, df: pd.DataFrame) -> float:
        """计算基准收益（买入持有）"""
        if df.empty or len(df) < 2:
            return 0.0
        
        start_price = df.iloc[0]["close"]
        end_price = df.iloc[-1]["close"]
        
        return (end_price - start_price) / start_price
    
    def calc_confidence(
        self, 
        valid_return: float, 
        baseline_return: float
    ) -> float:
        """计算置信度"""
        if baseline_return == 0:
            return 0.5
        
        excess = valid_return - baseline_return
        confidence = excess / abs(baseline_return) * 2
        
        return min(1.0, max(0.0, confidence))
    
    def validate_rule(
        self,
        rule: RuleCreate,
        valid_df: pd.DataFrame,
    ) -> float:
        """验证规则在验证集的表现"""
        try:
            result = self.backtest_service.run_backtest(
                valid_df,
                strategy_name=rule.best_strategy,
            )
            return result.get("total_return", 0.0) if result else 0.0
        except Exception:
            return 0.0
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_rule_discovery_service.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add pixiu/services/rule_discovery_service.py tests/test_rule_discovery_service.py
git commit -m "feat: 添加规则发现服务"
```

---

## Task 6: ECharts K线组件

**Files:**
- Create: `pixiu/components/echarts_kline.py`
- Modify: `pixiu/pages/home.py`

**Step 1: Write the component**

```python
"""ECharts K线图组件"""
import reflex as rx
from typing import Dict, List, Any
import json


def echarts_kline(
    ohlcv_data: List[Dict],
    timeline_data: List[Dict],
    height: str = "400px",
) -> rx.Component:
    """ECharts K线图
    
    Args:
        ohlcv_data: K线数据 [{"date": "2024-01-01", "open": 10, "high": 11, "low": 9, "close": 10.5}, ...]
        timeline_data: 择势时间线 [{"start": "2024-01-01", "end": "2024-03-01", "regime": "trend_up"}, ...]
        height: 图表高度
        
    Returns:
        Reflex组件
    """
    kline_data = [
        [d["open"], d["close"], d["low"], d["high"]]
        for d in ohlcv_data
    ]
    dates = [d["date"] for d in ohlcv_data]
    
    mark_areas = []
    for seg in timeline_data:
        color = {
            "trend_up": "rgba(34, 197, 94, 0.2)",
            "trend_down": "rgba(239, 68, 68, 0.2)",
            "range": "rgba(234, 179, 8, 0.2)",
        }.get(seg["regime"], "rgba(156, 163, 175, 0.2)")
        
        mark_areas.append({
            "xAxis": seg["start"],
            "yAxis": "min",
        }, {
            "xAxis": seg["end"],
            "yAxis": "max",
            "itemStyle": {"color": color},
        })
    
    option = {
        "title": {"text": "K线图 + 择势时间线"},
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "cross"},
        },
        "legend": {"data": ["K线", "择势"]},
        "grid": {
            "left": "10%",
            "right": "10%",
            "bottom": "15%",
        },
        "xAxis": {
            "type": "category",
            "data": dates,
            "scale": True,
            "boundaryGap": False,
            "axisLine": {"onZero": False},
            "splitLine": {"show": False},
            "min": "dataMin",
            "max": "dataMax",
        },
        "yAxis": {
            "scale": True,
            "splitArea": {"show": True},
        },
        "dataZoom": [
            {"type": "inside", "start": 50, "end": 100},
            {"show": True, "type": "slider", "top": "90%", "start": 50, "end": 100},
        ],
        "series": [
            {
                "name": "K线",
                "type": "candlestick",
                "data": kline_data,
                "markArea": {
                    "data": mark_areas,
                },
            }
        ],
    }
    
    option_json = json.dumps(option, ensure_ascii=False)
    
    return rx.box(
        rx.html(
            f"""
            <div id="echarts-kline" style="width:100%;height:{height}"></div>
            <script>
                var chart = echarts.init(document.getElementById('echarts-kline'));
                chart.setOption({option_json});
            </script>
            """,
        ),
        width="100%",
        height=height,
    )


def echarts_dual_kline(
    index_data: List[Dict],
    index_timeline: List[Dict],
    stock_data: List[Dict],
    stock_timeline: List[Dict],
    height: str = "600px",
) -> rx.Component:
    """双K线图（大盘+个股）
    
    Args:
        index_data: 大盘K线数据
        index_timeline: 大盘择势时间线
        stock_data: 个股K线数据
        stock_timeline: 个股择势时间线
        height: 总高度
        
    Returns:
        Reflex组件
    """
    return rx.vstack(
        rx.text("大盘择势", font_weight="bold", font_size="lg"),
        echarts_kline(index_data, index_timeline, height="300px"),
        rx.divider(),
        rx.text("个股择势", font_weight="bold", font_size="lg"),
        echarts_kline(stock_data, stock_timeline, height="300px"),
        spacing="4",
        width="100%",
    )
```

**Step 2: Add to home.py imports**

Modify: `pixiu/pages/home.py`

```python
from pixiu.components.echarts_kline import echarts_kline, echarts_dual_kline
```

**Step 3: Commit**

```bash
git add pixiu/components/echarts_kline.py pixiu/pages/home.py
git commit -m "feat: 添加ECharts K线图组件"
```

---

## Task 7: 规则表格组件

**Files:**
- Create: `pixiu/components/rule_table.py`

**Step 1: Write the component**

```python
"""规则表格组件"""
import reflex as rx
from typing import List
from pixiu.models.rules import Rule


def rule_table(rules: List[Rule]) -> rx.Component:
    """规则表格
    
    Args:
        rules: 规则列表
        
    Returns:
        Reflex组件
    """
    return rx.table.root(
        rx.table.header(
            rx.table.row(
                rx.table.column_header("择势组合"),
                rx.table.column_header("最佳策略"),
                rx.table.column_header("训练收益"),
                rx.table.column_header("验证收益"),
                rx.table.column_header("超额收益"),
                rx.table.column_header("置信度"),
            ),
        ),
        rx.table.body(
            *[
                rx.table.row(
                    rx.table.cell(rule.regime_combo),
                    rx.table.cell(rule.best_strategy),
                    rx.table.cell(f"{rule.train_return:.2%}"),
                    rx.table.cell(f"{rule.valid_return:.2%}"),
                    rx.table.cell(
                        f"{rule.excess_return:.2%}",
                        color=rx.cond(rule.excess_return > 0, "green.400", "red.400"),
                    ),
                    rx.table.cell(f"{rule.confidence:.2f}"),
                )
                for rule in rules
            ],
        ),
    )


def rule_card(rule: Rule) -> rx.Component:
    """规则卡片"""
    return rx.box(
        rx.hstack(
            rx.badge(rule.regime_combo, color_scheme="cyan"),
            rx.text(rule.best_strategy, font_weight="bold"),
            rx.spacer(),
            rx.text(f"+{rule.valid_return:.1%}", color="green.400"),
        ),
        rx.hstack(
            rx.text(f"训练: {rule.train_return:.1%}", font_size="sm", color="gray.400"),
            rx.text(f"基准: {rule.baseline_return:.1%}", font_size="sm", color="gray.400"),
            rx.text(f"置信度: {rule.confidence:.2f}", font_size="sm", color="gray.400"),
        ),
        padding="1rem",
        border="1px solid gray.700",
        border_radius="md",
        margin_bottom="0.5rem",
    )


def rules_list(rules: List[Rule]) -> rx.Component:
    """规则列表"""
    return rx.vstack(
        *[rule_card(rule) for rule in rules],
        spacing="2",
        width="100%",
    )
```

**Step 2: Commit**

```bash
git add pixiu/components/rule_table.py
git commit -m "feat: 添加规则表格组件"
```

---

## Task 8: 状态管理更新

**Files:**
- Modify: `pixiu/state.py`

**Step 1: Add new state variables**

Add to `pixiu/state.py` State class:

```python
# 规则发现相关
train_ratio: float = 0.8
discovered_rules: List[Dict] = []
validation_result: Dict = {}
matched_index_code: str = ""
matched_index_name: str = ""
is_discovering: bool = False

def set_train_ratio(self, value: list) -> None:
    """设置训练比例"""
    self.train_ratio = value[0] if value else 0.8

async def start_rule_discovery(self) -> None:
    """开始规则发现"""
    if not self.stock_data:
        return
    
    self.is_discovering = True
    
    try:
        from pixiu.analysis.regime_matcher import match_index, get_index_name
        from pixiu.services.rule_discovery_service import RuleDiscoveryService
        from pixiu.services.rule_storage import RuleStorage
        import pandas as pd
        
        self.matched_index_code = match_index(self.selected_stock)
        self.matched_index_name = get_index_name(self.matched_index_code)
        
        stock_df = pd.DataFrame(self.stock_data)
        
        from pixiu.services.data_service import DataService
        data_service = DataService()
        index_df = data_service.fetch_index_history(
            self.matched_index_code,
            self.start_date,
            self.end_date,
        )
        
        if index_df.empty:
            self.is_discovering = False
            return
        
        discovery_service = RuleDiscoveryService(train_ratio=self.train_ratio)
        result = discovery_service.discover_rules(
            stock_df,
            index_df,
            self.selected_stock,
        )
        
        storage = RuleStorage()
        storage.init_db()
        
        self.discovered_rules = []
        for rule_create in result.rules:
            rule_id = storage.save_rule(rule_create)
            self.discovered_rules.append({
                "id": rule_id,
                "market_regime": rule_create.market_regime,
                "stock_regime": rule_create.stock_regime,
                "best_strategy": rule_create.best_strategy,
                "train_return": rule_create.train_return,
                "valid_return": rule_create.valid_return,
                "baseline_return": rule_create.baseline_return,
                "confidence": rule_create.confidence,
                "regime_combo": f"{rule_create.market_regime}+{rule_create.stock_regime}",
            })
        
        train_df, valid_df = discovery_service.split_data(stock_df)
        
        total_valid_return = 0.0
        for rule in result.rules:
            valid_return = discovery_service.validate_rule(rule, valid_df)
            total_valid_return += valid_return
        
        avg_valid_return = total_valid_return / len(result.rules) if result.rules else 0
        baseline = discovery_service.calc_baseline_return(valid_df)
        
        self.validation_result = {
            "rule_guided_return": avg_valid_return,
            "baseline_return": baseline,
            "excess_return": avg_valid_return - baseline,
        }
        
    except Exception as e:
        print(f"Rule discovery error: {e}")
    
    finally:
        self.is_discovering = False

async def export_rules(self) -> None:
    """导出规则"""
    from pixiu.services.rule_storage import RuleStorage
    
    storage = RuleStorage()
    json_str = storage.export_rules("json")
    
    return rx.download(data=json_str, filename="pixiu_rules.json")
```

**Step 2: Commit**

```bash
git add pixiu/state.py
git commit -m "feat: 添加规则发现状态管理"
```

---

## Task 9: 首页集成

**Files:**
- Modify: `pixiu/pages/home.py`

**Step 1: Add new sections to home page**

Add after the existing sections in `pixiu/pages/home.py`:

```python
def step_rule_discovery() -> rx.Component:
    """规则发现区域"""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text("规则发现", font_size="lg", font_weight="bold"),
                rx.spacer(),
                rx.hstack(
                    rx.text("训练比例:", font_size="sm"),
                    rx.text(f"{State.train_ratio:.0%}", font_size="sm"),
                    rx.slider(
                        value=[State.train_ratio],
                        on_change=State.set_train_ratio,
                        min=0.5,
                        max=0.9,
                        step=0.1,
                        width="150px",
                    ),
                ),
            ),
            rx.cond(
                State.matched_index_name != "",
                rx.hstack(
                    rx.text("自动匹配大盘:", font_size="sm", color="gray.400"),
                    rx.badge(State.matched_index_name, color_scheme="cyan"),
                ),
            ),
            rx.button(
                "开始规则发现",
                on_click=State.start_rule_discovery,
                is_loading=State.is_discovering,
                color_scheme="cyan",
                size="lg",
            ),
            spacing="3",
        ),
        padding="1.5rem",
        border="1px solid gray.700",
        border_radius="lg",
        width="100%",
    )


def step_discovered_rules() -> rx.Component:
    """发现的规则展示"""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text("发现的规则", font_size="lg", font_weight="bold"),
                rx.badge(f"共{len(State.discovered_rules)}条", color_scheme="green"),
                rx.spacer(),
                rx.button(
                    "导出规则",
                    on_click=State.export_rules,
                    variant="outline",
                    size="sm",
                ),
            ),
            rx.table.root(
                rx.table.header(
                    rx.table.row(
                        rx.table.column_header("择势组合"),
                        rx.table.column_header("最佳策略"),
                        rx.table.column_header("训练收益"),
                        rx.table.column_header("置信度"),
                    ),
                ),
                rx.table.body(
                    *[
                        rx.table.row(
                            rx.table.cell(rule["regime_combo"]),
                            rx.table.cell(rule["best_strategy"]),
                            rx.table.cell(f"{rule['train_return']:.2%}"),
                            rx.table.cell(f"{rule['confidence']:.2f}"),
                        )
                        for rule in State.discovered_rules
                    ],
                ),
            ),
            spacing="3",
        ),
        padding="1.5rem",
        border="1px solid gray.700",
        border_radius="lg",
        width="100%",
    )


def step_validation_result() -> rx.Component:
    """验证结果"""
    return rx.box(
        rx.vstack(
            rx.text("规则验证结果", font_size="lg", font_weight="bold"),
            rx.hstack(
                rx.stat.root(
                    rx.stat.label("规则指导收益"),
                    rx.stat.number(f"{State.validation_result['rule_guided_return']:.2%}"),
                ),
                rx.stat.root(
                    rx.stat.label("基准收益"),
                    rx.stat.number(f"{State.validation_result['baseline_return']:.2%}"),
                ),
                rx.stat.root(
                    rx.stat.label("超额收益"),
                    rx.stat.number(
                        f"{State.validation_result['excess_return']:.2%}",
                        color=rx.cond(
                            State.validation_result["excess_return"] > 0,
                            "green.400",
                            "red.400",
                        ),
                    ),
                ),
            ),
            spacing="3",
        ),
        padding="1.5rem",
        border="1px solid gray.700",
        border_radius="lg",
        width="100%",
    )
```

**Step 2: Update index() function**

Update the `index()` function to include new sections:

```python
def index() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.heading("Pixiu 策略规则发现", size="lg"),
            step_indicator(),
            step_market_selection(),
            rx.cond(State.current_step >= 1, step_stock_search()),
            rx.cond(State.current_step >= 2, step_regime_analysis()),
            rx.cond(State.current_step >= 3, step_rule_discovery()),
            rx.cond(State.discovered_rules.length() > 0, step_discovered_rules()),
            rx.cond(State.validation_result != {}, step_validation_result()),
            spacing="4",
            align="stretch",
        ),
        max_width="1200px",
        margin="0 auto",
        padding="2rem",
    )
```

**Step 3: Commit**

```bash
git add pixiu/pages/home.py
git commit -m "feat: 集成规则发现功能到首页"
```

---

## Task 10: 初始化数据库

**Files:**
- Create: `data/.gitkeep`

**Step 1: Create data directory**

```bash
mkdir -p data
touch data/.gitkeep
```

**Step 2: Add to .gitignore**

Add to `.gitignore`:

```
data/*.db
```

**Step 3: Commit**

```bash
git add data/.gitkeep .gitignore
git commit -m "chore: 创建数据目录"
```

---

## Task 11: 集成测试

**Files:**
- Create: `tests/test_rule_discovery_integration.py`

**Step 1: Write integration test**

```python
"""规则发现集成测试"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime

from pixiu.analysis.regime_matcher import match_index
from pixiu.services.rule_discovery_service import RuleDiscoveryService
from pixiu.services.rule_storage import RuleStorage
from pixiu.services.strategy_recommender import StrategyRecommender


@pytest.fixture
def sample_data():
    """生成测试数据"""
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    np.random.seed(42)
    
    stock_close = 10 + np.cumsum(np.random.randn(200) * 0.02)
    stock_df = pd.DataFrame({
        "trade_date": dates,
        "open": stock_close * 0.99,
        "high": stock_close * 1.01,
        "low": stock_close * 0.98,
        "close": stock_close,
        "volume": np.random.randint(1000000, 5000000, 200),
    })
    
    index_close = 3000 + np.cumsum(np.random.randn(200) * 10)
    index_df = pd.DataFrame({
        "trade_date": dates,
        "open": index_close * 0.99,
        "high": index_close * 1.01,
        "low": index_close * 0.98,
        "close": index_close,
        "volume": np.random.randint(10000000, 50000000, 200),
    })
    
    return stock_df, index_df


def test_full_discovery_flow(sample_data):
    """测试完整的规则发现流程"""
    stock_df, index_df = sample_data
    
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    try:
        storage = RuleStorage(db_path)
        storage.init_db()
        
        discovery = RuleDiscoveryService(train_ratio=0.8)
        result = discovery.discover_rules(stock_df, index_df, "300001")
        
        assert result.total_combos > 0
        assert len(result.rules) > 0
        
        for rule in result.rules:
            rule_id = storage.save_rule(rule)
            assert rule_id > 0
        
        saved_rules = storage.list_rules()
        assert len(saved_rules) > 0
        
    finally:
        os.unlink(db_path)


def test_match_index_integration():
    """测试大盘匹配集成"""
    assert match_index("300001") == "sz399006"
    assert match_index("688001") == "sh000688"
    assert match_index("600000") == "sh000001"


def test_recommender_integration():
    """测试策略推荐集成"""
    recommender = StrategyRecommender()
    
    result = recommender.recommend("range", "trend_up")
    
    assert "recommended" in result
    assert len(result["recommended"]) > 0
```

**Step 2: Run integration test**

Run: `pytest tests/test_rule_discovery_integration.py -v`
Expected: PASS (3 tests)

**Step 3: Commit**

```bash
git add tests/test_rule_discovery_integration.py
git commit -m "test: 添加规则发现集成测试"
```

---

## Task 12: 最终验证

**Step 1: Run all tests**

```bash
pytest tests/ -v
```

Expected: All tests pass

**Step 2: Run the app**

```bash
cd /home/cunxiao/codes/pixiu
reflex run
```

**Step 3: Manual testing checklist**

- [ ] 选择股票后自动匹配正确的大盘指数
- [ ] 点击"开始规则发现"后显示加载状态
- [ ] 规则发现完成后显示规则表格
- [ ] 验证结果显示超额收益
- [ ] 点击"导出规则"下载JSON文件
- [ ] ECharts图表正确显示K线和择势色带

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: 策略规则发现系统完成"
```

---

## Summary

实现计划完成，共12个任务：

1. 大盘指数智能匹配
2. 规则数据模型
3. SQLite规则存储
4. AI策略推荐服务
5. 规则发现服务
6. ECharts K线组件
7. 规则表格组件
8. 状态管理更新
9. 首页集成
10. 初始化数据库
11. 集成测试
12. 最终验证

预计实现时间：4-6小时
