"""测试混合数据源策略"""

import pytest
from pixiu.services.data_service import DataService
from pixiu.services.database import Database
from pixiu.config import config


def test_data_source_priority_config():
    """测试数据源优先级配置"""
    assert "A股" in config.data_source_priority
    assert config.data_source_priority["A股"][0] == "baostock"
    assert config.data_source_priority["港股"][0] == "akshare"
    assert config.data_source_priority["美股"][0] == "akshare"
    assert config.data_source_priority["index"][0] == "baostock"


def test_a股_fallback_chain():
    """测试A股fallback链: baostock -> akshare -> mock"""
    priority = config.data_source_priority["A股"]
    assert priority == ["baostock", "akshare", "mock"]


def test_港股_fallback_chain():
    """测试港股fallback链: akshare -> mock"""
    priority = config.data_source_priority["港股"]
    assert priority == ["akshare", "mock"]


@pytest.mark.asyncio
async def test_fallback_to_mock_on_failure():
    """测试所有数据源失败时fallback到mock"""
    db = Database(":memory:")
    ds = DataService(db, use_mock=True)
    
    df = await ds.fetch_stock_history("000001", "A股")
    
    assert df is not None
    assert not df.empty
    assert "close" in df.columns


@pytest.mark.asyncio
async def test_fetch_returns_dataframe():
    """测试获取数据返回DataFrame"""
    db = Database(":memory:")
    ds = DataService(db, use_mock=True)
    
    df = await ds.fetch_stock_history("00700", "港股")
    
    assert df is not None
    assert isinstance(df, type(ds._generate_mock_history("00700")))
