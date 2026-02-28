import pytest
import pandas as pd
from pixiu.services.data_service import DataService
from pixiu.services.database import Database

@pytest.fixture
def data_service():
    db = Database(":memory:")
    return DataService(db, use_mock=False)

def test_fetch_from_baostock_returns_dataframe(data_service):
    """测试baostock获取数据"""
    df = data_service._fetch_from_baostock(
        "000001", 
        "A股",
        "2024-01-01",
        "2024-01-31"
    )
    assert isinstance(df, pd.DataFrame)
    
def test_fetch_from_baostock_with_none_dates(data_service):
    """测试不传日期时使用默认值"""
    df = data_service._fetch_from_baostock("000001", "A股")
    assert isinstance(df, pd.DataFrame)

def test_fetch_stock_history_accepts_date_params(data_service):
    """测试fetch_stock_history接受时间参数"""
    import asyncio
    df = asyncio.run(data_service.fetch_stock_history(
        "000001", 
        "A股",
        "2024-01-01",
        "2024-01-31"
    ))
    assert isinstance(df, pd.DataFrame)
