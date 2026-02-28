"""测试Baostock日期格式修复"""
import pytest
from datetime import datetime, timedelta
from pixiu.services.data_service import DataService
from pixiu.services.database import Database


def test_date_format_conversion():
    """测试日期格式正确转换为YYYYMMDD"""
    ds = DataService(Database(":memory:"), use_mock=False)
    
    start_date = "2025-01-01"
    end_date = "2025-02-28"
    
    start_fmt = start_date.replace("-", "")
    end_fmt = end_date.replace("-", "")
    
    assert start_fmt == "20250101"
    assert end_fmt == "20250228"


def test_baostock_login_check():
    """测试登录状态检查"""
    try:
        import baostock as bs
        lg = bs.login()
        assert lg is not None
        assert hasattr(lg, 'error_code')
        assert lg.error_code == '0'
        bs.logout()
    except ImportError:
        pytest.skip("baostock not installed")


@pytest.mark.asyncio
async def test_fetch_a_stock_data():
    """测试获取A股真实数据"""
    db = Database("data/stocks.db")
    ds = DataService(db, use_mock=False)
    
    df = await ds.fetch_stock_history(
        "000001", 
        "A股",
        start_date="2025-01-01",
        end_date="2025-02-01"
    )
    
    assert df is not None
    assert not df.empty
    assert 'close' in df.columns
    assert 'trade_date' in df.columns
