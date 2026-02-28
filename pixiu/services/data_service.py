"""数据获取服务模块"""

import asyncio
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd
import socket
import urllib3

try:
    import akshare as ak
except ImportError:
    ak = None

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

socket.setdefaulttimeout(30)

from pixiu.services.database import Database
from pixiu.models.stock import Stock, DailyQuote


class DataService:
    """数据获取服务"""
    
    MARKET_MAP = {
        "A股": "stock_zh_a",
        "港股": "stock_hk", 
        "美股": "stock_us"
    }
    
    def __init__(self, db: Database):
        self.db = db
    
    async def search_stocks(self, keyword: str, market: str = "A股") -> List[Stock]:
        """搜索股票"""
        if ak is None:
            logger.warning("akshare not installed, returning empty list")
            return []
        
        max_retries = 3
        retry_delay = 3
        
        for attempt in range(max_retries):
            try:
                if market == "A股":
                    df = await asyncio.wait_for(
                        asyncio.to_thread(ak.stock_zh_a_spot_em),
                        timeout=30
                    )
                    filtered = df[df['名称'].str.contains(keyword, na=False)]
                    return [
                        Stock(code=str(row['代码']), name=str(row['名称']), market="A股")
                        for _, row in filtered.head(20).iterrows()
                    ]
                elif market == "港股":
                    df = await asyncio.wait_for(
                        asyncio.to_thread(ak.stock_hk_spot_em),
                        timeout=30
                    )
                    filtered = df[df['名称'].str.contains(keyword, na=False)]
                    return [
                        Stock(code=str(row['代码']), name=str(row['名称']), market="港股")
                        for _, row in filtered.head(20).iterrows()
                    ]
                elif market == "美股":
                    df = await asyncio.wait_for(
                        asyncio.to_thread(ak.stock_us_spot_em),
                        timeout=30
                    )
                    filtered = df[df['名称'].str.contains(keyword, na=False)]
                    return [
                        Stock(code=str(row['代码']), name=str(row['名称']), market="美股")
                        for _, row in filtered.head(20).iterrows()
                    ]
            except asyncio.TimeoutError:
                logger.warning(f"搜索{market}股票超时 (尝试 {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    raise TimeoutError(f"搜索{market}股票超时，请稍后重试或切换到其他市场")
            except (ConnectionError, urllib3.exceptions.HTTPError, urllib3.exceptions.TimeoutError) as e:
                logger.warning(f"网络错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    raise ConnectionError(f"网络连接失败，请检查网络后重试")
            except Exception as e:
                logger.warning(f"搜索股票失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"搜索股票失败，已达到最大重试次数: {e}")
                    raise e
        return []
    
    async def fetch_stock_history(
        self, 
        code: str, 
        market: str = "A股",
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """获取股票历史数据"""
        if ak is None:
            logger.warning("akshare not installed")
            return pd.DataFrame()
        
        try:
            if not end_date:
                end_date = datetime.now().strftime("%Y%m%d")
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y%m%d")
            
            if market == "A股":
                df = await asyncio.to_thread(
                    ak.stock_zh_a_hist,
                    symbol=code.split('.')[0],
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"
                )
                df = df.rename(columns={
                    '日期': 'trade_date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume',
                    '成交额': 'amount',
                    '换手率': 'turnover_rate'
                })
            elif market == "港股":
                df = await asyncio.to_thread(
                    ak.stock_hk_daily_hist,
                    symbol=code,
                    adjust="qfq"
                )
                df = df.rename(columns={
                    '日期': 'trade_date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume',
                    '成交额': 'amount'
                })
            elif market == "美股":
                df = await asyncio.to_thread(
                    ak.stock_us_hist,
                    symbol=code,
                    adjust="qfq"
                )
                df = df.rename(columns={
                    '日期': 'trade_date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume'
                })
            
            df['code'] = code
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            return df.sort_values('trade_date')
            
        except Exception as e:
            logger.error(f"获取股票历史数据失败: {e}")
            return pd.DataFrame()
    
    async def download_and_save(
        self, 
        code: str, 
        market: str = "A股",
        force_full: bool = False
    ) -> Tuple[bool, int]:
        """下载并保存股票数据"""
        try:
            last_update = await self.db.get_last_update(market)
            
            if last_update and not force_full:
                start_date = (datetime.fromisoformat(last_update) + timedelta(days=1)).strftime("%Y%m%d")
            else:
                start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y%m%d")
            
            df = await self.fetch_stock_history(code, market, start_date)
            
            if df.empty:
                return False, 0
            
            quotes = []
            for _, row in df.iterrows():
                trade_date = row['trade_date']
                if hasattr(trade_date, 'date'):
                    trade_date = trade_date.date()
                quotes.append(DailyQuote(
                    code=row['code'],
                    trade_date=trade_date,
                    open=float(row['open']) if pd.notna(row.get('open')) else 0.0,
                    high=float(row['high']) if pd.notna(row.get('high')) else 0.0,
                    low=float(row['low']) if pd.notna(row.get('low')) else 0.0,
                    close=float(row['close']) if pd.notna(row.get('close')) else 0.0,
                    volume=float(row['volume']) if pd.notna(row.get('volume')) else 0.0,
                    amount=float(row.get('amount', 0)) if pd.notna(row.get('amount')) else 0.0,
                    turnover_rate=float(row.get('turnover_rate', 0)) if pd.notna(row.get('turnover_rate')) else 0.0
                ))
            
            await self.db.insert_quotes(quotes)
            await self.db.log_update(market, len(quotes))
            
            return True, len(quotes)
            
        except Exception as e:
            logger.error(f"下载保存数据失败: {e}")
            return False, 0
    
    async def get_cached_data(self, code: str) -> pd.DataFrame:
        """获取缓存的数据"""
        quotes = await self.db.get_quotes(code)
        if not quotes:
            return pd.DataFrame()
        
        df = pd.DataFrame([
            {
                'trade_date': q.trade_date,
                'open': q.open,
                'high': q.high,
                'low': q.low,
                'close': q.close,
                'volume': q.volume,
                'amount': q.amount,
                'turnover_rate': q.turnover_rate
            }
            for q in quotes
        ])
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        return df.set_index('trade_date')
