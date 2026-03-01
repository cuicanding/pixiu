"""数据获取服务模块"""

import asyncio
from typing import List, Tuple, Optional
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import socket

try:
    import akshare as ak
except ImportError:
    ak = None

try:
    import baostock as bs
except ImportError:
    bs = None

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

socket.setdefaulttimeout(30)

from pixiu.services.database import Database
from pixiu.models.stock import Stock, DailyQuote
from pixiu.config import config


MOCK_STOCKS = {
    "A股": [
        Stock(code="000001", name="平安银行", market="A股"),
        Stock(code="000002", name="万科A", market="A股"),
        Stock(code="000333", name="美的集团", market="A股"),
        Stock(code="000651", name="格力电器", market="A股"),
        Stock(code="000858", name="五粮液", market="A股"),
        Stock(code="002415", name="海康威视", market="A股"),
        Stock(code="002594", name="比亚迪", market="A股"),
        Stock(code="300750", name="宁德时代", market="A股"),
        Stock(code="600000", name="浦发银行", market="A股"),
        Stock(code="600036", name="招商银行", market="A股"),
        Stock(code="600519", name="贵州茅台", market="A股"),
        Stock(code="601318", name="中国平安", market="A股"),
        Stock(code="601398", name="工商银行", market="A股"),
    ],
    "港股": [
        Stock(code="00700", name="腾讯控股", market="港股"),
        Stock(code="09988", name="阿里巴巴", market="港股"),
        Stock(code="03690", name="美团", market="港股"),
        Stock(code="09999", name="网易", market="港股"),
        Stock(code="01810", name="小米集团", market="港股"),
        Stock(code="02318", name="中国平安", market="港股"),
        Stock(code="01299", name="友邦保险", market="港股"),
        Stock(code="00005", name="汇丰控股", market="港股"),
        Stock(code="00941", name="中国移动", market="港股"),
        Stock(code="00883", name="中国海洋石油", market="港股"),
    ],
    "美股": [
        Stock(code="AAPL", name="苹果", market="美股"),
        Stock(code="MSFT", name="微软", market="美股"),
        Stock(code="GOOGL", name="谷歌", market="美股"),
        Stock(code="AMZN", name="亚马逊", market="美股"),
        Stock(code="NVDA", name="英伟达", market="美股"),
        Stock(code="META", name="Meta", market="美股"),
        Stock(code="TSLA", name="特斯拉", market="美股"),
        Stock(code="BABA", name="阿里巴巴", market="美股"),
        Stock(code="JD", name="京东", market="美股"),
        Stock(code="PDD", name="拼多多", market="美股"),
    ],
    "科创板": [
        Stock(code="688981", name="中芯国际", market="A股"),
        Stock(code="688599", name="天合光能", market="A股"),
        Stock(code="688012", name="中微公司", market="A股"),
        Stock(code="688111", name="金山办公", market="A股"),
        Stock(code="688223", name="晶科能源", market="A股"),
        Stock(code="688256", name="寒武纪", market="A股"),
        Stock(code="688369", name="致远互联", market="A股"),
        Stock(code="688396", name="华润微", market="A股"),
        Stock(code="688567", name="孚能科技", market="A股"),
        Stock(code="688588", name="凌志软件", market="A股"),
    ],
}


class DataService:
    """数据获取服务"""
    
    def __init__(self, db: Database, use_mock: bool = False):
        self.db = db
        self.use_mock = use_mock or (ak is None)
    
    async def search_stocks(self, keyword: str, market: str = "A股") -> List[Stock]:
        """搜索股票"""
        keyword_upper = keyword.upper()
        
        # 从模拟数据中搜索（包括A股和科创板）
        mock_results = []
        if market == "A股":
            # 搜索A股和科创板
            for mkt in ["A股", "科创板"]:
                mock_results.extend([
                    s for s in MOCK_STOCKS.get(mkt, [])
                    if keyword_upper in s.code.upper() or keyword in s.name
                ])
        else:
            mock_results = [
                s for s in MOCK_STOCKS.get(market, [])
                if keyword_upper in s.code.upper() or keyword in s.name
            ]
        
        if self.use_mock:
            logger.info(f"使用模拟数据搜索: {keyword}, 找到 {len(mock_results)} 条")
            return mock_results[:20]
        
        # 尝试 akshare
        try:
            if market == "A股":
                df = await asyncio.wait_for(
                    asyncio.to_thread(ak.stock_zh_a_spot_em),
                    timeout=15
                )
                filtered = df[df['名称'].str.contains(keyword, na=False)]
                ak_results = [
                    Stock(code=str(row['代码']), name=str(row['名称']), market="A股")
                    for _, row in filtered.head(20).iterrows()
                ]
                # 合并akshare结果和科创板mock结果
                kcb_results = [
                    s for s in MOCK_STOCKS.get("科创板", [])
                    if keyword_upper in s.code.upper() or keyword in s.name
                ]
                return (ak_results + kcb_results)[:20]
            elif market == "港股":
                df = await asyncio.wait_for(
                    asyncio.to_thread(ak.stock_hk_spot_em),
                    timeout=15
                )
                filtered = df[df['名称'].str.contains(keyword, na=False)]
                return [
                    Stock(code=str(row['代码']), name=str(row['名称']), market="港股")
                    for _, row in filtered.head(20).iterrows()
                ]
        except Exception as e:
            logger.warning(f"akshare搜索失败: {e}, 使用模拟数据")
            return mock_results[:20]
        
        return mock_results[:20]
    
    async def fetch_stock_history(
        self, 
        code: str, 
        market: str = "A股",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取股票历史数据 - 按配置优先级自动切换数据源"""
        priority_list = config.data_source_priority.get(market, ["mock"])
        
        for source in priority_list:
            if source == "mock":
                logger.info(f"使用模拟数据获取 {code}")
                return self._generate_mock_history(code)
            
            if self.use_mock:
                continue
                
            if source == "baostock":
                if market == "index":
                    logger.info(f"指数数据不支持Baostock，跳过")
                    continue
                if market != "A股":
                    continue
                if bs is None:
                    logger.warning("baostock未安装，跳过")
                    continue
                try:
                    df = await asyncio.wait_for(
                        asyncio.to_thread(
                            self._fetch_from_baostock, code, market, start_date, end_date
                        ),
                        timeout=30
                    )
                    if df is not None and not df.empty:
                        logger.info(f"成功从Baostock获取 {code} 数据")
                        return df
                except Exception as e:
                    logger.warning(f"Baostock获取失败: {e}")
                    
            elif source == "akshare":
                if ak is None:
                    logger.warning("akshare未安装，跳过")
                    continue
                try:
                    df = await asyncio.wait_for(
                        asyncio.to_thread(self._fetch_from_akshare, code, market),
                        timeout=30
                    )
                    if df is not None and not df.empty:
                        logger.info(f"成功从AKShare获取 {code} 数据")
                        return df
                except Exception as e:
                    logger.warning(f"AKShare获取失败: {e}")
        
        logger.warning(f"所有数据源均失败，使用模拟数据: {code}")
        return self._generate_mock_history(code)
    
    def _fetch_from_akshare(self, code: str, market: str, max_retries: int = 3) -> pd.DataFrame:
        """从AKShare获取数据 - 增加重试机制、延迟和市场后缀支持"""
        import time
        import random
        
        if ak is None:
            raise ImportError("akshare not installed")
        
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
        end_date = datetime.now().strftime("%Y%m%d")
        
        for attempt in range(max_retries):
            try:
                time.sleep((1 + random.random()) * attempt)
                
                if market == "index":
                    df = ak.stock_zh_index_daily(symbol=f"sh{code}" if code.startswith("0") else f"sz{code}")
                    df = df.rename(columns={
                        'date': 'trade_date',
                        'open': 'open',
                        'high': 'high',
                        'low': 'low',
                        'close': 'close',
                        'volume': 'volume',
                    })
                    if 'amount' not in df.columns:
                        df['amount'] = df['close'] * df['volume']
                elif market == "A股":
                    if code.startswith("688"):
                        market_suffix = ".SH"
                        df = ak.stock_zh_a_hist(
                            symbol=code + market_suffix,
                            period="daily",
                            start_date=start_date,
                            end_date=end_date,
                            adjust=""
                        )
                    else:
                        market_suffix = ".SH" if code.startswith("6") else ".SZ"
                        df = ak.stock_zh_a_hist(
                            symbol=code + market_suffix,
                            period="daily",
                            start_date=start_date,
                            end_date=end_date,
                            adjust=""
                        )
                    df = df.rename(columns={
                        '日期': 'trade_date',
                        '开盘': 'open',
                        '最高': 'high',
                        '最低': 'low',
                        '收盘': 'close',
                        '成交量': 'volume',
                        '成交额': 'amount',
                    })
                elif market == "港股":
                    df = ak.stock_hk_hist(symbol=code, adjust="")
                    df = df.rename(columns={
                        'date': 'trade_date',
                        'open': 'open',
                        'high': 'high',
                        'low': 'low',
                        'close': 'close',
                        'volume': 'volume',
                    })
                    if 'amount' not in df.columns:
                        df['amount'] = df['close'] * df['volume']
                else:
                    df = ak.stock_us_hist(symbol=code, adjust="")
                    df = df.rename(columns={
                        'date': 'trade_date',
                        'open': 'open',
                        'high': 'high',
                        'low': 'low',
                        'close': 'close',
                        'volume': 'volume',
                    })
                    if 'amount' not in df.columns:
                        df['amount'] = df['close'] * df['volume']
                
                df['code'] = code
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                return df.sort_values('trade_date')
                
            except Exception as e:
                error_msg = str(e)
                if attempt < max_retries - 1:
                    logger.warning(f"AKShare获取{code}数据失败(尝试 {attempt + 1}/{max_retries}): {error_msg}, 等待重试...")
                    time.sleep(2 + random.random())
                else:
                    logger.error(f"AKShare获取{code}数据失败(已重试{max_retries}次): {error_msg}")
                    raise
        
        raise RuntimeError(f"AKShare获取{code}数据失败: 超过最大重试次数")
    
    def _fetch_from_baostock(
        self, 
        code: str, 
        market: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """从Baostock获取数据 - 支持所有A股（含科创板）
        
        注意：
        1. 股票代码格式必须带市场前缀：
           - 科创板（688开头）: sh.688XXX
           - 沪市主板（60开头）: sh.60XXXX
           - 创业板（30开头）: sz.30XXXX
           - 深市主板（00开头）: sz.00XXXX
        2. 日期格式必须是 YYYY-MM-DD（如 2024-01-01）
        3. 不支持指数、港股、美股
        """
        if bs is None:
            raise ImportError("baostock not installed")
        
        if market != "A股":
            raise ValueError(f"baostock只支持A股，不支持{market}")
        
        if start_date is None:
            start_date_fmt = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        else:
            if "-" in start_date:
                start_date_fmt = start_date
            else:
                start_date_fmt = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
        
        if end_date is None:
            end_date_fmt = datetime.now().strftime("%Y-%m-%d")
        else:
            if "-" in end_date:
                end_date_fmt = end_date
            else:
                end_date_fmt = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
        
        # 确定市场前缀（Baostock要求）
        # 上交所：60（主板）、68（科创板）→ sh.
        # 深交所：00（主板）、30（创业板）→ sz.
        if code.startswith("6"):
            bs_code = f"sh.{code}"  # 上交所（主板60x、科创板68x）
        elif code.startswith("0") or code.startswith("3"):
            bs_code = f"sz.{code}"  # 深交所（主板00x、创业板30x）
        else:
            logger.warning(f"未知的股票代码格式: {code}, 尝试使用深交所前缀")
            bs_code = f"sz.{code}"
        
        logger.debug(f"Baostock股票代码转换: {code} -> {bs_code}")
        
        lg = None
        try:
            lg = bs.login()
            if lg is None or lg.error_code != '0':
                error_msg = "登录失败" if lg is None else f"登录失败: {lg.error_msg}"
                logger.error(f"baostock {error_msg}")
                return pd.DataFrame()
            
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,code,open,high,low,close,volume,amount",
                start_date=start_date_fmt,
                end_date=end_date_fmt,
                frequency="d",
                adjustflag="3"
            )
            
            if rs is None or rs.error_code != '0':
                error_code = rs.error_code if rs is not None else 'N/A'
                error_msg = rs.error_msg if rs is not None else "rs is None"
                logger.error(f"baostock查询失败: {code}, error_code: {error_code}, error_msg: {error_msg}")
                return pd.DataFrame()
            
            data_list = []
            while (rs.error_code == '0') and rs.next():
                data_list.append(rs.get_row_data())
            
            if not data_list:
                logger.warning(f"baostock未获取到数据: {code}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data_list, columns=rs.fields)
            df = df.rename(columns={'date': 'trade_date'})
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df.sort_values('trade_date')
        except Exception as e:
            logger.error(f"baostock获取数据失败: {e}")
            return pd.DataFrame()
        finally:
            if lg is not None:
                bs.logout()
    
    def _generate_mock_history(self, code: str) -> pd.DataFrame:
        """生成模拟历史数据"""
        np.random.seed(hash(code) % 2**32)
        
        dates = pd.date_range(end=datetime.now(), periods=500, freq='D')
        base_price = np.random.uniform(10, 100)
        returns = np.random.randn(500) * 0.02
        close = base_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'trade_date': dates,
            'open': close * (1 + np.random.randn(500) * 0.005),
            'high': close * (1 + np.abs(np.random.randn(500) * 0.01)),
            'low': close * (1 - np.abs(np.random.randn(500) * 0.01)),
            'close': close,
            'volume': np.random.randint(1000000, 10000000, 500).astype(float),
            'amount': close * np.random.randint(10000000, 100000000, 500).astype(float),
        })
        df['code'] = code
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        return df.sort_values('trade_date')
    
    async def download_and_save(
        self, 
        code: str, 
        market: str = "A股",
        force_full: bool = False
    ) -> Tuple[bool, int]:
        """下载并保存股票数据 - 先尝试真实API，失败则用mock"""
        try:
            df = await self.fetch_stock_history(code, market)
            if df is None or df.empty:
                return False, 0
            
            quotes = []
            for _, row in df.iterrows():
                td = row['trade_date']
                if isinstance(td, pd.Timestamp):
                    td = td.date()
                quotes.append(DailyQuote(
                    code=str(row['code']),
                    trade_date=td,
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=float(row['volume']),
                    amount=float(row['amount'])
                ))
            
            await self.db.insert_quotes(quotes)
            await self.db.log_update(market, len(quotes))
            
            return True, len(quotes)
            
        except Exception as e:
            logger.error(f"下载保存数据失败: {e}")
            return False, 0
    
    async def get_cached_data(self, code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取缓存的数据
        
        Args:
            code: 股票代码
            start_date: 起始日期（可选），格式 YYYY-MM-DD
            end_date: 结束日期（可选），格式 YYYY-MM-DD
            
        Returns:
            DataFrame，如果缓存不足则返回空 DataFrame
        """
        quotes = await self.db.get_quotes(code)
        if not quotes:
            logger.debug(f"[缓存] {code} 无缓存数据")
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
            }
            for q in quotes
        ])
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date')
        
        # 检查缓存的日期范围是否覆盖需求
        if start_date or end_date:
            cache_start = df['trade_date'].min()
            cache_end = df['trade_date'].max()
            
            if start_date:
                req_start = pd.to_datetime(start_date)
                if cache_start > req_start:
                    logger.debug(f"[缓存] {code} 缓存起始日期 {cache_start.date()} 晚于需求 {start_date}，需要更新")
                    return pd.DataFrame()
            
            if end_date:
                req_end = pd.to_datetime(end_date)
                # 容忍3天的延迟（考虑周末/节假日）
                if (req_end - cache_end).days > 3:
                    logger.debug(f"[缓存] {code} 缓存结束日期 {cache_end.date()} 早于需求 {end_date}，需要更新")
                    return pd.DataFrame()
        
        logger.info(f"[缓存] {code} 命中缓存，共 {len(df)} 条数据，范围 {df['trade_date'].min().date()} 至 {df['trade_date'].max().date()}")
        
        # 如果指定了日期范围，过滤数据
        if start_date:
            df = df[df['trade_date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['trade_date'] <= pd.to_datetime(end_date)]
        
        return df
