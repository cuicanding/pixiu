"""数据库服务模块"""

import aiosqlite
from typing import Optional, List
from datetime import date

from pixiu.models.stock import Stock, DailyQuote


class Database:
    """SQLite数据库服务"""
    
    def __init__(self, db_path: str = "data/stocks.db"):
        self.db_path = db_path
    
    async def ensure_tables(self):
        """Ensure tables exist, create if not."""
        await self.create_tables()
    
    async def create_tables(self):
        """创建数据库表"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS stocks (
                    code TEXT PRIMARY KEY,
                    name TEXT,
                    market TEXT,
                    industry TEXT,
                    list_date DATE,
                    updated_at TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS daily_quotes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code TEXT,
                    trade_date DATE,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    amount REAL,
                    turnover_rate REAL,
                    FOREIGN KEY (code) REFERENCES stocks(code),
                    UNIQUE(code, trade_date)
                );
                
                CREATE INDEX IF NOT EXISTS idx_quotes_code_date 
                ON daily_quotes(code, trade_date);
                
                CREATE TABLE IF NOT EXISTS strategy_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code TEXT,
                    strategy_name TEXT,
                    signal_date DATE,
                    signal_type TEXT,
                    confidence REAL,
                    price REAL,
                    metadata TEXT
                );
                
                CREATE TABLE IF NOT EXISTS update_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market TEXT,
                    last_update TIMESTAMP,
                    records_updated INTEGER
                );
            """)
            await db.commit()
    
    async def insert_stock(self, stock: Stock):
        """插入或更新股票信息"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO stocks (code, name, market, industry, list_date, updated_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
            """, (stock.code, stock.name, stock.market, stock.industry, stock.list_date))
            await db.commit()
    
    async def get_stock(self, code: str) -> Optional[Stock]:
        """获取股票信息"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM stocks WHERE code = ?", (code,))
            row = await cursor.fetchone()
            if row:
                return Stock(**dict(row))
            return None
    
    async def insert_quotes(self, quotes: List[DailyQuote]):
        """批量插入行情数据"""
        async with aiosqlite.connect(self.db_path) as db:
                await db.executemany("""
                    INSERT OR REPLACE INTO daily_quotes 
                    (code, trade_date, open, high, low, close, volume, amount, turnover_rate)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    (q.code, q.trade_date, q.open, q.high, q.low, q.close, q.volume, q.amount, q.turnover_rate)
                    for q in quotes
                ])
                await db.commit()
    
    async def get_quotes(self, code: str, start_date: str = None, end_date: str = None) -> List[DailyQuote]:
        """获取股票行情数据"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            query = "SELECT * FROM daily_quotes WHERE code = ?"
            params = [code]
            
            if start_date:
                query += " AND trade_date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND trade_date <= ?"
                params.append(end_date)
            
            query += " ORDER BY trade_date"
            
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            return [DailyQuote(**dict(row)) for row in rows]
    
    async def get_last_update(self, market: str) -> Optional[str]:
        """获取最后更新时间"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT MAX(last_update) FROM update_logs WHERE market = ?",
                (market,)
            )
            result = await cursor.fetchone()
            return result[0] if result else None
    
    async def log_update(self, market: str, records_count: int):
        """记录更新日志"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO update_logs (market, last_update, records_updated)
                VALUES (?, datetime('now'), ?)
            """, (market, records_count))
            await db.commit()
