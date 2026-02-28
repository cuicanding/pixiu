"""Pixiu 配置管理"""

import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass, field

load_dotenv()


@dataclass
class Config:
    database_path: str = field(default_factory=lambda: os.getenv("DATABASE_PATH", "data/stocks.db"))
    cache_dir: str = field(default_factory=lambda: os.getenv("CACHE_DIR", "data/cache"))
    glm_api_key: str = field(default_factory=lambda: os.getenv("GLM_API_KEY", ""))

    initial_capital: float = 100000.0
    commission_rate: float = 0.0003
    slippage_rate: float = 0.0001
    position_size: float = 0.95
    risk_free_rate: float = 0.03

    regime_window_days: int = 60
    regime_adx_threshold: float = 25.0
    regime_slope_threshold: float = 0.005

    data_update_days: int = 30
    data_source_priority: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.data_source_priority:
            self.data_source_priority = {
                "A股": ["baostock", "akshare", "mock"],
                "港股": ["akshare", "mock"],
                "美股": ["akshare", "mock"],
                "index": ["baostock", "mock"],
            }

    @property
    def base_dir(self) -> Path:
        return Path(__file__).parent.parent

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    def ensure_dirs(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)


config = Config()
