"""配置模块测试"""

import pytest
from pixiu.config import Config


def test_config_default_values():
    """测试配置默认值"""
    config = Config()
    assert config.database_path == "data/stocks.db"
    assert config.cache_dir == "data/cache"
    assert config.initial_capital == 100000


def test_config_from_env(monkeypatch):
    """测试从环境变量加载配置"""
    monkeypatch.setenv("GLM_API_KEY", "test_key_123")
    config = Config()
    assert config.glm_api_key == "test_key_123"
