
import os
from typing import List, Dict, Optional, Any, Union
from pydantic import Field, BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# --- Nested Configuration Models ---

class SystemInfo(BaseModel):
    env: str = "dev"
    timezone: str = "Asia/Ho_Chi_Minh"
    # Added from feature flags
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    vector_store: str = "memory"

class MemoryConfig(BaseModel):
    context_ttl_seconds: int = 1800
    cleanup_interval_seconds: int = 300
    max_active_contexts: int = 10000
    
    # New: Lazy Loading & LRU Cache for Embeddings
    embedding_cache_size: int = 10000  # Number of vectors
    embedding_cache_ttl_seconds: int = 3600

class WeightsConfig(BaseModel):
    default_rule: float = 0.6
    default_embed: float = 0.4
    domain_boost: float = 0.1

class ThresholdsConfig(BaseModel):
    pairwise_gap: float = 0.1
    entity_max_boost: float = 0.15
    min_confidence: float = 0.4

class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"
    file_path: str = "logs/system.log"

class TimeoutsConfig(BaseModel):
    embedding_inference_ms: int = 500
    external_api_ms: int = 2000

class RedisConfig(BaseModel):
    enabled: bool = False
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ttl_seconds: int = 3600
    key_prefix: str = "chatbot:vector:"

# --- Main Settings Class ---

class Settings(BaseSettings):
    # Nested configs
    system: SystemInfo = SystemInfo()
    memory: MemoryConfig = MemoryConfig()
    weights: WeightsConfig = WeightsConfig()
    thresholds: ThresholdsConfig = ThresholdsConfig()
    logging: LoggingConfig = LoggingConfig()
    timeouts: TimeoutsConfig = TimeoutsConfig()
    redis: RedisConfig = RedisConfig()

    # Paths (can be overridden by env vars)
    CONFIG_DIR: str = "config"
    ACTION_CATALOG_PATH: str = "config/action_catalog.yaml"
    KEYWORD_RULES_PATH: str = "config/keyword_rules.yaml"
    SYSTEM_CONFIG_PATH: str = "config/system_config.yaml"

    model_config = SettingsConfigDict(
        env_prefix="CHATBOT_",
        env_nested_delimiter="__",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8"
    )

    @field_validator("ACTION_CATALOG_PATH", "KEYWORD_RULES_PATH", "SYSTEM_CONFIG_PATH", mode="before")
    @classmethod
    def validate_path(cls, v):
        # Allow relative paths to be resolved relative to CWD or CONFIG_DIR
        return str(v)

# Global Settings Instance
settings = Settings()
