"""
Centralized configuration management for the LLM API Aggregator.
"""

from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Security Settings
    ADMIN_TOKEN: Optional[str] = None
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8501"]
    ENCRYPTION_KEY: Optional[str] = None
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    
    # Database Settings
    DATABASE_URL: str = "sqlite:///./model_memory.db"
    REDIS_URL: Optional[str] = None
    
    # Rate Limiting Settings
    GLOBAL_REQUESTS_PER_MINUTE: int = 100
    USER_REQUESTS_PER_MINUTE: int = 10
    MAX_CONCURRENT_REQUESTS: int = 50
    
    # Provider Settings
    DEFAULT_PROVIDER: str = "auto"
    ENABLE_CACHING: bool = True
    CACHE_TTL: int = 3600
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    
    # Meta Controller Settings
    META_CONTROLLER_LEARNING_RATE: float = 0.1
    META_CONTROLLER_EXPLORATION_RATE: float = 0.1
    ENABLE_ML_FEATURES: bool = True
    
    # Auto Updater Settings
    AUTO_UPDATE_INTERVAL_MINUTES: int = 60
    
    # Development Settings
    DEBUG: bool = False
    TESTING: bool = False
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        extra='ignore',
        case_sensitive=True
    )


# Global settings instance
settings = Settings()
