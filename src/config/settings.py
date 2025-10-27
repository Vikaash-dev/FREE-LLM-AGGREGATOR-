from typing import List, Optional, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator

class Settings(BaseSettings):
    # API Server Settings
    ADMIN_TOKEN: Optional[str] = None
    # Use Union to allow string input which will be converted to list
    ALLOWED_ORIGINS: Union[List[str], str] = ["http://localhost:3000"]

    # Database Settings
    DATABASE_URL: str = "sqlite:///./model_memory.db"
    REDIS_URL: Optional[str] = None

    # General Application Settings
    LOG_LEVEL: str = "INFO"
    OPENHANDS_ENCRYPTION_KEY: Optional[str] = None

    # LLM Aggregator Settings
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0

    # Meta Controller Settings
    META_CONTROLLER_LEARNING_RATE: float = 0.1
    META_CONTROLLER_EXPLORATION_RATE: float = 0.1

    # Auto Updater Settings
    AUTO_UPDATE_INTERVAL_MINUTES: int = 60

    @field_validator('ALLOWED_ORIGINS', mode='before')
    @classmethod
    def parse_origins(cls, v):
        """Parse ALLOWED_ORIGINS from string or list."""
        if isinstance(v, str):
            # Split by comma
            return [origin.strip() for origin in v.split(',') if origin.strip()]
        return v

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8', extra='ignore')

settings = Settings()
