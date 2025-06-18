import os
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Application
    app_name: str = "OpenHands 2.0"
    app_version: str = "2.0.0"
    debug: bool = False

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4

    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    redis_url: str = Field(..., env="REDIS_URL")

    # Security
    secret_key: str = Field(..., env="SECRET_KEY")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # AI Models
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(None, env="GOOGLE_API_KEY")

    # Research Integration
    arxiv_api_key: Optional[str] = Field(None, env="ARXIV_API_KEY")
    github_token: Optional[str] = Field(None, env="GITHUB_TOKEN")

    # Performance
    cache_ttl: int = 3600
    max_concurrent_tasks: int = 100
    request_timeout: int = 300

    # Security
    max_input_length: int = 100000
    injection_threshold: float = 0.8
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

    # Monitoring
    prometheus_port: int = 9090
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
