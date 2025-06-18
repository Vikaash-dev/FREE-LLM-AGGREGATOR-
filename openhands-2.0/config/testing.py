from .settings import Settings

class TestingSettings(Settings):
    database_url: str = "sqlite:///./test.db" # type: ignore
    redis_url: str = "redis://localhost:6379/1" # type: ignore

    class Config:
        env_file = ".env.testing"
        case_sensitive = False # Added to match base Settings
