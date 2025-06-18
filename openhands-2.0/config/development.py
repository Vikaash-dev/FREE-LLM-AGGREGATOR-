from .settings import Settings

class DevelopmentSettings(Settings):
    debug: bool = True
    log_level: str = "DEBUG"

    class Config:
        env_file = ".env.development"
        case_sensitive = False # Added to match base Settings
