from .settings import Settings

class ProductionSettings(Settings):
    debug: bool = False
    log_level: str = "WARNING"

    class Config:
        env_file = ".env.production"
        case_sensitive = False # Added to match base Settings
