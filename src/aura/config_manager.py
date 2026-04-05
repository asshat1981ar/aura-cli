from typing import Dict
from pydantic import BaseModel, ValidationError

class ConfigManager:
    class Settings(BaseModel):
        api_key: str
        log_level: str = 'INFO'

    def __init__(self, env: Dict[str, str]) -> None:
        self.env = env
        self.settings = self.load_settings()

    def load_settings(self) -> Settings:
        try:
            return self.Settings(
                api_key=self.env.get('API_KEY', '')
            )
        except ValidationError as e:
            raise ValueError(f"Configuration error: {e.errors()}")

    def get_log_level(self) -> str:
        return self.settings.log_level

    def get_api_key(self) -> str:
        return self.settings.api_key
