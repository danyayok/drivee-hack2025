import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    """Настройки приложения с валидацией"""

    # API
    APP_NAME: str = "Price Optimizer API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ML Model
    MODEL_PATH: str = os.getenv("MODEL_PATH", "/app/models/catboost_taxi_smart.joblib")
    MODEL_CACHE_SIZE: int = 2000
    MODEL_TIMEOUT: int = 10

    # Async Workers
    THREAD_POOL_WORKERS: int = 16
    PROCESS_POOL_WORKERS: int = 4
    MAX_CONCURRENT_REQUESTS: int = 100

    # Optimization
    PRICE_RANGE_STEPS: int = 50
    MIN_PRICE_MULTIPLIER: float = 0.7
    MAX_PRICE_MULTIPLIER: float = 2.0

    # Cache
    CACHE_TTL: int = 600  # 5 minutes

    # Performance
    MAX_BATCH_SIZE: int = 200
    REQUEST_TIMEOUT: int = 10

    @property
    def model_path(self) -> str:
        """Абсолютный путь к модели"""
        return os.path.abspath(self.MODEL_PATH)

    def validate_settings(self):
        """Валидация настроек"""
        if not os.path.exists(self.MODEL_PATH):
            raise ValueError(f"Файл модели не найден: {self.MODEL_PATH}")

        if self.THREAD_POOL_WORKERS < 1:
            raise ValueError("THREAD_POOL_WORKERS должен быть >= 1")

        if self.PROCESS_POOL_WORKERS < 1:
            raise ValueError("PROCESS_POOL_WORKERS должен быть >= 1")

        if self.MIN_PRICE_MULTIPLIER >= self.MAX_PRICE_MULTIPLIER:
            raise ValueError("MIN_PRICE_MULTIPLIER должен быть меньше MAX_PRICE_MULTIPLIER")

    model_config = ConfigDict(env_file=".env", case_sensitive=False)


# Создаем и валидируем настройки
settings = Settings()

try:
    settings.validate_settings()
except ValueError as e:
    raise RuntimeError(f"Неверные настройки: {e}")