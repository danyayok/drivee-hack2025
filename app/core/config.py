import os
import platform
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    """Настройки приложения с валидацией"""

    # API
    APP_NAME: str = "Driveechock API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ML Model - кроссплатформенные пути
    MODEL_PATH: str = os.getenv("MODEL_PATH",
                                "models/catboost_taxi_smart.joblib" if platform.system() == "Windows"
                                else "/app/models/catboost_taxi_smart.joblib"
                                )

    # Пути для Linux
    LOG_DIR: str = os.getenv("LOG_DIR",
                             "logs" if platform.system() == "Windows"
                             else "/var/log/price-optimizer"
                             )

    DATA_PATH: str = os.getenv("DATA_PATH",
                               "train.csv" if platform.system() == "Windows"
                               else "/app/data/train.csv"
                               )

    # Настройки пулов в зависимости от ОС
    if platform.system() == "Windows":
        PROCESS_POOL_WORKERS: int = int(os.getenv("PROCESS_POOL_WORKERS", "4"))
    else:
        PROCESS_POOL_WORKERS: int = 0  # Отключаем ProcessPool на Linux

    THREAD_POOL_WORKERS: int = int(os.getenv("THREAD_POOL_WORKERS",
                                             "8" if platform.system() == "Windows" else "16"
                                             ))

    # ML Model Cache
    MODEL_CACHE_SIZE: int = 2000
    MODEL_TIMEOUT: int = 10

    # Optimization
    MAX_CONCURRENT_REQUESTS: int = 100
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
        path = os.path.abspath(self.MODEL_PATH)
        if not os.path.exists(path):
            # Для Linux создаем fallback
            if platform.system() != "Windows":
                fallback_path = "models/catboost_taxi_smart.joblib"
                if os.path.exists(fallback_path):
                    return os.path.abspath(fallback_path)
        return path

    def validate_settings(self):
        """Валидация настроек для Linux"""
        if platform.system() != "Windows":
            # На Linux проверяем доступность директорий
            log_dir = os.path.dirname(self.LOG_DIR)
            if not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir, exist_ok=True)
                    print(f"✅ Создана директория для логов: {log_dir}")
                except PermissionError:
                    print(f"⚠️ Нет прав для создания {log_dir}, используем текущую директорию")
                    self.LOG_DIR = "./logs"

            # Создаем директорию для моделей если нужно
            model_dir = os.path.dirname(self.MODEL_PATH)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)

        if self.THREAD_POOL_WORKERS < 1:
            raise ValueError("THREAD_POOL_WORKERS должен быть >= 1")

        if self.MIN_PRICE_MULTIPLIER >= self.MAX_PRICE_MULTIPLIER:
            raise ValueError("MIN_PRICE_MULTIPLIER должен быть меньше MAX_PRICE_MULTIPLIER")

    model_config = ConfigDict(env_file=".env", case_sensitive=False)


# Создаем и валидируем настройки
settings = Settings()

try:
    settings.validate_settings()
except ValueError as e:
    raise RuntimeError(f"Неверные настройки: {e}")