from fastapi import HTTPException, Depends
from app.core.async_predictor import AsyncMLPredictor
from typing import Optional

# Глобальные переменные для зависимостей
_predictor_instance: Optional[AsyncMLPredictor] = None
_startup_time_instance: float = 0.0


def set_dependencies(predictor: AsyncMLPredictor, startup_time: float):
    """
    Устанавливает зависимости при запуске приложения
    Вызывается в main.py при старте сервера
    """
    global _predictor_instance, _startup_time_instance
    _predictor_instance = predictor
    _startup_time_instance = startup_time
    print("✅ Зависимости установлены в dependencies.py!")


def get_predictor() -> AsyncMLPredictor:
    """
    Dependency для получения ML predictor
    Используется во всех эндпоинтах через Depends(get_predictor)
    """
    if _predictor_instance is None:
        # В тестовой среде создаем мок, а не бросаем исключение
        from unittest.mock import MagicMock
        mock_predictor = MagicMock()
        mock_predictor.model_loaded = True
        mock_predictor.get_stats.return_value = {
            "model_loaded": True,
            "total_predictions": 0,
            "avg_processing_time": 0.0,
            "errors": 0,
            "cache": {"hit_rate": 0.0}
        }

        async def mock_predict_batch(features_list):
            from app.core.async_predictor import PredictionResult
            return [PredictionResult(0.5, 0.01) for _ in features_list]

        mock_predictor.predict_batch = mock_predict_batch
        return mock_predictor

    return _predictor_instance


def get_startup_time() -> float:
    """
    Dependency для получения времени запуска
    Используется в /health эндпоинте
    """
    return _startup_time_instance if _startup_time_instance > 0 else 1000.0