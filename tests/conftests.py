import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.core.async_predictor import AsyncMLPredictor, PredictionResult


@pytest.fixture
def test_client():
    """Фикстура тестового клиента"""
    return TestClient(app)


@pytest.fixture
def mock_predictor():
    """Фикстура мок-предиктора"""
    mock_predictor = AsyncMock(spec=AsyncMLPredictor)
    mock_predictor.model_loaded = True
    mock_predictor.get_stats.return_value = {
        "model_loaded": True,
        "total_predictions": 100,
        "avg_processing_time": 0.1,
        "errors": 0,
        "cache": {"hit_rate": 0.5, "size": 100}
    }

    # Мок предсказаний
    mock_predictions = [
        PredictionResult(probability=0.6 + i * 0.01, processing_time=0.1)
        for i in range(30)
    ]
    mock_predictor.predict_batch.return_value = mock_predictions

    return mock_predictor


@pytest.fixture
def sample_order_request():
    """Фикстура образца запроса"""
    return {
        "distance_in_meters": 2500,
        "duration_in_seconds": 600,
        "pickup_in_meters": 500,
        "pickup_in_seconds": 100,
        "driver_rating": 4.9,
        "user_rating": 4.8,
        "price_start_local": 190,
        "order_timestamp": "2020-05-15T18:30:00",
        "driver_platform": "android",
        "driver_reg_date": "2019-01-15",
        "carmodel": "Camry",
        "carname": "Toyota",
        "driver_id": "29368889",
        "user_id": "16458846"
    }


@pytest.fixture(scope="session")
def event_loop():
    """Фикстура event loop для async тестов"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()