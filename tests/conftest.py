import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock
import sys
import os

# Добавляем корневую директорию в PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.core.async_predictor import AsyncMLPredictor, PredictionResult
from app.core.dependencies import set_dependencies


class MockPredictionResult:
    """Mock для PredictionResult"""

    def __init__(self, probability, processing_time, cache_hit=False, error=None):
        self.probability = probability
        self.processing_time = processing_time
        self.cache_hit = cache_hit
        self.error = error


@pytest.fixture
def test_client():
    """Тестовый клиент FastAPI"""
    return TestClient(app)


@pytest.fixture
def mock_predictor():
    """Мок ML predictor"""
    predictor = MagicMock(spec=AsyncMLPredictor)
    predictor.model_loaded = True

    # Мок для get_stats
    predictor.get_stats.return_value = {
        "model_loaded": True,
        "total_predictions": 1000,
        "avg_processing_time": 0.1,
        "errors": 5,
        "cache": {
            "hit_rate": 0.65,
            "size": 150,
            "hits": 650,
            "misses": 350
        }
    }

    # Мок для predict_batch
    async def mock_predict_batch(features_list):
        results = []
        for features in features_list:
            price_bid = features.get('price_bid_local', 200)
            price_start = features.get('price_start_local', 200)

            # Базовая вероятность уменьшается с ростом цены
            base_prob = 0.8
            price_ratio = price_bid / price_start
            if price_ratio > 1.5:
                probability = max(0.1, base_prob - (price_ratio - 1.5) * 0.3)
            elif price_ratio < 0.8:
                probability = min(0.95, base_prob + (0.8 - price_ratio) * 0.2)
            else:
                probability = base_prob - (price_ratio - 1) * 0.2

            probability = max(0.05, min(0.95, probability))

            results.append(PredictionResult(
                probability=probability,
                processing_time=0.01,
                cache_hit=False,
                error=None
            ))
        return results

    predictor.predict_batch = mock_predict_batch

    # Мок для predict_single
    async def mock_predict_single(features):
        results = await mock_predict_batch([features])
        return results[0] if results else PredictionResult(0.5, 0.01)

    predictor.predict_single = mock_predict_single

    return predictor


@pytest.fixture(autouse=True)
def setup_dependencies(mock_predictor):
    """Настройка зависимостей для всех тестов"""
    set_dependencies(mock_predictor, 1000.0)
    yield
    # Cleanup
    set_dependencies(None, 0.0)


@pytest.fixture
def sample_order_data():
    """Реальные данные из предоставленного датасета"""
    return {
        "distance_in_meters": 3404.0,
        "duration_in_seconds": 486.0,
        "pickup_in_meters": 790.0,
        "pickup_in_seconds": 169.0,
        "driver_rating": 5.0,
        "user_rating": 4.8,
        "price_start_local": 180.0,
        "order_timestamp": "2020-05-01T00:05:14",
        "driver_platform": "android",
        "driver_reg_date": "2019-09-22",
        "carmodel": "Logan",
        "carname": "Renault"
    }


@pytest.fixture
def multiple_order_samples():
    """Несколько реальных примеров заказов для тестирования"""
    return [
        {
            "distance_in_meters": 994.0,
            "duration_in_seconds": 176.0,
            "pickup_in_meters": 469.0,
            "pickup_in_seconds": 85.0,
            "driver_rating": 5.0,
            "user_rating": 4.9,
            "price_start_local": 160.0,
            "order_timestamp": "2020-05-01T00:06:25",
            "driver_platform": "android",
            "driver_reg_date": "2019-02-04",
            "carmodel": "Sandero Stepway",
            "carname": "Renault"
        },
        {
            "distance_in_meters": 2749.0,
            "duration_in_seconds": 471.0,
            "pickup_in_meters": 436.0,
            "pickup_in_seconds": 160.0,
            "driver_rating": 5.0,
            "user_rating": 4.7,
            "price_start_local": 150.0,
            "order_timestamp": "2020-05-01T00:08:19",
            "driver_platform": "android",
            "driver_reg_date": "2017-12-21",
            "carmodel": "Avensis",
            "carname": "Toyota"
        }
    ]