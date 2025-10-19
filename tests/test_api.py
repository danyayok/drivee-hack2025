import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch
import numpy as np
from datetime import datetime, timedelta

from app.main import app
from app.models.data_models import OrderRequest, OptimalPricesResponse
from app.core.async_predictor import PredictionResult


class TestAPIEndpoints:
    """Тесты API эндпоинтов"""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def sample_order_request(self):
        return {
            "distance_in_meters": 3500.0,
            "duration_in_seconds": 600.0,
            "pickup_in_meters": 500.0,
            "pickup_in_seconds": 120.0,
            "driver_rating": 4.8,
            "user_rating": 4.9,
            "price_start_local": 200.0,
            "order_timestamp": "2024-01-15T12:00:00",
            "driver_platform": "android",
            "driver_reg_date": "2023-01-01",
            "carname": "Toyota",
            "carmodel": "Camry",
            "driver_id": "29368889",
            "user_id": "16458846"
        }

    @pytest.fixture
    def mock_predictor(self):
        """Мок ML predictor"""
        predictor = Mock()
        predictor.model_loaded = True
        predictor.get_stats.return_value = {
            "model_loaded": True,
            "total_predictions": 1000,
            "avg_processing_time": 0.002,
            "errors": 5,
            "cache": {"hit_rate": 0.65}
        }

        # Мок для predict_batch - возвращает реалистичные вероятности
        async def mock_predict_batch(features_list):
            base_prob = 0.7
            results = []
            for i, features in enumerate(features_list):
                # Разные вероятности для разных цен
                price = features.get('price_bid_local', 200)
                prob = base_prob * (1 - (price - 200) / 1000)  # Уменьшаем вероятность с ростом цены
                prob = max(0.3, min(0.9, prob))  # Ограничиваем диапазон
                results.append(PredictionResult(prob, 0.001))
            return results

        predictor.predict_batch = AsyncMock(side_effect=mock_predict_batch)
        predictor.predict_single = AsyncMock(return_value=PredictionResult(0.7, 0.001))

        return predictor

    def test_root_endpoint(self, client):
        """Тест корневого эндпоинта"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data

    def test_health_check(self, client, mock_predictor):
        """Тест health check эндпоинта"""
        with patch('app.api.endpoints.get_predictor', return_value=mock_predictor):
            with patch('app.api.endpoints.get_startup_time', return_value=3600.0):
                response = client.get("/api/v1/health")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"
                assert data["model_loaded"] == True

    def test_get_optimal_prices_success(self, client, sample_order_request, mock_predictor):
        """Тест успешного получения оптимальных цен"""
        with patch('app.api.endpoints.get_predictor', return_value=mock_predictor):
            response = client.post("/api/v1/get_optimal_prices", json=sample_order_request)
            assert response.status_code == 200
            data = response.json()

            # Проверяем структуру ответа
            assert "price_curve" in data
            assert "processing_time_ms" in data
            assert "analysis" in data

            # Проверяем что есть варианты цен
            assert len(data["price_curve"]) > 0

            # Проверяем структуру ценовых точек
            price_point = data["price_curve"][0]
            expected_fields = ["price", "probability_percent", "expected_revenue",
                               "service_earnings", "driver_earnings"]
            for field in expected_fields:
                assert field in price_point

    def test_get_optimal_prices_validation_error(self, client):
        """Тест валидации входных данных"""
        invalid_request = {
            "distance_in_meters": -100,  # Отрицательное расстояние
            "duration_in_seconds": 600.0,
            "pickup_in_meters": 500.0,
            "pickup_in_seconds": 120.0,
            "driver_rating": 4.8,
            "user_rating": 4.9,
            "price_start_local": 200.0,
            "order_timestamp": "2024-01-15T12:00:00"
        }

        response = client.post("/api/v1/get_optimal_prices", json=invalid_request)
        assert response.status_code == 422  # Validation error

    def test_get_optimal_prices_business_logic_validation(self, client):
        """Тест бизнес-логики валидации"""
        invalid_request = {
            "distance_in_meters": 1000.0,
            "duration_in_seconds": 600.0,
            "pickup_in_meters": 1500.0,  # Дистанция подачи больше дистанции поездки
            "pickup_in_seconds": 120.0,
            "driver_rating": 4.8,
            "user_rating": 4.9,
            "price_start_local": 200.0,
            "order_timestamp": "2024-01-15T12:00:00"
        }

        response = client.post("/api/v1/get_optimal_prices", json=invalid_request)
        assert response.status_code == 422

    def test_service_stats(self, client, mock_predictor):
        """Тест получения статистики сервиса"""
        with patch('app.api.endpoints.get_predictor', return_value=mock_predictor):
            response = client.get("/api/v1/stats")
            assert response.status_code == 200
            data = response.json()

            # Проверяем основные поля
            assert "total_requests" in data
            assert "successful_requests" in data
            assert "failed_requests" in data
            assert "cache_hit_rate" in data

            # Проверяем системные метрики
            assert "memory_usage_mb" in data
            assert "cpu_usage_percent" in data

            # Проверяем финансовую статистику
            assert "total_service_revenue" in data
            assert "total_driver_earnings" in data

            # Проверяем статистику водителей
            assert "active_drivers" in data
            assert "top_drivers" in data

    def test_ready_endpoint(self, client, mock_predictor):
        """Тест readiness check"""
        with patch('app.main.predictor', mock_predictor):
            response = client.get("/ready")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"

    def test_live_endpoint(self, client):
        """Тест liveness check"""
        response = client.get("/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"

    def test_not_found_handler(self, client):
        """Тест обработки 404 ошибок"""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data