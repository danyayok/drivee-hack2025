import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
from datetime import datetime

from app.main import app
from app.models.data_models import OrderRequest
from app.core.async_predictor import PredictionResult


class TestAPIIntegration:
    """Интеграционные тесты API"""

    def setup_method(self):
        self.client = TestClient(app)
        self.valid_request = {
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

    def test_health_check(self):
        """Тест health check эндпоинта"""
        response = self.client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "total_predictions" in data
        assert data["status"] in ["healthy", "degraded"]

    def test_stats_endpoint(self):
        """Тест эндпоинта статистики"""
        response = self.client.get("/api/v1/stats")

        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "successful_requests" in data
        assert "failed_requests" in data
        assert "cache_hit_rate" in data

    def test_root_endpoint(self):
        """Тест корневого эндпоинта"""
        response = self.client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "status" in data
        assert data["status"] == "running"

    def test_ready_endpoint(self):
        """Тест readiness check"""
        response = self.client.get("/ready")

        # Может быть 200 или 503 в зависимости от состояния
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            assert response.json()["status"] == "ready"

    def test_live_endpoint(self):
        """Тест liveness check"""
        response = self.client.get("/live")

        assert response.status_code == 200
        assert response.json()["status"] == "alive"


class TestOptimalPricesEndpoint:
    """Тесты для основного эндпоинта оптимизации цен"""

    def setup_method(self):
        self.client = TestClient(app)
        self.base_request = {
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

    @patch('app.api.endpoints.get_predictor')
    def test_optimal_prices_success(self, mock_get_predictor):
        """Тест успешного запроса оптимальных цен"""
        # Мок предсказаний
        mock_predictor = AsyncMock()
        mock_predictions = [
            PredictionResult(probability=0.7, processing_time=0.1),
            PredictionResult(probability=0.5, processing_time=0.1),
            PredictionResult(probability=0.3, processing_time=0.1)
        ]
        mock_predictor.predict_batch.return_value = mock_predictions
        mock_get_predictor.return_value = mock_predictor

        response = self.client.post("/api/v1/get_optimal_prices", json=self.base_request)

        assert response.status_code == 200
        data = response.json()

        # Проверяем структуру ответа
        assert "price_curve" in data
        assert "processing_time_ms" in data
        assert "analysis" in data
        assert isinstance(data["price_curve"], list)
        assert data["processing_time_ms"] > 0

    @patch('app.api.endpoints.get_predictor')
    def test_optimal_prices_with_different_prices(self, mock_get_predictor):
        """Тест с разными базовыми ценами"""
        mock_predictor = AsyncMock()
        # Мок реалистичных вероятностей
        mock_predictions = [PredictionResult(probability=0.6, processing_time=0.1) for _ in range(30)]
        mock_predictor.predict_batch.return_value = mock_predictions
        mock_get_predictor.return_value = mock_predictor

        test_cases = [100, 200, 500, 1000]  # Разные базовые цены

        for base_price in test_cases:
            request_data = self.base_request.copy()
            request_data["price_start_local"] = base_price

            response = self.client.post("/api/v1/get_optimal_prices", json=request_data)

            assert response.status_code == 200
            data = response.json()
            assert len(data["price_curve"]) <= 8  # Не больше 8 предложений

    def test_optimal_prices_validation(self):
        """Тест валидации входных данных"""
        invalid_requests = [
            # Неверный рейтинг
            {**self.base_request, "driver_rating": 6.0},
            # Отрицательная дистанция
            {**self.base_request, "distance_in_meters": -100},
            # Отсутствует обязательное поле
            {**self.base_request, "price_start_local": None},
            # Неверный timestamp
            {**self.base_request, "order_timestamp": "invalid-date"},
        ]

        for invalid_request in invalid_requests:
            response = self.client.post("/api/v1/get_optimal_prices", json=invalid_request)
            assert response.status_code in [400, 422]  # Validation error

    @patch('app.api.endpoints.get_predictor')
    def test_optimal_prices_empty_response(self, mock_get_predictor):
        """Тест когда нет подходящих цен"""
        mock_predictor = AsyncMock()
        # Все вероятности вне диапазона фильтрации
        mock_predictions = [PredictionResult(probability=0.1, processing_time=0.1) for _ in range(30)]
        mock_predictor.predict_batch.return_value = mock_predictions
        mock_get_predictor.return_value = mock_predictor

        response = self.client.post("/api/v1/get_optimal_prices", json=self.base_request)

        assert response.status_code == 200
        data = response.json()
        assert len(data["price_curve"]) == 0  # Пустой результат

    @patch('app.api.endpoints.get_predictor')
    def test_optimal_prices_error_handling(self, mock_get_predictor):
        """Тест обработки ошибок ML модели"""
        mock_predictor = AsyncMock()
        mock_predictor.predict_batch.side_effect = Exception("ML Model error")
        mock_get_predictor.return_value = mock_predictor

        response = self.client.post("/api/v1/get_optimal_prices", json=self.base_request)

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "ML Model error" in data["detail"]


class TestDebugEndpoints:
    """Тесты отладочных эндпоинтов"""

    def setup_method(self):
        self.client = TestClient(app)

    @patch('app.api.endpoints.get_predictor')
    def test_debug_model(self, mock_get_predictor):
        """Тест отладочного эндпоинта модели"""
        mock_predictor = AsyncMock()
        mock_predictor.predict_batch.return_value = [
            PredictionResult(probability=0.5, processing_time=0.1, cache_hit=False),
            PredictionResult(probability=0.6, processing_time=0.1, cache_hit=True)
        ]
        mock_predictor.get_stats.return_value = {
            "model_loaded": True,
            "total_predictions": 100,
            "cache": {"hit_rate": 0.5}
        }
        mock_get_predictor.return_value = mock_predictor

        request_data = {
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
            "carname": "Toyota"
        }

        response = self.client.post("/api/v1/debug_model", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "test_results" in data
        assert "model_stats" in data
        assert "message" in data
        assert len(data["test_results"]) == 4  # 4 тестовые цены

    def test_debug_historical_data(self):
        """Тест отладочного эндпоинта исторических данных"""
        response = self.client.get("/api/v1/debug/historical_data")

        assert response.status_code == 200
        data = response.json()
        assert "driver_found" in data
        assert "user_found" in data
        assert "total_drivers" in data
        assert "total_users" in data


class TestPerformance:
    """Тесты производительности"""

    def setup_method(self):
        self.client = TestClient(app)
        self.valid_request = {
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

    @patch('app.api.endpoints.get_predictor')
    def test_response_time(self, mock_get_predictor):
        """Тест времени ответа"""
        mock_predictor = AsyncMock()
        # Быстрые предсказания
        mock_predictions = [PredictionResult(probability=0.5, processing_time=0.01) for _ in range(30)]
        mock_predictor.predict_batch.return_value = mock_predictions
        mock_get_predictor.return_value = mock_predictor

        import time
        start_time = time.time()

        response = self.client.post("/api/v1/get_optimal_prices", json=self.valid_request)

        end_time = time.time()
        response_time = (end_time - start_time) * 1000  # в миллисекундах

        assert response.status_code == 200
        # Ожидаем время меньше 5 секунд даже с моками
        assert response_time < 5000

    @patch('app.api.endpoints.get_predictor')
    def test_concurrent_requests(self, mock_get_predictor):
        """Тест конкурентных запросов"""
        import threading
        import queue

        mock_predictor = AsyncMock()
        mock_predictions = [PredictionResult(probability=0.5, processing_time=0.1) for _ in range(30)]
        mock_predictor.predict_batch.return_value = mock_predictions
        mock_get_predictor.return_value = mock_predictor

        results = queue.Queue()
        errors = queue.Queue()

        def make_request():
            try:
                response = self.client.post("/api/v1/get_optimal_prices", json=self.valid_request)
                results.put(response.status_code)
            except Exception as e:
                errors.put(e)

        # Запускаем 5 concurrent запросов
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Проверяем что все запросы успешны
        assert errors.empty(), f"Errors in concurrent requests: {list(errors.queue)}"

        status_codes = []
        while not results.empty():
            status_codes.append(results.get())

        assert all(code == 200 for code in status_codes)


class TestDataModels:
    """Тесты моделей данных"""

    def test_order_request_validation(self):
        """Тест валидации OrderRequest"""
        # Валидные данные
        valid_data = {
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
            "carname": "Toyota"
        }

        order_request = OrderRequest(**valid_data)
        assert order_request.price_start_local == 190
        assert order_request.driver_platform.value == "android"

    def test_order_request_business_logic(self):
        """Тест бизнес-логики валидации"""
        # pickup_in_seconds > duration_in_seconds - должно вызывать ошибку
        invalid_data = {
            "distance_in_meters": 2500,
            "duration_in_seconds": 100,  # меньше чем pickup
            "pickup_in_meters": 500,
            "pickup_in_seconds": 600,  # больше чем duration
            "driver_rating": 4.9,
            "user_rating": 4.8,
            "price_start_local": 190,
            "order_timestamp": "2020-05-15T18:30:00",
            "driver_platform": "android",
            "driver_reg_date": "2019-01-15"
        }

        with pytest.raises(ValueError):
            OrderRequest(**invalid_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])