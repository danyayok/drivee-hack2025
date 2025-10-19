import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from app.main import app
from app.core.async_predictor import PredictionResult


class TestIntegration:
    """Интеграционные тесты всей системы"""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def complete_workflow_data(self):
        """Данные для полного workflow"""
        return {
            "order_request": {
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
            },
            "completed_order": {
                "driver_id": "29368889",
                "user_id": "16458846",
                "price_start_local": 200.0,
                "price_bid_local": 240.0,
                "final_price": 240.0,
                "service_commission": 30.72,
                "driver_earnings": 209.28,
                "distance_in_meters": 3500.0,
                "duration_in_seconds": 600.0
            }
        }

    @pytest.mark.asyncio
    async def test_complete_workflow(self, client, complete_workflow_data):
        """Полный тест workflow: предсказание → запись заказа → статистика"""
        # Мок ML predictor
        mock_predictor = Mock()
        mock_predictor.model_loaded = True
        mock_predictor.get_stats.return_value = {
            "model_loaded": True,
            "total_predictions": 1000,
            "avg_processing_time": 0.002,
            "errors": 5,
            "cache": {"hit_rate": 0.65}
        }

        async def mock_predict_batch(features_list):
            return [PredictionResult(0.7, 0.001) for _ in features_list]

        mock_predictor.predict_batch = AsyncMock(side_effect=mock_predict_batch)

        with patch('app.api.endpoints.get_predictor', return_value=mock_predictor):
            # 1. Получаем оптимальные цены
            response = client.post("/api/v1/get_optimal_prices",
                                   json=complete_workflow_data["order_request"])
            assert response.status_code == 200
            price_data = response.json()
            assert len(price_data["price_curve"]) > 0

            # 2. Записываем завершенный заказ
            response = client.post("/api/v1/record_order",
                                   json=complete_workflow_data["completed_order"])
            assert response.status_code == 200
            assert response.json()["status"] == "success"

            # 3. Проверяем что статистика обновилась
            response = client.get("/api/v1/stats")
            assert response.status_code == 200
            stats = response.json()

            # Проверяем финансовую статистику
            assert stats["today_service_revenue"] == pytest.approx(30.72)
            assert stats["today_driver_earnings"] == pytest.approx(209.28)
            assert stats["today_orders"] == 1

            # Проверяем статистику водителей
            assert stats["active_drivers"] == 1
            assert len(stats["top_drivers"]) == 1
            assert stats["top_drivers"][0]["driver_id"] == "29368889"

    def test_error_handling(self, client):
        """Тест обработки ошибок во всей системе"""
        # Тест невалидных данных
        invalid_data = {
            "distance_in_meters": -100,  # Отрицательное значение
            "duration_in_seconds": 600.0,
            "pickup_in_meters": 500.0,
            "pickup_in_seconds": 120.0,
            "driver_rating": 4.8,
            "user_rating": 4.9,
            "price_start_local": 200.0,
            "order_timestamp": "2024-01-15T12:00:00"
        }

        response = client.post("/api/v1/get_optimal_prices", json=invalid_data)
        assert response.status_code == 422  # Validation error

        # Тест недоступности ML сервиса
        with patch('app.api.endpoints.get_predictor') as mock_get_predictor:
            mock_predictor = Mock()
            mock_predictor.model_loaded = False
            mock_get_predictor.return_value = mock_predictor

            response = client.get("/api/v1/health")
            assert response.status_code == 200
            health_data = response.json()
            assert health_data["status"] == "degraded"

    def test_performance_characteristics(self, client, complete_workflow_data):
        """Тест производительности системы"""
        import time

        with patch('app.api.endpoints.get_predictor') as mock_get_predictor:
            mock_predictor = Mock()
            mock_predictor.model_loaded = True
            mock_predictor.get_stats.return_value = {
                "model_loaded": True,
                "total_predictions": 1000,
                "avg_processing_time": 0.001,
                "errors": 0,
                "cache": {"hit_rate": 0.7}
            }

            async def fast_predict_batch(features_list):
                await asyncio.sleep(0.001)  # Быстрое предсказание
                return [PredictionResult(0.7, 0.001) for _ in features_list]

            mock_predictor.predict_batch = AsyncMock(side_effect=fast_predict_batch)
            mock_get_predictor.return_value = mock_predictor

            # Измеряем время ответа
            start_time = time.time()
            response = client.post("/api/v1/get_optimal_prices",
                                   json=complete_workflow_data["order_request"])
            end_time = time.time()

            assert response.status_code == 200
            processing_time = response.json()["processing_time_ms"]

            # Проверяем что время обработки разумное
            assert processing_time < 1000  # Меньше 1 секунды
            assert (end_time - start_time) * 1000 < 2000  # Меньше 2 секунд полного времени