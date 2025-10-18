import pytest
from datetime import datetime
from pydantic import ValidationError

from app.models.data_models import (
    OrderRequest, OptimalPricesResponse, HealthResponse,
    ServiceStats, DriverPlatform
)


class TestOrderRequest:
    """Тесты для OrderRequest модели"""

    def test_valid_order_request(self):
        """Тест валидного OrderRequest"""
        valid_data = {
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

        order = OrderRequest(**valid_data)

        assert order.distance_in_meters == 3404.0
        assert order.driver_platform == DriverPlatform.ANDROID
        assert order.price_start_local == 180.0

    def test_invalid_ratings(self):
        """Тест невалидных рейтингов"""
        invalid_data = {
            "distance_in_meters": 1000,
            "duration_in_seconds": 300,
            "pickup_in_meters": 500,
            "pickup_in_seconds": 100,
            "driver_rating": 6.0,  # > 5.0
            "user_rating": 4.5,
            "price_start_local": 200,
            "order_timestamp": "2020-05-01T00:05:14",
            "driver_platform": "android",
            "driver_reg_date": "2019-01-01"
        }

        with pytest.raises(ValidationError):
            OrderRequest(**invalid_data)

    def test_invalid_price(self):
        """Тест невалидной цены"""
        invalid_data = {
            "distance_in_meters": 1000,
            "duration_in_seconds": 300,
            "pickup_in_meters": 500,
            "pickup_in_seconds": 100,
            "driver_rating": 4.5,
            "user_rating": 4.5,
            "price_start_local": 0,  # <= 0
            "order_timestamp": "2020-05-01T00:05:14",
            "driver_platform": "android",
            "driver_reg_date": "2019-01-01"
        }

        with pytest.raises(ValidationError):
            OrderRequest(**invalid_data)

    def test_business_logic_validation(self):
        """Тест бизнес-логики валидации"""
        # pickup_in_seconds > duration_in_seconds
        invalid_data = {
            "distance_in_meters": 1000,
            "duration_in_seconds": 100,
            "pickup_in_meters": 500,
            "pickup_in_seconds": 200,  # > duration_in_seconds
            "driver_rating": 4.5,
            "user_rating": 4.5,
            "price_start_local": 200,
            "order_timestamp": "2020-05-01T00:05:14",
            "driver_platform": "android",
            "driver_reg_date": "2019-01-01"
        }

        with pytest.raises(ValidationError):
            OrderRequest(**invalid_data)

    def test_platform_enum(self):
        """Тест перечисления платформ"""
        order = OrderRequest(
            distance_in_meters=1000,
            duration_in_seconds=300,
            pickup_in_meters=500,
            pickup_in_seconds=100,
            driver_rating=4.5,
            user_rating=4.5,
            price_start_local=200,
            order_timestamp="2020-05-01T00:05:14",
            driver_platform="iOS",
            driver_reg_date="2019-01-01"
        )

        assert order.driver_platform == DriverPlatform.IOS


class TestResponseModels:
    """Тесты для моделей ответов"""

    def test_optimal_prices_response(self):
        """Тест OptimalPricesResponse"""
        price_points = [
            {
                "price": 200.0,
                "probability": 0.85,
                "expected_revenue": 170.0,
                "service_commission": 21.76,
                "driver_earnings": 148.24
            },
            {
                "price": 250.0,
                "probability": 0.70,
                "expected_revenue": 175.0,
                "service_commission": 22.4,
                "driver_earnings": 152.6
            }
        ]

        analysis = {
            "max_revenue_price": 250.0,
            "max_revenue_probability": 0.70,
            "max_probability_price": 200.0,
            "max_probability": 0.85,
            "recommendations": ["Рекомендуем 250.0₽: макс. доход 175.0₽"]
        }

        response = OptimalPricesResponse(
            price_curve=price_points,
            processing_time_ms=150.5,
            analysis=analysis
        )

        assert len(response.price_curve) == 2
        assert response.processing_time_ms == 150.5
        assert response.analysis["max_revenue_price"] == 250.0

    def test_health_response(self):
        """Тест HealthResponse"""
        health = HealthResponse(
            status="healthy",
            model_loaded=True,
            total_predictions=1000,
            cache_hit_rate=0.65,
            avg_processing_time=0.1,
            uptime_seconds=3600.0
        )

        assert health.status == "healthy"
        assert health.model_loaded is True
        assert health.total_predictions == 1000

    def test_service_stats(self):
        """Тест ServiceStats"""
        stats = ServiceStats(
            total_requests=5000,
            successful_requests=4950,
            failed_requests=50,
            avg_processing_time_ms=150.5,
            cache_hit_rate=0.72,
            memory_usage_mb=512.0,
            memory_available_mb=1536.0,
            cpu_usage_percent=25.5,
            total_service_revenue=125000.50,
            total_driver_earnings=850000.75,
            total_commission=125000.50,
            active_drivers=47,
            top_drivers=[
                {
                    "driver_id": "driver_001",
                    "driver_name": "Иван Петров",
                    "total_earnings": 45200.50,
                    "completed_orders": 187,
                    "rating": 4.9
                }
            ]
        )

        assert stats.total_requests == 5000
        assert stats.successful_requests == 4950
        assert stats.memory_usage_mb == 512.0
        assert stats.total_service_revenue == 125000.50
        assert len(stats.top_drivers) == 1