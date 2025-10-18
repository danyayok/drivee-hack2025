import pytest
from pydantic import ValidationError
from datetime import datetime

from app.models.data_models import (
    OrderRequest, OptimalPricesResponse, HealthResponse,
    ServiceStats, PricePoint, DriverPlatform
)


class TestOrderRequest:
    """Тесты модели OrderRequest"""

    def test_valid_order_request(self):
        """Тест создания валидного OrderRequest"""
        data = {
            "distance_in_meters": 2500.0,
            "duration_in_seconds": 600.0,
            "pickup_in_meters": 500.0,
            "pickup_in_seconds": 100.0,
            "driver_rating": 4.9,
            "user_rating": 4.8,
            "price_start_local": 190.0,
            "order_timestamp": "2020-05-15T18:30:00",
            "driver_platform": "android",
            "driver_reg_date": "2019-01-15",
            "carmodel": "Camry",
            "carname": "Toyota",
            "driver_id": "29368889",
            "user_id": "16458846"
        }

        order = OrderRequest(**data)
        assert order.distance_in_meters == 2500.0
        assert order.driver_platform == DriverPlatform.ANDROID
        assert order.driver_id == "29368889"

    def test_driver_platform_enum(self):
        """Тест enum платформы водителя"""
        # Разные варианты написания
        test_cases = [
            ("android", DriverPlatform.ANDROID),
            ("Android", DriverPlatform.ANDROID),
            ("ANDROID", DriverPlatform.ANDROID),
            ("ios", DriverPlatform.IOS),
            ("iOS", DriverPlatform.IOS),
        ]

        for platform_input, expected in test_cases:
            data = {
                "distance_in_meters": 2500.0,
                "duration_in_seconds": 600.0,
                "pickup_in_meters": 500.0,
                "pickup_in_seconds": 100.0,
                "driver_rating": 4.9,
                "user_rating": 4.8,
                "price_start_local": 190.0,
                "order_timestamp": "2020-05-15T18:30:00",
                "driver_platform": platform_input,
                "driver_reg_date": "2019-01-15"
            }

            order = OrderRequest(**data)
            assert order.driver_platform == expected

    def test_invalid_ratings(self):
        """Тест невалидных рейтингов"""
        invalid_cases = [
            {"driver_rating": -1.0},  # отрицательный
            {"driver_rating": 6.0},  # больше 5
            {"user_rating": -0.5},  # отрицательный
            {"user_rating": 5.1},  # больше 5
        ]

        base_data = {
            "distance_in_meters": 2500.0,
            "duration_in_seconds": 600.0,
            "pickup_in_meters": 500.0,
            "pickup_in_seconds": 100.0,
            "driver_rating": 4.9,
            "user_rating": 4.8,
            "price_start_local": 190.0,
            "order_timestamp": "2020-05-15T18:30:00",
            "driver_platform": "android",
            "driver_reg_date": "2019-01-15"
        }

        for invalid_case in invalid_cases:
            data = {**base_data, **invalid_case}
            with pytest.raises(ValidationError):
                OrderRequest(**data)

    def test_business_logic_validation(self):
        """Тест бизнес-логики валидации"""
        # pickup_in_meters > distance_in_meters
        invalid_data = {
            "distance_in_meters": 500.0,
            "duration_in_seconds": 600.0,
            "pickup_in_meters": 1000.0,  # больше чем дистанция
            "pickup_in_seconds": 100.0,
            "driver_rating": 4.9,
            "user_rating": 4.8,
            "price_start_local": 190.0,
            "order_timestamp": "2020-05-15T18:30:00",
            "driver_platform": "android",
            "driver_reg_date": "2019-01-15"
        }

        with pytest.raises(ValueError) as exc_info:
            OrderRequest(**invalid_data)
        assert "Дистанция подачи не может быть больше дистанции поездки" in str(exc_info.value)


class TestResponseModels:
    """Тесты моделей ответов"""

    def test_optimal_prices_response(self):
        """Тест модели OptimalPricesResponse"""
        price_points = [
            {
                "price": 200.0,
                "probability_percent": 70.5,
                "expected_revenue": 141.0,
                "service_earnings": 18.05,
                "driver_earnings": 122.95
            },
            {
                "price": 180.0,
                "probability_percent": 75.0,
                "expected_revenue": 135.0,
                "service_earnings": 17.28,
                "driver_earnings": 117.72
            }
        ]

        response = OptimalPricesResponse(
            price_curve=price_points,
            processing_time_ms=1200.5,
            analysis={
                "total_options": 2,
                "max_revenue_option": price_points[0]
            }
        )

        assert len(response.price_curve) == 2
        assert response.processing_time_ms == 1200.5
        assert response.analysis["total_options"] == 2

    def test_health_response(self):
        """Тест модели HealthResponse"""
        health = HealthResponse(
            status="healthy",
            model_loaded=True,
            total_predictions=1500,
            cache_hit_rate=0.65,
            avg_processing_time=0.245,
            uptime_seconds=86400.5
        )

        assert health.status == "healthy"
        assert health.model_loaded is True
        assert health.total_predictions == 1500
        assert health.cache_hit_rate == 0.65

    def test_service_stats(self):
        """Тест модели ServiceStats"""
        stats = ServiceStats(
            total_requests=1000,
            successful_requests=980,
            failed_requests=20,
            avg_processing_time_ms=245.5,
            cache_hit_rate=0.75,
            memory_usage_mb=512.0,
            memory_available_mb=1024.0,
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
                    "rating": 4.9,
                    "vehicle": "Toyota Camry"
                }
            ]
        )

        assert stats.total_requests == 1000
        assert stats.successful_requests == 980
        assert stats.failed_requests == 20
        assert stats.cache_hit_rate == 0.75


class TestPricePoint:
    """Тесты модели PricePoint"""

    def test_price_point_validation(self):
        """Тест валидации PricePoint"""
        # Валидные данные
        valid_data = {
            "price": 200.0,
            "probability_percent": 70.5,
            "expected_revenue": 141.0,
            "service_earnings": 18.05,
            "driver_earnings": 122.95
        }

        price_point = PricePoint(**valid_data)
        assert price_point.price == 200.0
        assert price_point.probability_percent == 70.5

    def test_invalid_probability_percent(self):
        """Тест невалидной вероятности"""
        invalid_cases = [
            {"probability_percent": -10.0},  # отрицательная
            {"probability_percent": 101.0},  # больше 100
        ]

        base_data = {
            "price": 200.0,
            "probability_percent": 70.5,
            "expected_revenue": 141.0,
            "service_earnings": 18.05,
            "driver_earnings": 122.95
        }

        for invalid_case in invalid_cases:
            data = {**base_data, **invalid_case}
            with pytest.raises(ValidationError):
                PricePoint(**data)