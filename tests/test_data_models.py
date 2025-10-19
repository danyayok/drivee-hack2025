import pytest
from datetime import datetime
from pydantic import ValidationError

from app.models.data_models import (
    OrderRequest, OptimalPricesResponse, HealthResponse, ServiceStats,
    PricePoint, DriverPlatform
)


class TestDataModels:
    """Тесты моделей данных Pydantic"""

    def test_order_request_validation(self):
        """Тест валидации OrderRequest"""
        # Валидные данные
        valid_data = {
            "distance_in_meters": 3500.0,
            "duration_in_seconds": 600.0,
            "pickup_in_meters": 500.0,
            "pickup_in_seconds": 120.0,
            "driver_rating": 4.8,
            "user_rating": 4.9,
            "price_start_local": 200.0,
            "order_timestamp": "2024-01-15T12:00:00",
            "driver_platform": "android",
            "driver_reg_date": "2023-01-01"
        }

        order = OrderRequest(**valid_data)
        assert order.distance_in_meters == 3500.0
        assert order.driver_platform == DriverPlatform.ANDROID

    def test_order_request_validation_errors(self):
        """Тест ошибок валидации OrderRequest"""
        # Невалидные данные - отрицательное расстояние
        invalid_data = {
            "distance_in_meters": -100.0,  # Отрицательное значение
            "duration_in_seconds": 600.0,
            "pickup_in_meters": 500.0,
            "pickup_in_seconds": 120.0,
            "driver_rating": 4.8,
            "user_rating": 4.9,
            "price_start_local": 200.0,
            "order_timestamp": "2024-01-15T12:00:00"
        }

        with pytest.raises(ValidationError):
            OrderRequest(**invalid_data)

    def test_order_request_business_logic_validation(self):
        """Тест бизнес-логики валидации"""
        # Дистанция подачи больше дистанции поездки
        invalid_data = {
            "distance_in_meters": 1000.0,
            "duration_in_seconds": 600.0,
            "pickup_in_meters": 1500.0,  # Больше чем distance_in_meters
            "pickup_in_seconds": 120.0,
            "driver_rating": 4.8,
            "user_rating": 4.9,
            "price_start_local": 200.0,
            "order_timestamp": "2024-01-15T12:00:00"
        }

        with pytest.raises(ValidationError):
            OrderRequest(**invalid_data)

    def test_driver_platform_enum(self):
        """Тест enum платформы водителя"""
        # Проверяем разные варианты написания
        assert DriverPlatform("ios") == DriverPlatform.IOS
        assert DriverPlatform("iOS") == DriverPlatform.IOS
        assert DriverPlatform("Android") == DriverPlatform.ANDROID
        assert DriverPlatform("android") == DriverPlatform.ANDROID

        # Проверяем регистронезависимость
        assert DriverPlatform("ANDROID") == DriverPlatform.ANDROID
        assert DriverPlatform("IOS") == DriverPlatform.IOS

    def test_optimal_prices_response(self):
        """Тест модели OptimalPricesResponse"""
        price_points = [
            PricePoint(
                price=240.0,
                probability_percent=70.5,
                expected_revenue=168.0,
                service_earnings=21.5,
                driver_earnings=146.5
            ),
            PricePoint(
                price=220.0,
                probability_percent=75.0,
                expected_revenue=165.0,
                service_earnings=21.1,
                driver_earnings=143.9
            )
        ]

        response = OptimalPricesResponse(
            price_curve=price_points,
            processing_time_ms=245.5,
            analysis={
                "total_options": 2,
                "max_revenue_option": price_points[0].model_dump()
            }
        )

        assert len(response.price_curve) == 2
        assert response.processing_time_ms == 245.5
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
        assert health.model_loaded == True
        assert health.total_predictions == 1500
        assert health.cache_hit_rate == 0.65

    def test_service_stats(self):
        """Тест модели ServiceStats"""
        stats = ServiceStats(
            total_requests=1000,
            successful_requests=980,
            failed_requests=20,
            avg_processing_time_ms=2.19,
            cache_hit_rate=0.65,
            memory_usage_mb=512.5,
            memory_available_mb=2048.0,
            cpu_usage_percent=15.2,
            total_service_revenue=125000.50,
            total_driver_earnings=850000.75,
            total_commission=125000.50,
            today_service_revenue=1250.75,
            today_driver_earnings=8520.25,
            today_orders=15,
            avg_order_value=278.57,
            active_drivers=47,
            total_drivers=156,
            top_drivers=[
                {
                    'driver_id': 'driver_001',
                    'driver_name': 'Иван Петров',
                    'total_earnings': 45200.50,
                    'completed_orders': 187,
                    'rating': 4.9,
                    'vehicle': 'Toyota Camry'
                }
            ],
            current_hour_orders=5,
            avg_order_completion_time=12.5,
            popular_routes=[
                {'from': 'Центр', 'to': 'Аэропорт', 'orders_count': 45}
            ]
        )

        assert stats.total_requests == 1000
        assert stats.today_orders == 15
        assert stats.active_drivers == 47
        assert len(stats.top_drivers) == 1

    def test_price_point_validation(self):
        """Тест валидации PricePoint"""
        # Валидные данные
        valid_point = PricePoint(
            price=240.0,
            probability_percent=70.0,  # ✅ Исправлено: должно быть ≤ 100
            expected_revenue=168.0,
            service_earnings=21.5,
            driver_earnings=146.5
        )
        assert valid_point.probability_percent == 70.0

        # Невалидные данные - probability_percent > 100
        with pytest.raises(ValidationError):
            PricePoint(
                price=240.0,
                probability_percent=105.0,  # > 100% - должно вызвать ошибку
                expected_revenue=168.0,
                service_earnings=21.5,
                driver_earnings=146.5
            )

    def test_timestamp_normalization(self):
        """Тест нормализации timestamp"""
        # Unix timestamp
        data_with_timestamp = {
            "distance_in_meters": 3500.0,
            "duration_in_seconds": 600.0,
            "pickup_in_meters": 500.0,
            "pickup_in_seconds": 120.0,
            "driver_rating": 4.8,
            "user_rating": 4.9,
            "price_start_local": 200.0,
            "order_timestamp": 1705315200,  # Unix timestamp
            "driver_platform": "android"
        }

        order = OrderRequest(**data_with_timestamp)
        # Проверяем что timestamp преобразован в ISO строку
        assert "T" in order.order_timestamp  # Должен содержать 'T' как в ISO формате

    def test_optional_fields(self):
        """Тест опциональных полей"""
        minimal_data = {
            "distance_in_meters": 3500.0,
            "duration_in_seconds": 600.0,
            "pickup_in_meters": 500.0,
            "pickup_in_seconds": 120.0,
            "driver_rating": 4.8,
            "user_rating": 4.9,
            "price_start_local": 200.0,
            "order_timestamp": "2024-01-15T12:00:00"
        }

        # Должен работать без опциональных полей
        order = OrderRequest(**minimal_data)
        assert order.carname is None
        assert order.carmodel is None
        assert order.driver_id is None
        assert order.user_id is None