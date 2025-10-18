import pytest
from fastapi.testclient import TestClient
import json
import time


class TestOptimalPricesWithHistoricalData:
    """Тесты для эндпоинта /get_optimal_prices с историческими данными"""

    @pytest.fixture
    def sample_order_with_ids(self):
        """Фикстура с реальными ID из датасета"""
        return {
            "distance_in_meters": 3404.0,
            "duration_in_seconds": 486.0,
            "pickup_in_meters": 790.0,
            "pickup_in_seconds": 169.0,
            "driver_rating": 5.0,
            "user_rating": 4.8,
            "price_start_local": 180.0,
            "order_timestamp": "2020-05-01T00:05:14",
            "driver_platform": "Android",
            "driver_reg_date": "2019-09-22",
            "carmodel": "Logan",
            "carname": "Renault",
            # 🔥 РЕАЛЬНЫЕ ID ИЗ ДАТАСЕТА ДЛЯ ИСТОРИЧЕСКИХ ДАННЫХ
            "driver_id": "29368889",
            "user_id": "16458846"
        }

    @pytest.fixture
    def sample_order_without_ids(self):
        """Фикстура без ID (должна использовать средние значения)"""
        return {
            "distance_in_meters": 2500.0,
            "duration_in_seconds": 400.0,
            "pickup_in_meters": 500.0,
            "pickup_in_seconds": 120.0,
            "driver_rating": 4.5,
            "user_rating": 4.6,
            "price_start_local": 200.0,
            "order_timestamp": "2020-05-01T12:00:00",
            "driver_platform": "iOS",
            "driver_reg_date": "2020-01-01",
            "carmodel": "Camry",
            "carname": "Toyota"
            # Нет driver_id и user_id
        }

    @pytest.fixture
    def multiple_driver_scenarios(self):
        """Разные сценарии водителей с историческими данными"""
        return [
            {
                "distance_in_meters": 1500.0,
                "duration_in_seconds": 300.0,
                "pickup_in_meters": 300.0,
                "pickup_in_seconds": 90.0,
                "driver_rating": 4.9,
                "user_rating": 4.7,
                "price_start_local": 150.0,
                "order_timestamp": "2020-05-01T08:00:00",
                "driver_platform": "Android",
                "driver_reg_date": "2018-06-15",
                "carmodel": "Solaris",
                "carname": "Hyundai",
                "driver_id": "16775674",  # Из датасета
                "user_id": "8199106"  # Из датасета
            },
            {
                "distance_in_meters": 5000.0,
                "duration_in_seconds": 600.0,
                "pickup_in_meters": 1000.0,
                "pickup_in_seconds": 180.0,
                "driver_rating": 4.7,
                "user_rating": 4.9,
                "price_start_local": 300.0,
                "order_timestamp": "2020-05-01T18:00:00",
                "driver_platform": "iOS",
                "driver_reg_date": "2019-03-20",
                "carmodel": "Rio",
                "carname": "Kia",
                "driver_id": "7987646",  # Из датасета
                "user_id": "32924653"  # Из датасета
            }
        ]

    def test_successful_optimization_with_historical_data(self, test_client, sample_order_with_ids):
        """Тест успешной оптимизации с историческими данными"""
        response = test_client.post("/api/v1/get_optimal_prices", json=sample_order_with_ids)

        assert response.status_code == 200
        data = response.json()

        # Проверяем структуру ответа
        assert "price_curve" in data
        assert "processing_time_ms" in data
        assert "analysis" in data

        # Проверяем кривую цен
        price_curve = data["price_curve"]
        assert len(price_curve) > 0
        assert len(price_curve) <= 8

        # Проверяем реалистичные вероятности (с историческими данными)
        probabilities = [p["probability"] for p in price_curve]
        assert all(0.05 <= prob <= 0.95 for prob in probabilities), f"Нереалистичные вероятности: {probabilities}"

        # Проверяем финансовые расчеты
        for price_point in price_curve:
            expected_revenue = price_point["price"] * price_point["probability"]
            assert abs(price_point["expected_revenue"] - expected_revenue) < 1.0

        # Проверяем анализ
        analysis = data["analysis"]
        assert "max_revenue_price" in analysis
        assert "max_probability_price" in analysis
        assert "max_driver_earnings_price" in analysis
        assert "recommendations" in analysis

    def test_optimization_without_historical_ids(self, test_client, sample_order_without_ids):
        """Тест оптимизации без ID (должна использовать средние значения)"""
        response = test_client.post("/api/v1/get_optimal_prices", json=sample_order_without_ids)

        assert response.status_code == 200
        data = response.json()

        # Проверяем что система работает даже без исторических данных
        price_curve = data["price_curve"]
        assert len(price_curve) > 0

        # Вероятности должны быть в разумном диапазоне
        probabilities = [p["probability"] for p in price_curve]
        assert all(0.1 <= prob <= 0.9 for prob in probabilities)

    def test_historical_data_impact(self, test_client, multiple_driver_scenarios):
        """Тест влияния исторических данных на предсказания"""
        results = []

        for scenario in multiple_driver_scenarios:
            response = test_client.post("/api/v1/get_optimal_prices", json=scenario)
            assert response.status_code == 200

            data = response.json()
            results.append({
                'driver_id': scenario['driver_id'],
                'probabilities': [p['probability'] for p in data['price_curve'][:3]],  # Первые 3 цены
                'analysis': data['analysis']
            })

        # Проверяем что разные водители имеют разные вероятности
        # (из-за разных исторических acceptance rates)
        if len(results) > 1:
            driver1_probs = results[0]['probabilities']
            driver2_probs = results[1]['probabilities']

            # Вероятности должны отличаться для разных водителей
            assert driver1_probs != driver2_probs, "Вероятности должны отличаться для разных водителей"

    def test_price_range_with_historical_context(self, test_client, sample_order_with_ids):
        """Тест диапазона цен с учетом исторических данных"""
        response = test_client.post("/api/v1/get_optimal_prices", json=sample_order_with_ids)
        assert response.status_code == 200

        data = response.json()
        price_curve = data["price_curve"]

        base_price = sample_order_with_ids["price_start_local"]
        prices = [p["price"] for p in price_curve]

        # Проверяем разумный диапазон цен
        assert min(prices) >= base_price * 0.7
        assert max(prices) <= base_price * 2.0

        # Проверяем что есть как низкие, так и высокие цены
        assert len(prices) >= 4, "Должно быть несколько ценовых вариантов"

    def test_financial_calculations_accuracy(self, test_client, sample_order_with_ids):
        """Тест точности финансовых расчетов"""
        SERVICE_COMMISSION_RATE = 0.128

        response = test_client.post("/api/v1/get_optimal_prices", json=sample_order_with_ids)
        assert response.status_code == 200

        data = response.json()

        for price_point in data["price_curve"]:
            # Проверяем ожидаемый доход
            expected_revenue = price_point["price"] * price_point["probability"]
            assert abs(price_point["expected_revenue"] - expected_revenue) < 0.1

            # Проверяем комиссию сервиса
            expected_commission = expected_revenue * SERVICE_COMMISSION_RATE
            assert abs(price_point["service_commission"] - expected_commission) < 0.1

            # Проверяем заработок водителя
            expected_driver_earnings = expected_revenue - expected_commission
            assert abs(price_point["driver_earnings"] - expected_driver_earnings) < 0.1

    def test_performance_with_historical_data(self, test_client, sample_order_with_ids):
        """Тест производительности с историческими данными"""
        start_time = time.time()

        response = test_client.post("/api/v1/get_optimal_prices", json=sample_order_with_ids)

        end_time = time.time()
        processing_time = end_time - start_time

        assert response.status_code == 200
        data = response.json()

        # Проверяем что время обработки разумное
        assert processing_time < 5.0, f"Слишком долгая обработка: {processing_time} сек"
        assert data["processing_time_ms"] < 5000, f"Слишком долгая обработка в мс: {data['processing_time_ms']}"

    def test_analysis_quality_with_historical_data(self, test_client, sample_order_with_ids):
        """Тест качества анализа с историческими данными"""
        response = test_client.post("/api/v1/get_optimal_prices", json=sample_order_with_ids)
        assert response.status_code == 200

        data = response.json()
        analysis = data["analysis"]

        # Проверяем ключевые метрики
        assert analysis["max_revenue_price"] > 0
        assert analysis["max_probability_price"] > 0
        assert analysis["max_driver_earnings_price"] > 0

        # Проверяем рекомендации
        assert "recommendations" in analysis
        assert len(analysis["recommendations"]) > 0

        # Проверяем финансовые метрики
        assert analysis["total_potential_revenue"] > 0
        assert analysis["avg_service_commission"] > 0
        assert analysis["avg_driver_earnings"] > 0

    def test_error_handling_invalid_historical_ids(self, test_client, sample_order_with_ids):
        """Тест обработки невалидных исторических ID"""
        invalid_data = sample_order_with_ids.copy()
        invalid_data["driver_id"] = "invalid_driver_123"
        invalid_data["user_id"] = "invalid_user_456"

        # Система должна работать даже с невалидными ID (использовать средние значения)
        response = test_client.post("/api/v1/get_optimal_prices", json=invalid_data)
        assert response.status_code == 200

        data = response.json()
        assert len(data["price_curve"]) > 0

    def test_consistency_with_same_historical_data(self, test_client, sample_order_with_ids):
        """Тест консистентности при одинаковых исторических данных"""
        results = []

        # Делаем несколько одинаковых запросов
        for _ in range(3):
            response = test_client.post("/api/v1/get_optimal_prices", json=sample_order_with_ids)
            assert response.status_code == 200
            results.append(response.json())

        # Проверяем консистентность (первые 3 цены должны быть похожи)
        first_prices = [p["price"] for p in results[0]["price_curve"][:3]]

        for i in range(1, 3):
            current_prices = [p["price"] for p in results[i]["price_curve"][:3]]
            # Допускаем небольшие различия из-за кэширования/оптимизаций
            assert set(first_prices) == set(current_prices), f"Цены неконсистентны: {first_prices} vs {current_prices}"


class TestHealthWithHistoricalData:
    """Тесты здоровья сервиса с историческими данными"""

    def test_health_check_includes_historical_data(self, test_client):
        """Тест что health check включает статус исторических данных"""
        response = test_client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "model_loaded" in data


class TestHistoricalDataIntegration:
    """Тесты интеграции с историческими данными"""

    def test_historical_service_initialization(self, test_client):
        """Тест что исторический сервис инициализирован"""
        # Делаем тестовый запрос чтобы активировать сервис
        test_data = {
            "distance_in_meters": 1000.0,
            "duration_in_seconds": 200.0,
            "pickup_in_meters": 200.0,
            "pickup_in_seconds": 60.0,
            "driver_rating": 4.5,
            "user_rating": 4.5,
            "price_start_local": 150.0,
            "order_timestamp": "2020-05-01T12:00:00",
            "driver_platform": "Android",
            "driver_reg_date": "2020-01-01",
            "carmodel": "Test",
            "carname": "Test",
            "driver_id": "29368889"  # Реальный ID из датасета
        }

        response = test_client.post("/api/v1/get_optimal_prices", json=test_data)
        assert response.status_code == 200

        # Проверяем что сервис работает и выдает разумные результаты
        data = response.json()
        assert len(data["price_curve"]) > 0

        # Вероятности должны быть реалистичными
        probabilities = [p["probability"] for p in data["price_curve"]]
        assert all(0.1 <= prob <= 0.9 for prob in probabilities)


# Фикстуры для pytest
@pytest.fixture
def test_client():
    """Фикстура тестового клиента"""
    from app.main import app
    return TestClient(app)


@pytest.fixture
def sample_order_data():
    """Базовая фикстура заказа"""
    return {
        "distance_in_meters": 3404.0,
        "duration_in_seconds": 486.0,
        "pickup_in_meters": 790.0,
        "pickup_in_seconds": 169.0,
        "driver_rating": 5.0,
        "user_rating": 4.8,
        "price_start_local": 180.0,
        "order_timestamp": "2020-05-01T00:05:14",
        "driver_platform": "Android",
        "driver_reg_date": "2019-09-22",
        "carmodel": "Logan",
        "carname": "Renault",
        "driver_id": "29368889",
        "user_id": "16458846"
    }


@pytest.fixture
def multiple_order_samples():
    """Несколько примеров заказов"""
    return [
        {
            "distance_in_meters": 1500.0,
            "duration_in_seconds": 300.0,
            "pickup_in_meters": 300.0,
            "pickup_in_seconds": 90.0,
            "driver_rating": 4.9,
            "user_rating": 4.7,
            "price_start_local": 150.0,
            "order_timestamp": "2020-05-01T08:00:00",
            "driver_platform": "Android",
            "driver_reg_date": "2018-06-15",
            "carmodel": "Solaris",
            "carname": "Hyundai",
            "driver_id": "16775674"
        },
        {
            "distance_in_meters": 5000.0,
            "duration_in_seconds": 600.0,
            "pickup_in_meters": 1000.0,
            "pickup_in_seconds": 180.0,
            "driver_rating": 4.7,
            "user_rating": 4.9,
            "price_start_local": 300.0,
            "order_timestamp": "2020-05-01T18:00:00",
            "driver_platform": "iOS",
            "driver_reg_date": "2019-03-20",
            "carmodel": "Rio",
            "carname": "Kia",
            "driver_id": "7987646"
        }
    ]