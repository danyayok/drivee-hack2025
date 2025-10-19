import pytest
import sqlite3
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from app.api.endpoints import StatsService, init_stats_db


class TestStatisticsService:
    """Тесты сервиса статистики"""

    @pytest.fixture
    def stats_service(self, tmp_path):
        """Создание тестового сервиса статистики"""
        db_path = tmp_path / "test_stats.db"

        # ✅ ПЕРЕДАЕМ ПУТЬ К БД В init_stats_db
        original_db_path = "stats.db"  # Сохраняем оригинальный путь

        # Временно меняем глобальную переменную db_path в endpoints.py
        import app.api.endpoints as endpoints_module
        endpoints_module.STATS_DB_PATH = str(db_path)

        # Инициализируем БД с правильным путем
        init_stats_db()

        service = StatsService()
        service.db_path = str(db_path)  # ✅ Устанавливаем путь для сервиса

        yield service

        # Восстанавливаем оригинальный путь
        endpoints_module.STATS_DB_PATH = original_db_path

    @pytest.fixture
    def sample_order_data(self):
        return {
            'order_id': 'test_order_123',
            'driver_id': '29368889',
            'user_id': '16458846',
            'price_start_local': 200.0,
            'price_bid_local': 240.0,
            'final_price': 240.0,
            'service_commission': 30.72,
            'driver_earnings': 209.28,
            'distance_in_meters': 3500.0,
            'duration_in_seconds': 600.0
        }

    def test_record_completed_order(self, stats_service, sample_order_data):
        """Тест записи завершенного заказа"""
        stats_service.record_completed_order(sample_order_data)

        # Проверяем что данные записались в БД
        conn = sqlite3.connect(stats_service.db_path)
        cursor = conn.cursor()

        # Проверяем completed_orders
        cursor.execute("SELECT * FROM completed_orders WHERE order_id = ?",
                       (sample_order_data['order_id'],))
        order = cursor.fetchone()
        assert order is not None
        assert order[6] == sample_order_data['final_price']  # final_price

        # Проверяем active_drivers
        cursor.execute("SELECT * FROM active_drivers WHERE driver_id = ?",
                       (sample_order_data['driver_id'],))
        driver = cursor.fetchone()
        assert driver is not None
        assert driver[2] == sample_order_data['driver_earnings']  # total_earnings
        assert driver[3] == 1  # completed_orders

        # Проверяем financial_stats
        today = datetime.now().date().isoformat()
        cursor.execute("SELECT * FROM financial_stats WHERE date = ?", (today,))
        financial = cursor.fetchone()
        assert financial is not None
        assert financial[1] == sample_order_data['service_commission']  # total_service_revenue

        conn.close()

    def test_get_financial_stats(self, stats_service, sample_order_data):
        """Тест получения финансовой статистики"""
        # Записываем тестовые заказы
        stats_service.record_completed_order(sample_order_data)

        # Получаем статистику
        stats = stats_service.get_financial_stats(days=30)

        assert 'total_service_revenue' in stats
        assert 'total_driver_earnings' in stats
        assert 'total_commission' in stats
        assert 'today_service_revenue' in stats
        assert 'today_driver_earnings' in stats
        assert 'today_orders' in stats
        assert 'avg_order_value' in stats

        # ✅ Проверяем РЕАЛЬНЫЕ вычисления из БД, а не fallback
        assert stats['total_service_revenue'] == pytest.approx(30.72)
        assert stats['total_driver_earnings'] == pytest.approx(209.28)
        assert stats['today_orders'] == 1
        assert stats['avg_order_value'] == pytest.approx(30.72)

    def test_get_driver_stats(self, stats_service, sample_order_data):
        """Тест получения статистики водителей"""
        # Записываем тестового водителя
        stats_service.record_completed_order(sample_order_data)

        stats = stats_service.get_driver_stats()

        assert 'active_drivers' in stats
        assert 'total_drivers' in stats
        assert 'top_drivers' in stats

        # ✅ Проверяем что водитель попал в топ (должен быть 1 водитель)
        assert len(stats['top_drivers']) == 1
        top_driver = stats['top_drivers'][0]
        assert top_driver['driver_id'] == '29368889'
        assert top_driver['total_earnings'] == pytest.approx(209.28)
        assert top_driver['completed_orders'] == 1

    def test_get_realtime_stats(self, stats_service, sample_order_data):
        """Тест получения статистики в реальном времени"""
        stats_service.record_completed_order(sample_order_data)

        stats = stats_service.get_realtime_stats()

        # Проверяем что все метрики присутствуют
        assert 'current_hour_orders' in stats
        assert 'avg_order_completion_time' in stats
        assert 'popular_routes' in stats
        assert 'total_service_revenue' in stats
        assert 'active_drivers' in stats

        # Проверяем вычисление времени выполнения
        assert stats['avg_order_completion_time'] == pytest.approx(10.0)  # 600 сек = 10 мин

    def test_multiple_orders_same_driver(self, stats_service, sample_order_data):
        """Тест нескольких заказов одного водителя"""
        # Первый заказ
        stats_service.record_completed_order(sample_order_data)

        # Второй заказ того же водителя
        second_order = sample_order_data.copy()
        second_order['order_id'] = 'test_order_456'
        second_order['service_commission'] = 25.6
        second_order['driver_earnings'] = 174.4

        stats_service.record_completed_order(second_order)

        # Проверяем агрегацию
        driver_stats = stats_service.get_driver_stats()
        financial_stats = stats_service.get_financial_stats()

        # ✅ Проверяем что earnings суммируются (РЕАЛЬНЫЕ данные из БД)
        top_driver = driver_stats['top_drivers'][0]
        assert top_driver['total_earnings'] == pytest.approx(209.28 + 174.4)
        assert top_driver['completed_orders'] == 2

        # Проверяем финансовую статистику
        assert financial_stats['total_service_revenue'] == pytest.approx(30.72 + 25.6)
        assert financial_stats['today_orders'] == 2

    def test_fallback_stats(self):
        """Тест fallback статистики при ошибках"""
        # ✅ Создаем сервис с невалидным путем к БД
        service = StatsService()
        service.db_path = "/invalid/path/stats.db"

        # Должны вернуться fallback значения
        financial_stats = service.get_financial_stats()
        driver_stats = service.get_driver_stats()

        # ✅ Проверяем fallback значения
        assert financial_stats['total_service_revenue'] == 125000.50
        assert driver_stats['active_drivers'] == 47
        assert len(driver_stats['top_drivers']) == 2  # ✅ В fallback 2 водителя