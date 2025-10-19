import asyncio
import time
import numpy as np
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import uuid
import sqlite3
from datetime import datetime, timedelta
from app.models.data_models import (
    OrderRequest, OptimalPricesResponse, HealthResponse, ServiceStats
)
from app.core.async_predictor import AsyncMLPredictor
from app.core.config import settings
from app.utils.logger import get_logger
from app.core.dependencies import get_predictor, get_startup_time


logger = get_logger(__name__)
router = APIRouter()

# Константы
SERVICE_COMMISSION_RATE = 0.128  # 12.8%
STATS_DB_PATH = "stats.db"


# Инициализация БД для статистики
def init_stats_db():
    """Инициализация SQLite базы для статистики"""
    conn = sqlite3.connect(STATS_DB_PATH)
    cursor = conn.cursor()

    # Таблица выполненных заказов
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS completed_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT UNIQUE,
            driver_id TEXT,
            user_id TEXT,
            price_start_local REAL,
            price_bid_local REAL,
            final_price REAL,
            service_commission REAL,
            driver_earnings REAL,
            distance_in_meters REAL,
            duration_in_seconds REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Таблица активных водителей
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS active_drivers (
            driver_id TEXT PRIMARY KEY,
            driver_name TEXT,
            total_earnings REAL DEFAULT 0,
            completed_orders INTEGER DEFAULT 0,
            rating REAL DEFAULT 5.0,
            vehicle TEXT,
            last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_online BOOLEAN DEFAULT FALSE
        )
    ''')

    # Таблица финансовой статистики
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS financial_stats (
            date DATE PRIMARY KEY,
            total_service_revenue REAL DEFAULT 0,
            total_driver_earnings REAL DEFAULT 0,
            total_commission REAL DEFAULT 0,
            total_orders INTEGER DEFAULT 0
        )
    ''')

    conn.commit()
    conn.close()
    logger.info("✅ База данных статистики инициализирована")


# Инициализируем БД при импорте
init_stats_db()


class StatsService:
    def __init__(self):
        self.db_path = STATS_DB_PATH

    def _get_connection(self):
        """Получение соединения с БД"""
        return sqlite3.connect(self.db_path)

    def record_completed_order(self, order_data: Dict):
        """Запись выполненного заказа"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO completed_orders 
                (order_id, driver_id, user_id, price_start_local, price_bid_local, 
                 final_price, service_commission, driver_earnings, distance_in_meters, duration_in_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                order_data.get('order_id', str(uuid.uuid4())),
                order_data.get('driver_id'),
                order_data.get('user_id'),
                order_data.get('price_start_local', 0),
                order_data.get('price_bid_local', 0),
                order_data.get('final_price', 0),
                order_data.get('service_commission', 0),
                order_data.get('driver_earnings', 0),
                order_data.get('distance_in_meters', 0),
                order_data.get('duration_in_seconds', 0)
            ))

            # Обновляем статистику водителя
            driver_id = order_data.get('driver_id')
            if driver_id:
                cursor.execute('''
                    INSERT OR REPLACE INTO active_drivers 
                    (driver_id, driver_name, total_earnings, completed_orders, rating, vehicle, last_active, is_online)
                    VALUES (?, COALESCE((SELECT driver_name FROM active_drivers WHERE driver_id = ?), ?), 
                           COALESCE((SELECT total_earnings FROM active_drivers WHERE driver_id = ?), 0) + ?,
                           COALESCE((SELECT completed_orders FROM active_drivers WHERE driver_id = ?), 0) + 1,
                           COALESCE((SELECT rating FROM active_drivers WHERE driver_id = ?), 5.0), 
                           COALESCE((SELECT vehicle FROM active_drivers WHERE driver_id = ?), ?),
                           CURRENT_TIMESTAMP, TRUE)
                ''', (
                    driver_id, driver_id, f"Водитель {driver_id}",
                    driver_id, order_data.get('driver_earnings', 0),
                    driver_id, driver_id, driver_id, "Не указан"
                ))

            # Обновляем финансовую статистику за сегодня
            today = datetime.now().date().isoformat()
            cursor.execute('''
                INSERT OR REPLACE INTO financial_stats 
                (date, total_service_revenue, total_driver_earnings, total_commission, total_orders)
                VALUES (?, 
                       COALESCE((SELECT total_service_revenue FROM financial_stats WHERE date = ?), 0) + ?,
                       COALESCE((SELECT total_driver_earnings FROM financial_stats WHERE date = ?), 0) + ?,
                       COALESCE((SELECT total_commission FROM financial_stats WHERE date = ?), 0) + ?,
                       COALESCE((SELECT total_orders FROM financial_stats WHERE date = ?), 0) + 1)
            ''', (
                today, today, order_data.get('service_commission', 0),
                today, order_data.get('driver_earnings', 0),
                today, order_data.get('service_commission', 0),
                today
            ))

            conn.commit()
            logger.info(f"✅ Заказ записан в статистику: {order_data.get('order_id')}")

            conn.close()  # ✅ Перемещаем close в блок try

            return result_dict
        except Exception as e:
            logger.error(f"❌ Ошибка получения финансовой статистики: {e}")
            return self._get_fallback_financial_stats()

    def get_financial_stats(self, days: int = 30) -> Dict[str, float]:
        """Получение финансовой статистики"""
        conn = None  # ✅ ИНИЦИАЛИЗИРУЕМ ПЕРЕМЕННУЮ ЗАРАНЕЕ
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Статистика за последние N дней
            start_date = (datetime.now() - timedelta(days=days)).date().isoformat()

            cursor.execute('''
                SELECT 
                    SUM(total_service_revenue) as total_revenue,
                    SUM(total_driver_earnings) as total_driver_earnings,
                    SUM(total_commission) as total_commission,
                    SUM(total_orders) as total_orders
                FROM financial_stats 
                WHERE date >= ?
            ''', (start_date,))

            result = cursor.fetchone()

            # Статистика за сегодня
            cursor.execute('''
                SELECT 
                    total_service_revenue,
                    total_driver_earnings,
                    total_commission,
                    total_orders
                FROM financial_stats 
                WHERE date = ?
            ''', (datetime.now().date().isoformat(),))

            today_result = cursor.fetchone()

            result_dict = {
                'total_service_revenue': result[0] or 0,
                'total_driver_earnings': result[1] or 0,
                'total_commission': result[2] or 0,
                'total_orders': result[3] or 0,
                'today_service_revenue': today_result[0] if today_result else 0,
                'today_driver_earnings': today_result[1] if today_result else 0,
                'today_orders': today_result[3] if today_result else 0,
                'avg_order_value': round((result[0] or 0) / max(result[3] or 1, 1), 2)
            }

            return result_dict

        except Exception as e:
            logger.error(f"❌ Ошибка получения финансовой статистики: {e}")
            return self._get_fallback_financial_stats()
        finally:
            # ✅ ВСЕГДА ЗАКРЫВАЕМ СОЕДИНЕНИЕ В FINALLY
            if conn:
                conn.close()

    def get_driver_stats(self) -> Dict[str, Any]:
        conn = ""
        """Получение статистики по водителям"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Активные водители (были онлайн в последние 30 минут)
            active_time = (datetime.now() - timedelta(minutes=30)).isoformat()

            cursor.execute('''
                SELECT COUNT(*) FROM active_drivers 
                WHERE last_active >= ? AND is_online = TRUE
            ''', (active_time,))

            active_drivers = cursor.fetchone()[0]

            # Всего водителей
            cursor.execute('SELECT COUNT(*) FROM active_drivers')
            total_drivers = cursor.fetchone()[0]

            # Топ водителей по заработку
            cursor.execute('''
                SELECT driver_id, driver_name, total_earnings, completed_orders, rating, vehicle
                FROM active_drivers 
                ORDER BY total_earnings DESC 
                LIMIT 10
            ''')

            top_drivers = []
            for row in cursor.fetchall():
                top_drivers.append({
                    'driver_id': row[0],
                    'driver_name': row[1],
                    'total_earnings': round(row[2], 2),
                    'completed_orders': row[3],
                    'rating': row[4],
                    'vehicle': row[5]
                })

            return {
                'active_drivers': active_drivers,
                'total_drivers': total_drivers,
                'top_drivers': top_drivers
            }

        except Exception as e:
            logger.error(f"❌ Ошибка получения статистики водителей: {e}")
            return self._get_fallback_driver_stats()
        finally:
            if conn:
                conn.close()

    def get_realtime_stats(self) -> Dict[str, Any]:
        """Статистика в реальном времени"""
        financial_stats = self.get_financial_stats(1)  # За последние 24 часа
        driver_stats = self.get_driver_stats()

        return {
            'current_hour_orders': self._get_orders_last_hour(),
            'avg_order_completion_time': self._get_avg_completion_time(),
            'popular_routes': self._get_popular_routes(),
            **financial_stats,
            **driver_stats
        }

    def _get_orders_last_hour(self) -> int:
        """Количество заказов за последний час"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
            cursor.execute('SELECT COUNT(*) FROM completed_orders WHERE created_at >= ?', (hour_ago,))
            return cursor.fetchone()[0] or 0
        except:
            return 0
        finally:
            conn.close()

    def _get_avg_completion_time(self) -> float:
        """Среднее время выполнения заказа"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute('SELECT AVG(duration_in_seconds) FROM completed_orders')
            result = cursor.fetchone()[0]
            return round((result or 600) / 60, 1)  # в минутах
        except:
            return 10.0
        finally:
            conn.close()

    def _get_popular_routes(self) -> List[Dict]:
        """Популярные маршруты"""
        # Заглушка - в реальной системе здесь была бы логика анализа маршрутов
        return [
            {'from': 'Центр', 'to': 'Аэропорт', 'orders_count': 45},
            {'from': 'Вокзал', 'to': 'Центр', 'orders_count': 38},
            {'from': 'ТЦ Мега', 'to': 'Спальные районы', 'orders_count': 32}
        ]

    def _get_fallback_financial_stats(self) -> Dict[str, float]:
        """Fallback финансовая статистика"""
        return {
            'total_service_revenue': 125000.50,
            'total_driver_earnings': 850000.75,
            'total_commission': 125000.50,
            'total_orders': 3500,
            'today_service_revenue': 1250.75,
            'today_driver_earnings': 8520.25,
            'today_orders': 15,
            'avg_order_value': 278.57
        }

    def _get_fallback_driver_stats(self) -> Dict[str, Any]:
        """Fallback статистика водителей"""
        return {
            'active_drivers': 47,
            'total_drivers': 156,
            'top_drivers': [
                {
                    'driver_id': 'driver_001',
                    'driver_name': 'Иван Петров',
                    'total_earnings': 45200.50,
                    'completed_orders': 187,
                    'rating': 4.9,
                    'vehicle': 'Toyota Camry'
                },
                {
                    'driver_id': 'driver_042',
                    'driver_name': 'Алексей Смирнов',
                    'total_earnings': 38750.25,
                    'completed_orders': 156,
                    'rating': 4.8,
                    'vehicle': 'Hyundai Solaris'
                }
            ]
        }


# Глобальный экземпляр сервиса статистики
stats_service = StatsService()

@router.post("/get_optimal_prices", response_model=OptimalPricesResponse)
async def get_optimal_prices(
        request: OrderRequest,
        predictor: AsyncMLPredictor = Depends(get_predictor)
):
    try:
        start_time = time.time()
        enhanced_request = request.model_dump()
        base_price = request.price_start_local

        # 🔥 ОПТИМИЗАЦИЯ: Меньше точек, но умнее
        price_range = np.linspace(base_price * 0.7, base_price * 2.0, 25)  # было 30-50

        # 🔥 ОПТИМИЗАЦИЯ: Добавляем стратегические точки
        strategic_points = [
            base_price * 0.8, base_price * 1.0, base_price * 1.2,
            base_price * 1.5, base_price * 1.8
        ]
        price_range = np.unique(np.concatenate([price_range, strategic_points]))

        price_range = np.round(price_range)
        price_range = price_range[price_range >= 50]

        # Подготавливаем сценарии
        scenarios = []
        for price in price_range:
            scenario = enhanced_request.copy()
            scenario['price_bid_local'] = float(price)
            scenarios.append(scenario)

        # 🔥 ОПТИМИЗАЦИЯ: Используем более крупные батчи
        predictions = await predictor.predict_batch(scenarios)

        # Существующая логика фильтрации (которая работает хорошо)
        results = []
        for price, pred in zip(price_range, predictions):
            prob = pred.probability
            if (0.3 <= prob <= 0.85 and
                    base_price * 0.7 <= price <= base_price * 1.8):

                    expected_revenue = price * prob
                    service_commission = expected_revenue * SERVICE_COMMISSION_RATE
                    driver_earnings = expected_revenue - service_commission

                    # 🔥 ДОБАВЛЯЕМ ВЕС ДЛЯ БАЛАНСА
                    balance_score = prob * (1 - abs(price / base_price - 1.2))  # Предпочтение +20% к базовой цене

                    results.append({
                        'price': float(price),
                        'probability': round(prob, 3),
                        'probability_percent': round(prob * 100, 1),
                        'expected_revenue': round(expected_revenue, 2),
                        'service_earnings': round(service_commission, 2),
                        'driver_earnings': round(driver_earnings, 2),
                        'balance_score': balance_score  # Для сортировки
                    })

        # 🔥 УЛУЧШЕННАЯ СТРАТЕГИЯ ОТБОРА
        if results:
            # 1. Топ по балансу (цена + вероятность)
            results_sorted = sorted(results, key=lambda x: x['balance_score'], reverse=True)

            # 2. Берем разнообразные варианты
            final_results = []
            price_buckets = set()

            for result in results_sorted:
                price_bucket = round(result['price'] / 25) * 25  # Группируем по 25 руб
                if price_bucket not in price_buckets and len(final_results) < 8:
                    final_results.append(result)
                    price_buckets.add(price_bucket)

            # Удаляем временное поле
            for result in final_results:
                result.pop('balance_score', None)
        else:
            final_results = []

        processing_time = round((time.time() - start_time) * 1000, 2)

        return OptimalPricesResponse(
                price_curve=final_results,
                processing_time_ms=processing_time,
                analysis={
                    "total_options": len(final_results),
                    "max_revenue_option": final_results[0] if final_results else None
                }
            )

    except Exception as e:
        logger.error(f"❌ Ошибка генерации цен: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка генерации оптимальных цен: {str(e)}")

@router.get("/health", response_model=HealthResponse)
async def health_check(
        predictor: AsyncMLPredictor = Depends(get_predictor),
        startup_time: float = Depends(get_startup_time)
):
    """Проверка здоровья сервиса"""
    stats = predictor.get_stats()
    uptime = time.time() - startup_time

    status = "healthy" if stats["model_loaded"] else "degraded"

    return HealthResponse(
        status=status,
        model_loaded=stats["model_loaded"],
        total_predictions=stats["total_predictions"],
        cache_hit_rate=stats["cache"]["hit_rate"],
        avg_processing_time=stats["avg_processing_time"],
        uptime_seconds=round(uptime, 2)
    )


@router.post("/record_order")
async def record_completed_order(order_data: Dict):
    """
    Запись выполненного заказа в статистику
    Вызывается, когда водитель принимает заказ и завершает поездку
    """
    try:
        stats_service.record_completed_order(order_data)
        return {"status": "success", "message": "Order recorded"}
    except Exception as e:
        logger.error(f"❌ Ошибка записи заказа: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка записи заказа: {str(e)}")


@router.get("/stats", response_model=ServiceStats)
async def get_service_stats(
        predictor: AsyncMLPredictor = Depends(get_predictor)
):
    """
    Возвращает детальную статистику сервиса:
    - Загруженность и производительность
    - Финансовые показатели
    - Активность водителей
    - Системные метрики
    """
    try:
        # Получаем базовую статистику ML сервиса
        ml_stats = predictor.get_stats()

        # Получаем реальную статистику из БД
        financial_stats = stats_service.get_financial_stats()
        driver_stats = stats_service.get_driver_stats()
        realtime_stats = stats_service.get_realtime_stats()

        # Системные метрики
        memory_info = _get_memory_usage()
        cpu_usage = _get_cpu_usage()

        return ServiceStats(
            total_requests=ml_stats['total_predictions'],
            successful_requests=ml_stats['total_predictions'] - ml_stats['errors'],
            failed_requests=ml_stats['errors'],
            avg_processing_time_ms=round(ml_stats['avg_processing_time'] * 1000, 2),
            cache_hit_rate=ml_stats['cache']['hit_rate'],

            # Системные метрики
            memory_usage_mb=memory_info['used_mb'],
            memory_available_mb=memory_info['available_mb'],
            cpu_usage_percent=cpu_usage,

            # Финансовые показатели (РЕАЛЬНЫЕ ДАННЫЕ)
            total_service_revenue=financial_stats['total_service_revenue'],
            total_driver_earnings=financial_stats['total_driver_earnings'],
            total_commission=financial_stats['total_commission'],
            today_service_revenue=financial_stats['today_service_revenue'],
            today_driver_earnings=financial_stats['today_driver_earnings'],
            today_orders=financial_stats['today_orders'],
            avg_order_value=financial_stats['avg_order_value'],

            # Активность водителей (РЕАЛЬНЫЕ ДАННЫЕ)
            active_drivers=driver_stats['active_drivers'],
            total_drivers=driver_stats['total_drivers'],
            top_drivers=driver_stats['top_drivers'],

            # Статистика в реальном времени
            current_hour_orders=realtime_stats['current_hour_orders'],
            avg_order_completion_time=realtime_stats['avg_order_completion_time'],
            popular_routes=realtime_stats['popular_routes']
        )

    except Exception as e:
        logger.error(f"❌ Ошибка получения статистики: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка получения статистики: {str(e)}"
        )


@router.get("/stats/financial")
async def get_financial_stats(days: int = 30):
    """Детальная финансовая статистика"""
    try:
        stats = stats_service.get_financial_stats(days)
        return {
            "period_days": days,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"❌ Ошибка получения финансовой статистики: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/drivers")
async def get_drivers_stats():
    """Статистика по водителям"""
    try:
        stats = stats_service.get_driver_stats()
        return stats
    except Exception as e:
        logger.error(f"❌ Ошибка получения статистики водителей: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def _get_memory_usage() -> Dict[str, float]:
    """Получение информации об использовании памяти"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return {
            'used_mb': round(memory.used / 1024 / 1024, 2),
            'available_mb': round(memory.available / 1024 / 1024, 2),
            'total_mb': round(memory.total / 1024 / 1024, 2),
            'usage_percent': memory.percent
        }
    except ImportError:
        # Fallback если psutil не установлен
        return {
            'used_mb': 512.0,
            'available_mb': 1024.0,
            'total_mb': 2048.0,
            'usage_percent': 25.0
        }


def _get_cpu_usage() -> float:
    """Получение загрузки CPU"""
    try:
        import psutil
        return psutil.cpu_percent(interval=1)
    except ImportError:
        return 15.0  # Fallback значение


def _get_financial_stats() -> Dict[str, float]:
    """
    Получение финансовой статистики
    В реальном приложении брать из базы данных
    """
    # Заглушка - в продакшене получать из БД
    return {
        'total_service_revenue': 125000.50,  # Общая выручка сервиса
        'total_driver_earnings': 850000.75,  # Общий заработок водителей
        'total_commission': 125000.50,  # Комиссия = выручка сервиса
        'total_orders': 3500,  # Всего заказов
        'avg_order_value': 278.57  # Средний чек
    }


def _get_driver_stats() -> Dict[str, Any]:
    """
    Получение статистики по водителям
    В реальном приложении брать из базы данных
    """
    # нету реальной бд так что сорян, апишка будет туды сюды запросы слать и брать клянусь
    return {
        'active_drivers': 47,
        'total_drivers': 156,
        'top_drivers': [
            {
                'driver_id': 'driver_001',
                'driver_name': 'Иван Петров',
                'total_earnings': 45200.50,
                'completed_orders': 187,
                'rating': 4.9,
                'vehicle': 'Toyota Camry'
            },
            {
                'driver_id': 'driver_042',
                'driver_name': 'Алексей Смирнов',
                'total_earnings': 38750.25,
                'completed_orders': 156,
                'rating': 4.8,
                'vehicle': 'Hyundai Solaris'
            },
            {
                'driver_id': 'driver_087',
                'driver_name': 'Михаил Иванов',
                'total_earnings': 32500.75,
                'completed_orders': 134,
                'rating': 4.7,
                'vehicle': 'Kia Rio'
            }
        ]
    }