import asyncio
import time
import numpy as np
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import uuid

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


@router.post("/get_optimal_prices", response_model=OptimalPricesResponse)
async def get_optimal_prices(
        request: OrderRequest,
        predictor: AsyncMLPredictor = Depends(get_predictor)
):
    try:
        start_time = time.time()
        enhanced_request = request.dict()
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

        # Системные метрики
        memory_info = _get_memory_usage()
        cpu_usage = _get_cpu_usage()

        # Финансовая статистика (в реальном приложении брать из БД)
        financial_stats = _get_financial_stats()

        # Статистика по водителям (в реальном приложении брать из БД)
        driver_stats = _get_driver_stats()

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

            # Финансовые показатели
            total_service_revenue=financial_stats['total_service_revenue'],
            total_driver_earnings=financial_stats['total_driver_earnings'],
            total_commission=financial_stats['total_commission'],

            # Активность водителей
            active_drivers=driver_stats['active_drivers'],
            top_drivers=driver_stats['top_drivers']
        )

    except Exception as e:
        logger.error(f"❌ Ошибка получения статистики: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка получения статистики: {str(e)}"
        )


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
    # Заглушка - в продакшене получать из БД
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