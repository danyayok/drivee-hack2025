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

        logger.info(f"🎯 Получен запрос. Водитель: {request.driver_id}, Пользователь: {request.user_id}")

        # 🔥 ДОБАВЛЯЕМ ИСТОРИЧЕСКИЕ ДАННЫЕ В ЗАПРОС
        enhanced_request = request.dict()

        # Генерируем диапазон цен
        price_range = await _generate_price_range(request.price_start_local)

        # Подготавливаем тестовые сценарии
        test_scenarios = []
        for test_price in price_range:
            scenario = enhanced_request.copy()
            scenario['price_bid_local'] = round(test_price, 2)
            test_scenarios.append(scenario)

        # 🔥 ПРЕДСКАЗАНИЕ С ИСТОРИЧЕСКИМИ ДАННЫМИ
        logger.info(f"🔍 Тестируем {len(test_scenarios)} цен с историческими данными...")
        prediction_results = await predictor.predict_batch(test_scenarios)

        # Анализируем результаты с финансовыми расчетами
        results = []
        for test_price, pred_result in zip(price_range, prediction_results):
            expected_revenue = test_price * pred_result.probability
            service_commission = expected_revenue * SERVICE_COMMISSION_RATE
            driver_earnings = expected_revenue - service_commission

            results.append({
                'price': round(test_price, 2),
                'probability': round(pred_result.probability, 3),
                'expected_revenue': round(expected_revenue, 2),
                'service_commission': round(service_commission, 2),
                'driver_earnings': round(driver_earnings, 2)
            })

        # Сортируем по ожидаемому доходу и берем топ-5
        top_prices = sorted(results, key=lambda x: x['expected_revenue'], reverse=True)[:5]

        # Добавляем стратегические варианты
        strategic_prices = _select_strategic_prices(results)

        # Объединяем и убираем дубликаты
        final_prices = _merge_price_suggestions(top_prices, strategic_prices)

        # Дополнительный анализ с финансовыми метриками
        analysis = _analyze_results(final_prices, request.price_start_local)

        processing_time = time.time() - start_time

        logger.info(f"✅ Сгенерировано {len(final_prices)} ценовых предложений. "
                    f"Время: {processing_time:.3f}s")

        return OptimalPricesResponse(
            price_curve=final_prices,
            processing_time_ms=round(processing_time * 1000, 2),
            analysis=analysis
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Ошибка генерации цен: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка генерации оптимальных цен: {str(e)}"
        )


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

async def _generate_price_range(start_price: float) -> np.ndarray:
    """Генерация диапазона цен для тестирования"""
    # Тестируем цены от -30% до +100% от базовой
    min_price = start_price * 0.7
    max_price = start_price * 2.0
    steps = 50  # Количество точек для тестирования

    # ВАЖНО: используем нормальные цены, а не копейки!
    price_range = np.linspace(min_price, max_price, steps)

    # Округляем до рублей
    price_range = np.round(price_range)

    # Убираем дубликаты после округления
    price_range = np.unique(price_range)

    # Фильтруем слишком низкие цены (минимально 50 руб)
    price_range = price_range[price_range >= 50]

    logger.info(f"💰 Генерируем цены от {min_price:.0f} до {max_price:.0f} руб, шагов: {len(price_range)}")

    return price_range


def _select_strategic_prices(results: List[Dict]) -> List[Dict]:
    """Выбор стратегически важных цен"""
    strategic = []

    if not results:
        return strategic

    # 1. Цена с максимальной вероятностью принятия
    max_prob = max(results, key=lambda x: x['probability'])
    strategic.append(max_prob)

    # 2. Баланс цены и вероятности (ближайшая к 70% вероятности)
    balanced = min(results, key=lambda x: abs(x['probability'] - 0.7))
    strategic.append(balanced)

    # 3. Агрессивная цена (высокая цена, приемлемая вероятность)
    high_price_candidates = [r for r in results if r['probability'] > 0.4]
    if high_price_candidates:
        aggressive = max(high_price_candidates, key=lambda x: x['price'])
        strategic.append(aggressive)

    return strategic


def _merge_price_suggestions(top_prices: List[Dict], strategic_prices: List[Dict]) -> List[Dict]:
    """Объединение и дедупликация ценовых предложений"""
    merged = []
    seen_prices = set()

    # Добавляем топ цены по доходу
    for price in top_prices:
        price_key = price['price']
        if price_key not in seen_prices:
            merged.append(price)
            seen_prices.add(price_key)

    # Добавляем стратегические цены
    for price in strategic_prices:
        price_key = price['price']
        if price_key not in seen_prices and len(merged) < 8:  # Максимум 8 предложений
            merged.append(price)
            seen_prices.add(price_key)

    # Сортируем по цене для удобства восприятия
    return sorted(merged, key=lambda x: x['price'])


def _analyze_results(results: List[Dict], start_price: float) -> Dict[str, Any]:
    """Анализ результатов с финансовыми метриками"""
    if not results:
        return {"error": "Нет данных для анализа"}
    max_revenue_price = max(results, key=lambda x: x['expected_revenue'])
    max_probability_price = max(results, key=lambda x: x['probability'])
    max_driver_earnings = max(results, key=lambda x: x['driver_earnings'])

    # Финансовый анализ
    total_potential_revenue = sum(p['expected_revenue'] for p in results)
    avg_service_commission = np.mean([p['service_commission'] for p in results])
    avg_driver_earnings = np.mean([p['driver_earnings'] for p in results])

    # Рекомендации
    recommendations = []

    if max_revenue_price['probability'] > 0.7:
        rec = f"Рекомендуем {max_revenue_price['price']}₽: макс. доход {max_revenue_price['expected_revenue']}₽"
        rec += f" (сервис: {max_revenue_price['service_commission']}₽, водитель: {max_revenue_price['driver_earnings']}₽)"
        recommendations.append(rec)
    else:
        balanced = min(results, key=lambda x: abs(x['probability'] - 0.7))
        rec = f"Баланс: {balanced['price']}₽ с вероятностью {balanced['probability'] * 100}%"
        rec += f" (сервис: {balanced['service_commission']}₽, водитель: {balanced['driver_earnings']}₽)"
        recommendations.append(rec)

    # Добавляем рекомендацию по заработку водителя
    if max_driver_earnings['probability'] > 0.5:
        rec = f"Для водителя: {max_driver_earnings['price']}₽ → заработок {max_driver_earnings['driver_earnings']}₽"
        recommendations.append(rec)

    return {
        # Ключевые цены
        "max_revenue_price": max_revenue_price['price'],
        "max_revenue_probability": max_revenue_price['probability'],
        "max_probability_price": max_probability_price['price'],
        "max_probability": max_probability_price['probability'],
        "max_driver_earnings_price": max_driver_earnings['price'],
        "max_driver_earnings": max_driver_earnings['driver_earnings'],

        # Финансовые метрики
        "total_potential_revenue": round(total_potential_revenue, 2),
        "avg_service_commission": round(avg_service_commission, 2),
        "avg_driver_earnings": round(avg_driver_earnings, 2),
        "service_commission_rate": f"{SERVICE_COMMISSION_RATE * 100}%",

        "recommendations": recommendations,
        "total_options": len(results)
    }


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