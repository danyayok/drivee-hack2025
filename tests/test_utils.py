import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from app.utils.logger import get_logger
from app.api.endpoints import (
    _generate_price_range, _select_strategic_prices,
    _merge_price_suggestions, _analyze_results
)


class TestPriceUtils:
    """Тесты утилит для работы с ценами"""

    @pytest.mark.asyncio
    async def test_generate_price_range(self):
        """Тест генерации диапазона цен"""
        start_price = 200.0
        price_range = await _generate_price_range(start_price)

        assert len(price_range) == 50
        assert min(price_range) >= start_price * 0.7  # -30%
        assert max(price_range) <= start_price * 2.0  # +100%
        assert start_price * 0.9 in price_range  # Проверяем типичные значения
        assert start_price * 1.1 in price_range
        assert start_price * 1.5 in price_range

    def test_select_strategic_prices(self):
        """Тест выбора стратегических цен"""
        results = [
            {"price": 150.0, "probability": 0.9, "expected_revenue": 135.0, "service_commission": 17.28,
             "driver_earnings": 117.72},
            {"price": 180.0, "probability": 0.8, "expected_revenue": 144.0, "service_commission": 18.43,
             "driver_earnings": 125.57},
            {"price": 200.0, "probability": 0.7, "expected_revenue": 140.0, "service_commission": 17.92,
             "driver_earnings": 122.08},
            {"price": 250.0, "probability": 0.5, "expected_revenue": 125.0, "service_commission": 16.0,
             "driver_earnings": 109.0},
            {"price": 300.0, "probability": 0.3, "expected_revenue": 90.0, "service_commission": 11.52,
             "driver_earnings": 78.48},
            {"price": 350.0, "probability": 0.45, "expected_revenue": 157.5, "service_commission": 20.16,
             "driver_earnings": 137.34}
        ]

        strategic = _select_strategic_prices(results)

        # Должны быть выбраны: максимальная вероятность, ближайшая к 70%, агрессивная цена
        assert len(strategic) >= 2
        assert any(p["price"] == 150.0 for p in strategic)  # Макс вероятность
        assert any(p["price"] == 200.0 for p in strategic)  # Ближайшая к 70%

    def test_merge_price_suggestions(self):
        """Тест объединения ценовых предложений"""
        top_prices = [
            {"price": 200.0, "probability": 0.8, "expected_revenue": 160.0, "service_commission": 20.48,
             "driver_earnings": 139.52},
            {"price": 220.0, "probability": 0.75, "expected_revenue": 165.0, "service_commission": 21.12,
             "driver_earnings": 143.88},
            {"price": 180.0, "probability": 0.85, "expected_revenue": 153.0, "service_commission": 19.58,
             "driver_earnings": 133.42}
        ]

        strategic_prices = [
            {"price": 190.0, "probability": 0.82, "expected_revenue": 155.8, "service_commission": 19.94,
             "driver_earnings": 135.86},
            {"price": 210.0, "probability": 0.78, "expected_revenue": 163.8, "service_commission": 20.97,
             "driver_earnings": 142.83}
        ]

        merged = _merge_price_suggestions(top_prices, strategic_prices)

        # Проверяем что нет дубликатов
        prices = [p["price"] for p in merged]
        assert len(prices) == len(set(prices))

        # Проверяем что merged отсортирован по цене
        assert merged == sorted(merged, key=lambda x: x["price"])

    def test_analyze_results(self):
        """Тест анализа результатов"""
        results = [
            {"price": 150.0, "probability": 0.9, "expected_revenue": 135.0, "service_commission": 17.28,
             "driver_earnings": 117.72},
            {"price": 200.0, "probability": 0.7, "expected_revenue": 140.0, "service_commission": 17.92,
             "driver_earnings": 122.08},
            {"price": 250.0, "probability": 0.5, "expected_revenue": 125.0, "service_commission": 16.0,
             "driver_earnings": 109.0}
        ]

        start_price = 180.0
        analysis = _analyze_results(results, start_price)

        assert "max_revenue_price" in analysis
        assert "max_probability_price" in analysis
        assert "max_driver_earnings_price" in analysis
        assert "recommendations" in analysis
        assert "total_options" in analysis

        assert analysis["max_revenue_price"] == 200.0
        assert analysis["max_probability_price"] == 150.0
        assert analysis["total_options"] == 3


class TestLogger:
    """Тесты для логгера"""

    def test_logger_creation(self):
        """Тест создания логгера"""
        logger = get_logger("test_module")

        assert logger.name == "test_module"
        assert logger.level == 20  # INFO level

    def test_logger_different_levels(self):
        """Тест логгера с разными уровнями"""
        debug_logger = get_logger("debug_module", level=10)  # DEBUG
        error_logger = get_logger("error_module", level=40)  # ERROR

        assert debug_logger.level == 10
        assert error_logger.level == 40