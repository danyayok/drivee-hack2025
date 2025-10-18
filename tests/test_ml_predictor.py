import pytest
import asyncio
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from app.core.async_predictor import AsyncMLPredictor, PredictionResult


class TestAsyncMLPredictor:
    """Тесты для AsyncMLPredictor"""

    @pytest.mark.asyncio
    async def test_predict_batch_success(self, mock_predictor):
        """Тест успешного пакетного предсказания"""
        test_features = [
            {
                "distance_in_meters": 1000,
                "duration_in_seconds": 300,
                "price_bid_local": 200,
                "price_start_local": 180
            },
            {
                "distance_in_meters": 2000,
                "duration_in_seconds": 600,
                "price_bid_local": 300,
                "price_start_local": 250
            }
        ]

        results = await mock_predictor.predict_batch(test_features)

        assert len(results) == len(test_features)
        for result in results:
            assert isinstance(result, AsyncMock)
            assert 0.05 <= result.probability <= 0.95
            assert result.processing_time == 0.01
            assert result.cache_hit is False
            assert result.error is None

    @pytest.mark.asyncio
    async def test_predict_single(self, mock_predictor):
        """Тест предсказания для одного примера"""
        test_features = {
            "distance_in_meters": 1500,
            "duration_in_seconds": 400,
            "price_bid_local": 250,
            "price_start_local": 200
        }

        result = await mock_predictor.predict_single(test_features)

        assert isinstance(result, AsyncMock)
        assert 0.05 <= result.probability <= 0.95

    @pytest.mark.asyncio
    async def test_predict_batch_empty(self, mock_predictor):
        """Тест пустого батча"""
        results = await mock_predictor.predict_batch([])
        assert results == []

    def test_cache_key_generation(self, mock_predictor):
        """Тест генерации ключей кэша"""
        features_list = [
            {"price_bid_local": 200.0, "distance_in_meters": 1000},
            {"price_bid_local": 250.0, "distance_in_meters": 1500}
        ]

        cache_key = mock_predictor._generate_cache_key(features_list)

        assert isinstance(cache_key, str)
        assert len(cache_key) == 32  # MD5 hash length

    def test_cache_consistency(self, mock_predictor):
        """Тест консистентности кэша"""
        features1 = [{"price_bid_local": 200.0, "distance_in_meters": 1000}]
        features2 = [{"distance_in_meters": 1000, "price_bid_local": 200.0}]  # Тот же порядок полей

        key1 = mock_predictor._generate_cache_key(features1)
        key2 = mock_predictor._generate_cache_key(features2)

        assert key1 == key2  # Должны быть одинаковыми

    def test_get_stats(self, mock_predictor):
        """Тест получения статистики"""
        stats = mock_predictor.get_stats()

        assert stats["model_loaded"] is True
        assert stats["total_predictions"] == 1000
        assert stats["avg_processing_time"] == 0.1
        assert "cache" in stats
        assert stats["cache"]["hit_rate"] == 0.65


class TestPredictionResult:
    """Тесты для PredictionResult"""

    def test_prediction_result_creation(self):
        """Тест создания PredictionResult"""
        result = PredictionResult(
            probability=0.75,
            processing_time=0.05,
            cache_hit=False,
            error=None
        )

        assert result.probability == 0.75
        assert result.processing_time == 0.05
        assert result.cache_hit is False
        assert result.error is None

    def test_prediction_result_with_error(self):
        """Тест PredictionResult с ошибкой"""
        result = PredictionResult(
            probability=0.5,
            processing_time=0.1,
            cache_hit=False,
            error="Model timeout"
        )

        assert result.error == "Model timeout"