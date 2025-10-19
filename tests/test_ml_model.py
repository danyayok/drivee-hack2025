import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import joblib
import asyncio

from app.core.async_predictor import (
    AsyncMLPredictor, HistoricalDataService, ModelLoader,
    prepare_features_sync, predict_batch_sync, PredictionResult
)


class TestMLComponents:
    """Тесты ML компонентов"""

    @pytest.fixture
    def sample_features(self):
        return {
            'price_start_local': 200.0,
            'price_bid_local': 240.0,
            'distance_in_meters': 3500.0,
            'duration_in_seconds': 600.0,
            'pickup_in_meters': 500.0,
            'pickup_in_seconds': 120.0,
            'driver_rating': 4.8,
            'user_rating': 4.9,
            'order_timestamp': '2024-01-15T12:00:00',
            'driver_platform': 'android',
            'driver_reg_date': '2023-01-01',
            'carname': 'Toyota',
            'carmodel': 'Camry',
            'driver_id': '29368889',
            'user_id': '16458846'
        }

    @pytest.fixture
    def mock_historical_service(self):
        """Мок сервиса исторических данных"""
        service = Mock(spec=HistoricalDataService)
        service.get_driver_stats.return_value = {
            'driver_acceptance_rate': 0.75,
            'driver_total_orders': 150,
            'driver_avg_rating': 4.7,
            'driver_avg_bid_price': 220.0
        }
        service.get_user_stats.return_value = {
            'user_acceptance_rate': 0.8,
            'user_total_orders': 80,
            'user_avg_rating': 4.8,
            'user_avg_bid_price': 210.0
        }

        # ✅ ДОБАВЛЯЕМ ОТСУТСТВУЮЩИЕ МЕТОДЫ
        service._get_fallback_driver_stats = Mock(return_value={
            'driver_acceptance_rate': 0.35,
            'driver_total_orders': 0,
            'driver_avg_rating': 4.0,
            'driver_avg_bid_price': 200.0
        })
        service._get_fallback_user_stats = Mock(return_value={
            'user_acceptance_rate': 0.4,
            'user_total_orders': 0,
            'user_avg_rating': 4.5,
            'user_avg_bid_price': 200.0
        })

        return service

    def test_prepare_features_sync(self, sample_features, mock_historical_service):
        """Тест подготовки фичей"""
        with patch('app.core.async_predictor.historical_service', mock_historical_service):
            df = prepare_features_sync(sample_features)

            # Проверяем что DataFrame создан
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 1

            # Проверяем ключевые фичи
            expected_features = [
                "price_ratio", "price_diff", "distance_km", "duration_min",
                "driver_cancel_rate", "user_cancel_rate", "price_per_km",
                "order_hour", "platform", "carmodel", "carname"
            ]

            for feature in expected_features:
                assert feature in df.columns

            # Проверяем вычисления
            assert df["price_ratio"].iloc[0] == pytest.approx(240.0 / 200.0)
            assert df["distance_km"].iloc[0] == pytest.approx(3.5)
            assert df["duration_min"].iloc[0] == pytest.approx(10.0)
            assert df["driver_cancel_rate"].iloc[0] == pytest.approx(0.25)  # 1 - 0.75

    def test_prepare_features_with_missing_data(self, mock_historical_service):
        """Тест подготовки фичей с недостающими данными"""
        minimal_features = {
            'price_start_local': 200.0,
            'price_bid_local': 240.0,
            'distance_in_meters': 3500.0,
            'duration_in_seconds': 600.0,
            'driver_rating': 4.5,
            'user_rating': 4.6
            # Отсутствуют timestamp, platform и другие поля
        }

        with patch('app.core.async_predictor.historical_service', mock_historical_service):
            df = prepare_features_sync(minimal_features)

            # Проверяем что фичи созданы с fallback значениями
            assert "order_hour" in df.columns
            assert "platform" in df.columns
            assert df["platform"].iloc[0] == "android"  # default value
            assert df["order_hour"].iloc[0] == 12  # default value

    @pytest.mark.asyncio
    async def test_async_predictor_initialization(self):
        """Тест инициализации AsyncMLPredictor"""
        with patch('app.core.async_predictor.joblib.load') as mock_load:
            with patch('os.path.exists', return_value=True):
                # Мок модели
                mock_model = Mock()
                mock_load.return_value = {'model': mock_model, 'feature_names': []}

                predictor = AsyncMLPredictor("dummy_model_path.joblib")
                await predictor.initialize()

                assert predictor.model_loaded == True
                assert predictor.model == mock_model

    @pytest.mark.asyncio
    async def test_async_predictor_predict_batch(self, sample_features):
        """Тест пакетного предсказания"""
        with patch('app.core.async_predictor.predict_batch_sync') as mock_predict:
            # Мок предсказаний
            mock_predict.return_value = np.array([0.7, 0.8, 0.6])

            predictor = AsyncMLPredictor("dummy_model_path.joblib")
            predictor.model_loaded = True

            features_list = [sample_features, sample_features, sample_features]
            results = await predictor.predict_batch(features_list)

            assert len(results) == 3
            assert all(isinstance(r, PredictionResult) for r in results)
            assert results[0].probability == 0.7
            assert results[1].probability == 0.8
            assert results[2].probability == 0.6

    @pytest.mark.asyncio
    async def test_async_predictor_predict_single(self, sample_features):
        """Тест одиночного предсказания"""
        with patch('app.core.async_predictor.predict_batch_sync') as mock_predict:
            mock_predict.return_value = np.array([0.75])

            predictor = AsyncMLPredictor("dummy_model_path.joblib")
            predictor.model_loaded = True

            result = await predictor.predict_single(sample_features)

            assert isinstance(result, PredictionResult)
            assert result.probability == 0.75

    @pytest.mark.asyncio
    async def test_async_predictor_predict_batch_with_cache(self, sample_features):
        """Тест пакетного предсказания с кэшированием"""
        with patch('app.core.async_predictor.predict_batch_sync') as mock_predict:
            mock_predict.return_value = np.array([0.7, 0.8, 0.6])

            predictor = AsyncMLPredictor("dummy_model_path.joblib")
            predictor.model_loaded = True

            features_list = [sample_features, sample_features, sample_features]

            # Первый вызов - кэш промах
            results1 = await predictor.predict_batch(features_list)
            assert len(results1) == 3
            assert all(not r.cache_hit for r in results1)

            # Второй вызов - кэш попадание
            results2 = await predictor.predict_batch(features_list)
            assert len(results2) == 3
            assert all(r.cache_hit for r in results2)

    def test_predictor_stats(self):
        """Тест получения статистики predictor"""
        predictor = AsyncMLPredictor("dummy_model_path.joblib")
        predictor.model_loaded = True
        predictor.metrics = {
            'total_predictions': 100,
            'total_processing_time': 0.5,
            'errors': 2,
            'batch_sizes': [10, 20, 30]
        }

        stats = predictor.get_stats()

        assert stats["model_loaded"] == True
        assert stats["total_predictions"] == 100
        assert stats["errors"] == 2
        assert "avg_processing_time" in stats
        assert "cache" in stats

        # Проверяем вычисление средней обработки
        assert stats["avg_processing_time"] == pytest.approx(0.005)  # 0.5 / 100

    @pytest.mark.asyncio
    async def test_async_predictor_shutdown(self):
        """Тест корректного завершения работы predictor"""
        predictor = AsyncMLPredictor("dummy_model_path.joblib")

        # Мокируем пулы
        predictor.process_pool = Mock()
        predictor.thread_pool = Mock()

        await predictor.shutdown()

        # Проверяем что пулы были закрыты
        predictor.process_pool.shutdown.assert_called_once_with(wait=True)
        predictor.thread_pool.shutdown.assert_called_once_with(wait=True)


class TestHistoricalDataService:
    """Тесты сервиса исторических данных"""

    @pytest.fixture
    def sample_train_data(self, tmp_path):
        """Создание тестовых данных"""
        data = {
            'driver_id': [29368889, 29368889, 12345678, 12345678],
            'user_id': [16458846, 16458846, 87654321, 87654321],
            'is_done': ['done', 'cancel', 'done', 'done'],
            'driver_rating': [4.8, 4.7, 4.9, 4.8],
            'user_rating': [4.9, 4.8, 4.7, 4.9],
            'price_bid_local': [240.0, 220.0, 260.0, 250.0],
            'price_start_local': [200.0, 200.0, 220.0, 220.0],
            'distance_in_meters': [3500.0, 2800.0, 4000.0, 3200.0],
            'duration_in_seconds': [600.0, 480.0, 720.0, 540.0]
        }
        df = pd.DataFrame(data)

        # Сохраняем во временный файл
        file_path = tmp_path / "test_train.csv"
        df.to_csv(file_path, index=False)
        return file_path

    def test_historical_service_initialization(self, sample_train_data):
        """Тест инициализации HistoricalDataService"""
        service = HistoricalDataService(str(sample_train_data))

        # Проверяем что данные загружены
        assert len(service.df) == 4
        assert service.df['is_done'].isin([0, 1]).all()

        # Проверяем что статистика рассчитана
        assert len(service.driver_stats) > 0
        assert len(service.user_stats) > 0

    def test_get_driver_stats(self, sample_train_data):
        """Тест получения статистики водителя"""
        service = HistoricalDataService(str(sample_train_data))

        stats = service.get_driver_stats(29368889)

        assert 'driver_acceptance_rate' in stats
        assert 'driver_total_orders' in stats
        assert 'driver_avg_rating' in stats
        assert 'driver_avg_bid_price' in stats

        # Проверяем что acceptance_rate рассчитывается правильно
        # 2 заказа, 1 принят → 50% acceptance rate
        assert stats['driver_acceptance_rate'] == pytest.approx(0.5)
        assert stats['driver_total_orders'] == 2
        assert stats['driver_avg_rating'] == pytest.approx(4.75)  # (4.8 + 4.7) / 2

    def test_get_driver_stats_fallback(self, sample_train_data):
        """Тест fallback статистики для неизвестного водителя"""
        service = HistoricalDataService(str(sample_train_data))

        stats = service.get_driver_stats(99999999)  # Несуществующий ID

        assert 'driver_acceptance_rate' in stats
        assert stats['driver_acceptance_rate'] == 0.35  # Fallback value

    def test_get_user_stats(self, sample_train_data):
        """Тест получения статистики пользователя"""
        service = HistoricalDataService(str(sample_train_data))

        stats = service.get_user_stats(16458846)

        assert 'user_acceptance_rate' in stats
        assert 'user_total_orders' in stats
        assert 'user_avg_rating' in stats
        assert 'user_avg_bid_price' in stats

        # 2 заказа, 1 принят → 50% acceptance rate
        assert stats['user_acceptance_rate'] == pytest.approx(0.5)

    def test_get_combined_stats(self, sample_train_data):
        """Тест получения комбинированной статистики"""
        service = HistoricalDataService(str(sample_train_data))

        stats = service.get_combined_stats(29368889, 16458846)

        assert 'user_acceptance_rate' in stats
        assert 'driver_acceptance_rate' in stats
        assert 'user_avg_bid_price' in stats
        assert 'driver_avg_bid_price' in stats
        assert 'avg_response_delay' in stats