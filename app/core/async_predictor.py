import asyncio
import time
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple
import joblib
import hashlib
import json
from dataclasses import dataclass
import os
import signal
from functools import lru_cache
import threading
from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    """Результат предсказания"""
    probability: float
    processing_time: float
    cache_hit: bool = False
    error: Optional[str] = None


def init_worker():
    """
    Инициализация воркера процесса - игнорируем SIGINT в дочерних процессах
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class HistoricalDataService:
    """Сервис для работы с историческими данными"""

    def __init__(self, data_path: str = "train.csv"):
        self.data_path = data_path
        self.df = None
        self.driver_stats = {}
        self.user_stats = {}
        self._load_data()

    def _load_data(self):
        """Загружаем и предобрабатываем данные"""
        try:
            self.df = pd.read_csv(self.data_path)

            # Преобразуем target
            self.df['is_done'] = self.df['is_done'].astype(str).str.lower().map({"done": 1, "cancel": 0})
            self.df = self.df[self.df['is_done'].isin([0, 1])]

            # Рассчитываем статистики
            self._calculate_stats()
            logger.info(f"✅ Загружено {len(self.df)} исторических записей")

        except Exception as e:
            logger.error(f"❌ Ошибка загрузки исторических данных: {e}")
            self.df = pd.DataFrame()

    def _calculate_stats(self):
        """Рассчитываем acceptance rates и другую статистику"""
        # Driver statistics
        driver_stats = self.df.groupby('driver_id').agg({
            'is_done': ['mean', 'count'],
            'driver_rating': 'mean',
            'price_bid_local': 'mean'
        }).round(3)
        driver_stats.columns = ['acceptance_rate', 'total_orders', 'avg_rating', 'avg_bid_price']
        self.driver_stats = driver_stats.to_dict('index')

        # User statistics
        user_stats = self.df.groupby('user_id').agg({
            'is_done': ['mean', 'count'],
            'user_rating': 'mean'
        }).round(3)
        user_stats.columns = ['acceptance_rate', 'total_orders', 'avg_rating']
        self.user_stats = user_stats.to_dict('index')

        logger.info(f"📊 Рассчитано stats для {len(self.driver_stats)} водителей и {len(self.user_stats)} пользователей")

    def get_driver_stats(self, driver_id) -> Dict:
        """Получаем статистику водителя"""
        if not driver_id:
            return {
                'driver_acceptance_rate': 0.35,
                'driver_total_orders': 0,
                'driver_avg_rating': 4.0,
                'driver_avg_bid_price': 200.0
            }

        stats = self.driver_stats.get(str(driver_id), {})
        return {
            'driver_acceptance_rate': stats.get('acceptance_rate', 0.35),
            'driver_total_orders': stats.get('total_orders', 0),
            'driver_avg_rating': stats.get('avg_rating', 4.0),
            'driver_avg_bid_price': stats.get('avg_bid_price', 200.0)
        }

    def get_user_stats(self, user_id) -> Dict:
        """Получаем статистику пользователя"""
        if not user_id:
            return {
                'user_acceptance_rate': 0.4,
                'user_total_orders': 0,
                'user_avg_rating': 4.5
            }

        stats = self.user_stats.get(str(user_id), {})
        return {
            'user_acceptance_rate': stats.get('acceptance_rate', 0.4),
            'user_total_orders': stats.get('total_orders', 0),
            'user_avg_rating': stats.get('avg_rating', 4.5)
        }

    def get_combined_stats(self, driver_id, user_id) -> Dict:
        """Комбинированная статистика для пары водитель-пользователь"""
        driver_stats = self.get_driver_stats(driver_id)
        user_stats = self.get_user_stats(user_id)

        return {**driver_stats, **user_stats}


# Глобальный экземпляр
historical_service = HistoricalDataService()


class ModelLoader:
    """Загрузчик моделей с кэшированием в процессе"""

    _models = {}  # Кэш моделей по путям

    @classmethod
    def load_model(cls, model_path: str) -> Tuple[any, List[str]]:
        """
        Загрузка модели с кэшированием в рамках процесса
        """
        if model_path in cls._models:
            return cls._models[model_path]

        try:
            logger.info(f"🔄 Загружаем модель в процессе {os.getpid()}")
            model_data = joblib.load(model_path)
            model = model_data['model']
            feature_names = model_data['feature_names']

            # Валидация что модель имеет метод predict_proba
            if not hasattr(model, 'predict_proba'):
                raise ValueError("Модель должна иметь метод predict_proba")

            cls._models[model_path] = (model, feature_names)

            logger.info(f"✅ Модель загружена в процессе {os.getpid()}, "
                        f"фичей: {len(feature_names)}")

            return model, feature_names

        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели в процессе {os.getpid()}: {e}")
            raise


def prepare_features_sync(features: Dict) -> pd.DataFrame:
    """
    Подготовка фичей ДОЛЖНА БЫТЬ ИДЕНТИЧНОЙ обучению новой модели!
    Теперь с реальными историческими данными!
    """
    df = pd.DataFrame([features])

    # 🔥 ПОЛУЧАЕМ ИСТОРИЧЕСКИЕ ДАННЫЕ
    driver_id = features.get('driver_id')
    user_id = features.get('user_id')

    historical_data = historical_service.get_combined_stats(driver_id, user_id)

    # === СТАТИЧЕСКИЕ ФИЧИ ===

    # Временные фичи
    if 'order_timestamp' in features:
        try:
            order_time = pd.to_datetime(features['order_timestamp'])
            df['order_hour'] = order_time.hour
            df['order_dayofweek'] = order_time.dayofweek
            df['is_peak_hour'] = df['order_hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
            df['is_weekend'] = (df['order_dayofweek'] >= 5).astype(int)
            df['is_night'] = ((df['order_hour'] >= 23) | (df['order_hour'] <= 5)).astype(int)
        except Exception:
            # Fallback значения
            df['order_hour'] = 12
            df['order_dayofweek'] = 0
            df['is_peak_hour'] = 0
            df['is_weekend'] = 0
            df['is_night'] = 0

    # Гео фичи
    if 'distance_in_meters' in features:
        df['distance_km'] = features['distance_in_meters'] / 1000
        df['log_distance_in_meters'] = np.log1p(features['distance_in_meters'])

    if 'duration_in_seconds' in features:
        df['duration_minutes'] = features['duration_in_seconds'] / 60
        df['log_duration_in_seconds'] = np.log1p(features['duration_in_seconds'])

    if 'pickup_in_meters' in features:
        df['pickup_km'] = features['pickup_in_meters'] / 1000
        df['pickup_minutes'] = features['pickup_in_seconds'] / 60

    # Скорости
    if 'distance_km' in df.columns and 'duration_minutes' in df.columns:
        df['trip_speed_kmh'] = df['distance_km'] / (df['duration_minutes'] / 60 + 1e-6)

    if 'pickup_km' in df.columns and 'pickup_minutes' in df.columns:
        df['pickup_speed_kmh'] = df['pickup_km'] / (df['pickup_minutes'] / 60 + 1e-6)

    # Опыт водителя
    if 'driver_reg_date' in features:
        try:
            reg_date = pd.to_datetime(features['driver_reg_date'])
            order_time = pd.to_datetime(features.get('order_timestamp', pd.Timestamp.now()))
            df['driver_experience_days'] = (order_time - reg_date).days
            df['log_driver_experience_days'] = np.log1p(df['driver_experience_days'])
        except Exception:
            df['driver_experience_days'] = 365
            df['log_driver_experience_days'] = np.log1p(365)

    # === ИСТОРИЧЕСКИЕ ФИЧИ ===
    # 🔥 ИСПОЛЬЗУЕМ РЕАЛЬНЫЕ ДАННЫЕ ВМЕСТО ФИКТИВНЫХ
    df['driver_acceptance_rate'] = historical_data.get('driver_acceptance_rate', 0.35)
    df['user_acceptance_rate'] = historical_data.get('user_acceptance_rate', 0.4)
    df['driver_total_orders'] = historical_data.get('driver_total_orders', 0)
    df['user_total_orders'] = historical_data.get('user_total_orders', 0)

    # === ДИНАМИЧЕСКИЕ ФИЧИ (зависят от price_bid_local) ===
    if 'price_bid_local' in features and 'price_start_local' in features:
        # Эти фичи КРИТИЧЕСКИ важны для новой модели!
        df['price_diff'] = features['price_bid_local'] - features['price_start_local']
        df['price_ratio'] = features['price_bid_local'] / (features['price_start_local'] + 1e-6)
        df['price_increase_pct'] = (df['price_diff'] / (features['price_start_local'] + 1e-6)) * 100

        df['price_per_km'] = features['price_bid_local'] / (df['distance_km'] + 1e-6)
        df['price_per_min'] = features['price_bid_local'] / (df['duration_minutes'] + 1e-6)

        # Взаимодействия с историческими данными
        if 'driver_rating' in features:
            df['rating_price_interaction'] = features['driver_rating'] * df['price_ratio']

        if 'driver_experience_days' in df.columns:
            df['experience_price_ratio'] = df['driver_experience_days'] * df['price_ratio']

        # 🔥 НОВЫЕ ВЗАИМОДЕЙСТВИЯ С ИСТОРИЧЕСКИМИ ДАННЫМИ
        df['historical_rate_price_ratio'] = df['driver_acceptance_rate'] * df['price_ratio']
        df['experience_price_interaction'] = df['driver_total_orders'] * df['price_ratio']

    # === КАТЕГОРИАЛЬНЫЕ ФИЧИ ===
    # ВАЖНО: преобразуем в строки как при обучении!
    if 'driver_platform' in features:
        df['platform'] = str(features['driver_platform'])
    else:
        df['platform'] = 'Android'

    if 'carmodel' in features:
        df['carmodel'] = str(features['carmodel'])
    else:
        df['carmodel'] = 'unknown'

    if 'carname' in features:
        df['carname'] = str(features['carname'])
    else:
        df['carname'] = 'unknown'

    return df


def predict_batch_sync(model_path: str, features_list: List[Dict]) -> np.ndarray:
    """
    Синхронная функция предсказания для ProcessPoolExecutor
    Теперь с историческими данными!
    """
    start_time = time.time()

    try:
        # Загружаем модель (кэшируется в процессе)
        model, feature_names = ModelLoader.load_model(model_path)

        if len(features_list) == 0:
            return np.array([])

        # Подготавливаем все фичи
        prepared_features = []
        for features in features_list:
            try:
                # 🔥 ПОДГОТОВКА С ИСТОРИЧЕСКИМИ ДАННЫМИ
                prepared = prepare_features_sync(features)

                # Убедимся что все фичи присутствуют в правильном порядке
                for feature in feature_names:
                    if feature not in prepared.columns:
                        prepared[feature] = 0.0  # Значение по умолчанию

                # Упорядочиваем фичи как при обучении
                prepared = prepared[feature_names]
                prepared_features.append(prepared)

            except Exception as e:
                logger.warning(f"⚠️ Ошибка подготовки фичей: {e}")
                # Создаем fallback фичи
                fallback_df = pd.DataFrame([{f: 0.0 for f in feature_names}])
                prepared_features.append(fallback_df)

        # Объединяем в один DataFrame
        combined_df = pd.concat(prepared_features, ignore_index=True)

        # Финальная проверка порядка фичей
        if list(combined_df.columns) != feature_names:
            logger.error(f"❌ Критическая ошибка: несоответствие фичей!")
            combined_df = combined_df.reindex(columns=feature_names, fill_value=0.0)

        # Предсказание
        probabilities = model.predict_proba(combined_df)[:, 1]

        # 🔥 ВАЛИДАЦИЯ ВЕРОЯТНОСТЕЙ
        probabilities = np.array([max(0.05, min(0.95, float(prob))) for prob in probabilities])

        processing_time = time.time() - start_time
        logger.debug(f"🧠 Предсказание {len(features_list)} samples за {processing_time:.3f}s "
                     f"в процессе {os.getpid()}")

        return probabilities

    except Exception as e:
        logger.error(f"❌ Ошибка предсказания в процессе {os.getpid()}: {e}")
        # Возвращаем fallback значения
        return np.full(len(features_list), 0.5)


class CacheWithTTL:
    """Потокобезопасный кэш с TTL и периодической очисткой"""

    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        self._cache = {}
        self._lock = threading.RLock()  # Reentrant lock для вложенных вызовов
        self._hits = 0
        self._misses = 0
        self._last_cleanup = time.time()

    def get(self, key: str) -> Optional[any]:
        """Получение из кэша с проверкой TTL"""
        with self._lock:
            # Периодическая очистка (раз в минуту)
            if time.time() - self._last_cleanup > 60:
                self._cleanup_expired()

            if key in self._cache:
                entry = self._cache[key]
                if time.time() - entry['timestamp'] < self.ttl:
                    self._hits += 1
                    return entry['value']
                else:
                    # Удаляем просроченную запись
                    del self._cache[key]

            self._misses += 1
            return None

    def set(self, key: str, value: any):
        """Сохранение в кэш с очисткой по LRU"""
        with self._lock:
            # Очистка если достигли лимита
            if len(self._cache) >= self.max_size:
                # Удаляем самую старую запись
                oldest_key = min(self._cache.keys(),
                                 key=lambda k: self._cache[k]['timestamp'])
                del self._cache[oldest_key]

            self._cache[key] = {
                'value': value,
                'timestamp': time.time()
            }

    def _cleanup_expired(self):
        """Очистка просроченных записей"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time - entry['timestamp'] > self.ttl
        ]
        for key in expired_keys:
            del self._cache[key]

        self._last_cleanup = current_time
        if expired_keys:
            logger.debug(f"🧹 Очищено {len(expired_keys)} просроченных записей кэша")

    def stats(self) -> Dict:
        """Статистика кэша"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0

            return {
                'size': len(self._cache),
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': round(hit_rate, 3),
                'max_size': self.max_size,
                'ttl': self.ttl
            }


class AsyncMLPredictor:
    """
    Асинхронный ML сервис для предсказаний с историческими данными
    """

    def __init__(self, model_path: str = settings.MODEL_PATH):
        self.model_path = os.path.abspath(model_path)
        self.model_loaded = False

        # 🔥 ИСТОРИЧЕСКИЙ СЕРВИС
        self.historical_service = historical_service

        # 🔧 ProcessPool для CPU-bound операций
        self.process_pool = ProcessPoolExecutor(
            max_workers=settings.PROCESS_POOL_WORKERS,
            initializer=init_worker,  # Игнорируем SIGINT в дочерних процессах
            mp_context=None
        )

        # ThreadPool для I/O операций
        self.thread_pool = ThreadPoolExecutor(
            max_workers=settings.THREAD_POOL_WORKERS
        )

        # 🗂 Улучшенный кэш с TTL
        self.cache = CacheWithTTL(
            max_size=settings.MODEL_CACHE_SIZE,
            ttl=settings.CACHE_TTL
        )

        # 📊 Метрики
        self.metrics = {
            'total_predictions': 0,
            'total_processing_time': 0.0,
            'errors': 0,
            'batch_sizes': []
        }

        self.start_time = time.time()
        logger.info(f"🚀 AsyncMLPredictor инициализирован с историческими данными")

    async def initialize(self):
        """Проверка доступности модели"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Файл модели не найден: {self.model_path}")

            # Тестовая загрузка в основном процессе
            test_data = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                joblib.load,
                self.model_path
            )

            # Проверка структуры модели
            required_keys = ['model', 'feature_names']
            for key in required_keys:
                if key not in test_data:
                    raise ValueError(f"Модель не содержит обязательный ключ: {key}")

            expected_features = test_data['feature_names']
            logger.info(f"✅ Модель доступна. Ожидаемые фичи ({len(expected_features)}): {expected_features}")

            self.model_loaded = True

        except Exception as e:
            logger.error(f"❌ Ошибка инициализации модели: {e}")
            self.model_loaded = False
            raise

    async def predict_price_acceptance(self, base_features: Dict, test_prices: List[float]) -> List[PredictionResult]:
        """
        Предсказание вероятности принятия для нескольких цен

        Args:
            base_features: Базовые фичи заказа (без price_bid_local)
            test_prices: Список цен для тестирования

        Returns:
            Список PredictionResult для каждой цены
        """
        if not self.model_loaded:
            return [PredictionResult(0.5, 0.0, error="Service not ready")
                    for _ in test_prices]

        start_time = time.time()

        try:
            # Подготавливаем фичи для каждой цены
            test_scenarios = []
            for price in test_prices:
                scenario = base_features.copy()
                scenario['price_bid_local'] = price
                test_scenarios.append(scenario)

            # Предсказание
            results = await self.predict_batch(test_scenarios)

            processing_time = time.time() - start_time
            logger.info(f"🔍 Протестировано {len(test_prices)} цен за {processing_time:.3f}s")

            return results

        except Exception as e:
            logger.error(f"❌ Ошибка предсказания цен: {e}")
            return [PredictionResult(0.5, processing_time=0.1, error=str(e))
                    for _ in test_prices]

    async def predict_batch(self, features_list: List[Dict]) -> List[PredictionResult]:
        """
        Оптимизированное пакетное предсказание с историческими данными
        """
        if not features_list:
            return []

        if not self.model_loaded:
            return [PredictionResult(0.5, 0.0, error="Service not ready")
                    for _ in features_list]

        start_time = time.time()

        # Генерация ключа кэша
        cache_key = self._generate_cache_key(features_list)

        # Проверка кэша
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            processing_time = time.time() - start_time
            return [
                PredictionResult(
                    probability=prob,
                    processing_time=processing_time / len(features_list),
                    cache_hit=True
                )
                for prob in cached_result
            ]

        try:
            # 🔥 Асинхронный вызов в ProcessPool с историческими данными
            loop = asyncio.get_event_loop()
            probabilities = await asyncio.wait_for(
                loop.run_in_executor(
                    self.process_pool,
                    predict_batch_sync,
                    self.model_path,
                    features_list
                ),
                timeout=settings.MODEL_TIMEOUT
            )

            processing_time = time.time() - start_time

            # Сохраняем в кэш
            self.cache.set(cache_key, probabilities.tolist())

            # Обновляем метрики
            self._update_metrics(processing_time, len(features_list))

            logger.debug(f"🧠 Batch: {len(features_list)} samples, {processing_time:.3f}s, "
                         f"avg: {processing_time / len(features_list) * 1000:.1f}ms/sample")

            return [
                PredictionResult(
                    probability=float(prob),
                    processing_time=processing_time / len(features_list),
                    cache_hit=False
                )
                for prob in probabilities
            ]

        except asyncio.TimeoutError:
            self.metrics['errors'] += 1
            logger.error(f"⏰ Таймаут предсказания для {len(features_list)} samples")
            return [
                PredictionResult(
                    probability=0.5,
                    processing_time=time.time() - start_time,
                    error="Prediction timeout"
                )
                for _ in features_list
            ]
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"❌ Ошибка пакетного предсказания: {e}")

            # Fallback
            return [
                PredictionResult(
                    probability=0.5,
                    processing_time=time.time() - start_time,
                    error=str(e)
                )
                for _ in features_list
            ]

    async def predict_single(self, features: Dict) -> PredictionResult:
        """Предсказание для одного примера"""
        results = await self.predict_batch([features])
        return results[0] if results else PredictionResult(0.5, 0.0, error="No results")

    def _generate_cache_key(self, features_list: List[Dict]) -> str:
        """Генерация консистентного ключа кэша"""
        try:
            # Нормализация фичей для консистентности
            normalized_list = []
            for features in features_list:
                normalized = {}

                # Сортируем ключи и нормализуем значения
                for key in sorted(features.keys()):
                    value = features[key]

                    # Нормализация чисел
                    if isinstance(value, (int, float)):
                        # Для цен используем 2 знака, для остального 6
                        if 'price' in key.lower():
                            normalized[key] = round(float(value), 2)
                        else:
                            normalized[key] = round(float(value), 6)
                    else:
                        normalized[key] = str(value)

                normalized_list.append(normalized)

            # Сериализация
            features_str = json.dumps(normalized_list, sort_keys=True, ensure_ascii=False)
            return hashlib.md5(features_str.encode()).hexdigest()

        except Exception as e:
            logger.warning(f"⚠️ Ошибка генерации ключа кэша: {e}")
            # Fallback - хэш от количества элементов
            return hashlib.md5(str(len(features_list)).encode()).hexdigest()

    def _update_metrics(self, processing_time: float, batch_size: int):
        """Обновление метрик производительности"""
        self.metrics['total_predictions'] += batch_size
        self.metrics['total_processing_time'] += processing_time
        self.metrics['batch_sizes'].append(batch_size)

        # Ограничиваем размер истории
        if len(self.metrics['batch_sizes']) > 1000:
            self.metrics['batch_sizes'] = self.metrics['batch_sizes'][-1000:]

    def get_stats(self) -> Dict:
        """Детальная статистика сервиса"""
        cache_stats = self.cache.stats()

        total_predictions = self.metrics['total_predictions']
        avg_processing_time = (
            self.metrics['total_processing_time'] / total_predictions
            if total_predictions > 0 else 0
        )

        # Анализ размеров батчей
        batch_sizes = self.metrics['batch_sizes']
        avg_batch_size = np.mean(batch_sizes) if batch_sizes else 0

        uptime = time.time() - self.start_time

        return {
            "model_loaded": self.model_loaded,
            "model_path": self.model_path,
            "total_predictions": total_predictions,
            "avg_processing_time": round(avg_processing_time, 4),
            "avg_batch_size": round(avg_batch_size, 1),
            "errors": self.metrics['errors'],
            "error_rate": round(self.metrics['errors'] / max(total_predictions, 1), 4),
            "uptime_hours": round(uptime / 3600, 2),
            "cache": cache_stats,
            "processes": settings.PROCESS_POOL_WORKERS,
            "threads": settings.THREAD_POOL_WORKERS,
            "historical_data_loaded": len(self.historical_service.driver_stats) > 0
        }

    async def shutdown(self):
        """Корректное завершение работы"""
        logger.info("🛑 Завершаем работу AsyncMLPredictor...")

        # Очистка кэша
        self.cache._cleanup_expired()

        try:
            # Просто вызываем shutdown без таймаутов
            self.process_pool.shutdown(wait=True)
            self.thread_pool.shutdown(wait=True)
            logger.info("✅ ProcessPool и ThreadPool завершены")

        except Exception as e:
            logger.warning(f"⚠️ Ошибка при завершении пулов: {e}")

        logger.info("✅ AsyncMLPredictor остановлен")