import asyncio
import time
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Dict, Optional
import joblib
import hashlib
import json
from dataclasses import dataclass
from catboost import Pool
import platform
import os
import signal
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
    """Инициализация воркера процесса - только для Windows"""
    if platform.system() == "Windows":
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        except:
            pass  # Игнорируем ошибки сигналов


class HistoricalDataService:
    def __init__(self, data_path: str = None):
        # ✅ КРОССПЛАТФОРМЕННЫЙ ПУТЬ К ДАННЫМ
        self.data_path = data_path or settings.DATA_PATH
        self.df = pd.DataFrame()
        self.driver_stats = {}
        self.user_stats = {}
        self.avg_response_delay = 5.0

        # ✅ ПРОВЕРКА СУЩЕСТВОВАНИЯ ФАЙЛА ДЛЯ LINUX
        if not os.path.exists(self.data_path):
            logger.warning(f"⚠️ Файл данных не найден: {self.data_path}")
            # Пробуем найти в текущей директории для Linux
            fallback_path = "train.csv"
            if os.path.exists(fallback_path):
                self.data_path = fallback_path
                logger.info(f"✅ Используем fallback путь: {fallback_path}")
            else:
                logger.warning("⚠️ Файл train.csv не найден, используем пустые данные")
                return

        self._load_data()

    def _get_fallback_driver_stats(self):
        """Fallback статистика для водителя"""
        return {
            'driver_acceptance_rate': 0.35,
            'driver_total_orders': 0,
            'driver_avg_rating': 4.0,
            'driver_avg_bid_price': 200.0
        }

    def _get_fallback_user_stats(self):
        """Fallback статистика для пользователя"""
        return {
            'user_acceptance_rate': 0.4,
            'user_total_orders': 0,
            'user_avg_rating': 4.5,
            'user_avg_bid_price': 200.0
        }

    def get_driver_stats(self, driver_id) -> Dict:
        """Универсальный поиск с исправленными типами"""
        if driver_id is None:
            return self._get_fallback_driver_stats()

        # Приводим к int (как теперь хранится в self.driver_stats)
        try:
            driver_id_int = int(driver_id)
        except (ValueError, TypeError):
            logger.warning(f"⚠️ Неверный формат driver_id: {driver_id}")
            return self._get_fallback_driver_stats()

        if driver_id_int in self.driver_stats:
            stats = self.driver_stats[driver_id_int]
            logger.info(f"✅ Найдены реальные данные для водителя {driver_id_int}")
            return {
                'driver_acceptance_rate': stats.get('acceptance_rate', 0.35),
                'driver_total_orders': stats.get('total_orders', 0),
                'driver_avg_rating': stats.get('avg_rating', 4.0),
                'driver_avg_bid_price': stats.get('avg_bid_price', 200.0)
            }

        logger.warning(f"⚠️ Данные для водителя {driver_id_int} не найдены, используем fallback")
        return self._get_fallback_driver_stats()

    def get_user_stats(self, user_id) -> Dict:
        """Исправленная версия для пользователей"""
        if user_id is None:
            return self._get_fallback_user_stats()

        # Приводим к int (как теперь хранится в self.user_stats)
        try:
            user_id_int = int(user_id)
        except (ValueError, TypeError):
            logger.warning(f"⚠️ Неверный формат user_id: {user_id}")
            return self._get_fallback_user_stats()

        if user_id_int in self.user_stats:
            stats = self.user_stats[user_id_int]
            logger.info(f"✅ Найдены реальные данные для пользователя {user_id_int}")
            return {
                'user_acceptance_rate': stats.get('acceptance_rate', 0.4),
                'user_total_orders': stats.get('total_orders', 0),
                'user_avg_rating': stats.get('avg_rating', 4.5),
                'user_avg_bid_price': stats.get('avg_bid_price', 200.0)
            }

        logger.warning(f"⚠️ Данные для пользователя {user_id_int} не найдены, используем fallback")
        return self._get_fallback_user_stats()

    def _load_data(self):
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"🔍 Проверка данных - driver_id примеры: {self.df['driver_id'].head(10).tolist()}")
            print(f"🔍 Проверка данных - user_id примеры: {self.df['user_id'].head(10).tolist()}")

            # Проверьте конкретно нужные ID
            driver_29368889 = self.df[self.df['driver_id'] == 29368889]
            user_16458846 = self.df[self.df['user_id'] == 16458846]

            print(f"🔍 Найдено записей для driver_id 29368889: {len(driver_29368889)}")
            print(f"🔍 Найдено записей для user_id 16458846: {len(user_16458846)}")
            self.df['is_done'] = self.df['is_done'].astype(str).str.lower().map({"done": 1, "cancel": 0})
            self.df = self.df[self.df['is_done'].isin([0, 1])]

            # 🔥 ДОБАВИТЬ user_rating ЕСЛИ ЕГО НЕТ
            if 'user_rating' not in self.df.columns:
                self.df['user_rating'] = 4.6  # среднее значение
                logger.info("🔧 Добавили user_rating в данные")

            self._calculate_stats()
            logger.info(f"✅ Загружено {len(self.df)} исторических записей")
            logger.info(f"📊 Колонки: {self.df.columns.tolist()}")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки исторических данных: {e}")
            self.df = pd.DataFrame()

    def _calculate_stats(self):
        """ПОЛНОСТЬЮ ИСПРАВЛЕННАЯ версия расчета статистик"""
        try:
            # Driver statistics
            driver_stats = self.df.groupby('driver_id').agg({
                'is_done': ['mean', 'count'],
                'driver_rating': 'mean',
                'price_bid_local': 'mean'
            }).round(3)

            driver_stats.columns = ['acceptance_rate', 'total_orders', 'avg_rating', 'avg_bid_price']

            # 🔥 ИСПРАВЛЕНИЕ: Явное преобразование в dict с проверкой
            self.driver_stats = {}
            for driver_id, row in driver_stats.iterrows():
                # Явно приводим driver_id к int (как в исходных данных)
                driver_id_int = int(driver_id) if pd.notna(driver_id) else None
                if driver_id_int:
                    self.driver_stats[driver_id_int] = {
                        'acceptance_rate': row['acceptance_rate'],
                        'total_orders': int(row['total_orders']),
                        'avg_rating': row['avg_rating'],
                        'avg_bid_price': row['avg_bid_price']
                    }

            # User statistics
            user_stats = self.df.groupby('user_id').agg({
                'is_done': ['mean', 'count'],
                'user_rating': 'mean',
                'price_bid_local': 'mean'
            }).round(3)

            user_stats.columns = ['acceptance_rate', 'total_orders', 'avg_rating', 'avg_bid_price']

            self.user_stats = {}
            for user_id, row in user_stats.iterrows():
                user_id_int = int(user_id) if pd.notna(user_id) else None
                if user_id_int:
                    self.user_stats[user_id_int] = {
                        'acceptance_rate': row['acceptance_rate'],
                        'total_orders': int(row['total_orders']),
                        'avg_rating': row['avg_rating'],
                        'avg_bid_price': row['avg_bid_price']
                    }

            logger.info(
                f"✅ Рассчитано stats для {len(self.driver_stats)} водителей и {len(self.user_stats)} пользователей")
            logger.info(f"🔍 Проверка driver_id 29368889: {'29368889' in self.driver_stats}")
            logger.info(f"🔍 Проверка user_id 16458846: {'16458846' in self.user_stats}")
            logger.info(f"🔍 Проверка конкретных ID в driver_stats:")
            if 29368889 in self.driver_stats:
                logger.info(f"✅ driver_id 29368889: {self.driver_stats[29368889]}")
            else:
                logger.info(f"❌ driver_id 29368889 не найден")
                logger.info(f"🔍 Доступные driver_ids: {list(self.driver_stats.keys())[:10]}")

            if 16458846 in self.user_stats:
                logger.info(f"✅ user_id 16458846: {self.user_stats[16458846]}")
            else:
                logger.info(f"❌ user_id 16458846 не найден")
        except Exception as e:
            logger.error(f"❌ Ошибка расчета статистик: {e}")
            self.driver_stats = {}
            self.user_stats = {}

    def get_combined_stats(self, driver_id, user_id) -> Dict:
        """ИСПРАВЛЕННАЯ версия"""
        driver_stats = self.get_driver_stats(driver_id)
        user_stats = self.get_user_stats(user_id)

        return {
            'user_acceptance_rate': user_stats['user_acceptance_rate'],
            'driver_acceptance_rate': driver_stats['driver_acceptance_rate'],
            'user_avg_bid_price': user_stats.get('user_avg_bid_price', 200),
            'driver_avg_bid_price': driver_stats.get('driver_avg_bid_price', 200),
            'avg_response_delay': self.avg_response_delay
        }


# Глобальный экземпляр
historical_service = HistoricalDataService()


class ModelLoader:
    _models = {}

    @classmethod
    def load_model(cls, model_path: str):
        if model_path in cls._models:
            return cls._models[model_path]

        try:
            logger.info(f"🔄 Загружаем CatBoost модель в процессе {os.getpid()}")

            # ✅ ПРОВЕРКА СУЩЕСТВОВАНИЯ ФАЙЛА ДЛЯ LINUX
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Файл модели не найден: {model_path}")

            model_data = joblib.load(model_path)

            if isinstance(model_data, dict) and 'model' in model_data:
                model = model_data['model']
                feature_names = model_data.get('feature_names', None)
                cat_features = model_data.get('cat_features', None)

                # 🔥 ФИЛЬТРУЕМ driver_id ИЗ КАТЕГОРИАЛЬНЫХ ФИЧ
                if cat_features:
                    cat_features = [f for f in cat_features if f != 'driver_id']
                    logger.info(f"🔧 Отфильтровали cat_features: {cat_features}")
                elif cat_features is None:
                    cat_features = ["carmodel", "carname", "platform"]
                    logger.info(f"🔧 Создали cat_features: {cat_features}")
            else:
                model = model_data
                feature_names = None
                cat_features = ["carmodel", "carname", "platform"]

            cls._models[model_path] = (model, feature_names, cat_features)
            return model, feature_names, cat_features

        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            raise


def prepare_features_sync(features: Dict) -> pd.DataFrame:
    """ТОЧНОЕ соответствие фичам из обучения - ИСПРАВЛЕННАЯ ВЕРСИЯ"""
    df = pd.DataFrame([features])

    # === ОСНОВНЫЕ ФИЧИ ===
    price_start = features.get('price_start_local', 200)
    price_bid = features.get('price_bid_local', 200)

    df["price_ratio"] = price_bid / (price_start + 1e-6)
    df["price_diff"] = price_bid - price_start

    # Гео-фичи
    df["distance_km"] = features.get('distance_in_meters', 2000) / 1000
    df["duration_min"] = features.get('duration_in_seconds', 600) / 60
    df["trip_speed_kmh"] = df["distance_km"] / (df["duration_min"] / 60 + 1e-6)

    # Ценовая эффективность
    df["price_per_km"] = price_bid / (df["distance_km"] + 1e-6)
    df["price_per_min"] = price_bid / (df["duration_min"] + 1e-6)

    # Взаимодействия
    df["rating_price_interaction"] = features.get('driver_rating', 4.5) * df["price_ratio"]
    df["rating_speed_interaction"] = features.get('driver_rating', 4.5) * df["trip_speed_kmh"]

    # Время
    if 'order_timestamp' in features:
        try:
            order_time = pd.to_datetime(features['order_timestamp'])
            df["order_hour"] = order_time.hour
            df["order_dayofweek"] = order_time.dayofweek
            df["order_month"] = order_time.month
            df["is_peak_hour"] = df["order_hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
            df["is_weekend"] = 1 if order_time.dayofweek >= 5 else 0
        except:
            # Fallback значения
            df["order_hour"] = 12
            df["order_dayofweek"] = 0
            df["order_month"] = 5
            df["is_peak_hour"] = 0
            df["is_weekend"] = 0
    else:
        df["order_hour"] = 12
        df["order_dayofweek"] = 0
        df["order_month"] = 5
        df["is_peak_hour"] = 0
        df["is_weekend"] = 0

    # Пикап
    df["pickup_in_meters"] = features.get('pickup_in_meters', 500)
    df["pickup_in_seconds"] = features.get('pickup_in_seconds', 120)
    df["pickup_speed_kmh"] = (df["pickup_in_meters"] / 1000) / (df["pickup_in_seconds"] / 3600 + 1e-6)

    driver_id = features.get('driver_id')
    user_id = features.get('user_id')

    # Получаем исторические данные
    driver_stats = historical_service.get_driver_stats(driver_id)
    user_stats = historical_service.get_user_stats(user_id)

    df["driver_cancel_rate"] = 1 - driver_stats.get('driver_acceptance_rate', 0.35)
    df["user_cancel_rate"] = 1 - user_stats.get('user_acceptance_rate', 0.4)
    df["driver_avg_price"] = driver_stats.get('driver_avg_bid_price', 200)
    df["user_avg_price"] = user_stats.get('user_avg_bid_price', 200)

    # Response delay
    df["response_delay_sec"] = 5.0  # временное значение

    # Опыт водителя
    if 'driver_reg_date' in features and 'order_timestamp' in features:
        try:
            reg_date = pd.to_datetime(features['driver_reg_date'])
            order_time = pd.to_datetime(features['order_timestamp'])
            df["driver_experience_days"] = (order_time - reg_date).days
        except:
            df["driver_experience_days"] = 365
    else:
        df["driver_experience_days"] = 365

    # Логарифмические фичи
    df["log_price_bid_local"] = np.log1p(price_bid)
    df["log_distance_in_meters"] = np.log1p(features.get('distance_in_meters', 2000))
    df["log_duration_in_seconds"] = np.log1p(features.get('duration_in_seconds', 600))
    df["log_driver_experience_days"] = np.log1p(df["driver_experience_days"])

    # Взаимодействие опыта и цены
    df["experience_price_ratio"] = df["driver_experience_days"] * df["price_ratio"]

    # === КАТЕГОРИАЛЬНЫЕ ФИЧИ ===
    df["platform"] = str(features.get('driver_platform', 'android'))
    df["carmodel"] = str(features.get('carmodel', 'unknown'))
    df["carname"] = str(features.get('carname', 'unknown'))

    # === ОСНОВНЫЕ ЧИСЛОВЫЕ ФИЧИ ===
    df["distance_in_meters"] = features.get('distance_in_meters', 2000)
    df["duration_in_seconds"] = features.get('duration_in_seconds', 600)
    df["pickup_in_meters"] = features.get('pickup_in_meters', 500)
    df["pickup_in_seconds"] = features.get('pickup_in_seconds', 120)
    df["driver_rating"] = features.get('driver_rating', 4.5)
    df["price_start_local"] = price_start
    df["price_bid_local"] = price_bid

    df["user_rating"] = features.get('user_rating', 4.6)

    return df


def predict_batch_sync(model_path: str, features_list: List[Dict]) -> np.ndarray:
    start_time = time.time()
    try:
        model, feature_names, cat_features = ModelLoader.load_model(model_path)

        # 🔥 ЕСЛИ cat_features НЕТ - СОЗДАЕМ САМИ
        if not cat_features:
            cat_features = ["carmodel", "carname", "platform"]
            logger.info(f"🔧 Создали cat_features: {cat_features}")

        if not features_list:
            return np.array([])

        prepared_features = []
        for features in features_list:
            try:
                df = prepare_features_sync(features)

                if feature_names:
                    for f in feature_names:
                        if f not in df.columns:
                            df[f] = 0.0
                    df = df[feature_names]

                prepared_features.append(df)
            except Exception as e:
                logger.warning(f"⚠️ Ошибка подготовки фичей: {e}")
                fallback_data = {f: 0.0 for f in (feature_names or [])}
                fallback_df = pd.DataFrame([fallback_data])
                prepared_features.append(fallback_df)

        combined_df = pd.concat(prepared_features, ignore_index=True)

        # 🔥 КРИТИЧЕСКИ ВАЖНО: используем Pool с cat_features
        logger.info(f"🔧 Предсказание с cat_features: {cat_features}")

        # Создаем Pool с указанием категориальных фич
        pool = Pool(combined_df, cat_features=cat_features)
        probabilities = model.predict_proba(pool)

        if probabilities.ndim == 2 and probabilities.shape[1] == 2:
            probabilities = probabilities[:, 1]

        logger.info(f"🎯 Диапазон вероятностей: {probabilities.min():.3f} - {probabilities.max():.3f}")
        probabilities = np.clip(probabilities, 0.05, 0.95)
        return probabilities

    except Exception as e:
        logger.error(f"❌ Ошибка предсказания: {e}")
        return np.full(len(features_list), 0.5)


class CacheWithTTL:
    """Потокобезопасный кэш с TTL"""

    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        self._cache = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._last_cleanup = time.time()

    def get(self, key: str) -> Optional[any]:
        with self._lock:
            if time.time() - self._last_cleanup > 60:
                self._cleanup_expired()
            entry = self._cache.get(key)
            if entry and time.time() - entry['timestamp'] < self.ttl:
                self._hits += 1
                return entry['value']
            if entry:
                del self._cache[key]
            self._misses += 1
            return None

    def set(self, key: str, value: any):
        with self._lock:
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]['timestamp'])
                del self._cache[oldest_key]
            self._cache[key] = {'value': value, 'timestamp': time.time()}

    def _cleanup_expired(self):
        current_time = time.time()
        expired = [k for k, v in self._cache.items() if current_time - v['timestamp'] > self.ttl]
        for k in expired:
            del self._cache[k]
        self._last_cleanup = current_time

    def stats(self) -> Dict:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total else 0
            return {'size': len(self._cache), 'hits': self._hits, 'misses': self._misses,
                    'hit_rate': round(hit_rate, 3),
                    'max_size': self.max_size, 'ttl': self.ttl}


class AsyncMLPredictor:
    """Асинхронный ML сервис с историческими данными"""

    def __init__(self, model_path: str = settings.MODEL_PATH):
        self.model_path = settings.model_path  # ✅ Используем property из config
        self.model_loaded = False
        self.model = None
        self.feature_names = None
        self.historical_service = historical_service

        # ✅ КРОСС-ПЛАТФОРМЕННЫЕ ПУЛЫ
        if platform.system() == "Windows" and settings.PROCESS_POOL_WORKERS > 0:
            self.process_pool = ProcessPoolExecutor(
                max_workers=settings.PROCESS_POOL_WORKERS,
                initializer=init_worker
            )
            logger.info(f"🔧 ProcessPool создан для Windows: {settings.PROCESS_POOL_WORKERS} workers")
        else:
            self.process_pool = None
            logger.info("🔧 ProcessPool отключен для Linux")

        self.thread_pool = ThreadPoolExecutor(max_workers=settings.THREAD_POOL_WORKERS)
        self.cache = CacheWithTTL(max_size=settings.MODEL_CACHE_SIZE, ttl=settings.CACHE_TTL)
        self.metrics = {'total_predictions': 0, 'total_processing_time': 0.0, 'errors': 0, 'batch_sizes': []}
        self.start_time = time.time()
        logger.info(f"🚀 AsyncMLPredictor инициализирован для {platform.system()}")

    async def initialize(self):
        """Инициализация модели"""
        try:
            # ✅ ПРОВЕРКА СУЩЕСТВОВАНИЯ МОДЕЛИ ДЛЯ LINUX
            if not os.path.exists(self.model_path):
                # Пробуем найти модель в альтернативных путях для Linux
                if platform.system() != "Windows":
                    alternative_paths = [
                        "models/catboost_taxi_smart.joblib",
                        "./catboost_taxi_smart.joblib",
                        "/app/models/catboost_taxi_smart.joblib"
                    ]

                    for alt_path in alternative_paths:
                        if os.path.exists(alt_path):
                            self.model_path = alt_path
                            logger.info(f"✅ Найдена модель по альтернативному пути: {alt_path}")
                            break
                    else:
                        raise FileNotFoundError(f"Файл модели не найден: {self.model_path}")
                else:
                    raise FileNotFoundError(f"Файл модели не найден: {self.model_path}")

            loop = asyncio.get_event_loop()
            model_data = await loop.run_in_executor(self.thread_pool, joblib.load, self.model_path)

            if isinstance(model_data, dict) and 'model' in model_data:
                self.model = model_data['model']
                self.feature_names = model_data.get('feature_names', None)
            else:
                self.model = model_data
                self.feature_names = None

            self.model_loaded = True
            logger.info(f"✅ Модель загружена: {self.model_path}")

        except Exception as e:
            logger.error(f"❌ Ошибка инициализации модели: {e}")
            self.model_loaded = False
            raise

    async def predict_batch(self, features_list: List[Dict]) -> List[PredictionResult]:
        if not features_list or not self.model_loaded:
            return [PredictionResult(0.5, 0.0, error="Service not ready") for _ in features_list]

        start_time = time.time()
        cache_key = self._generate_cache_key(features_list)
        cached = self.cache.get(cache_key)
        if cached is not None:
            processing_time = (time.time() - start_time) / len(cached)
            return [PredictionResult(prob, processing_time, cache_hit=True) for prob in cached]

        try:
            loop = asyncio.get_event_loop()

            # ✅ ВСЕГДА ИСПОЛЬЗУЕМ THREADPOOL НА LINUX
            if self.process_pool and platform.system() == "Windows":
                probabilities = await loop.run_in_executor(
                    self.process_pool, predict_batch_sync, self.model_path, features_list
                )
            else:
                probabilities = await loop.run_in_executor(
                    self.thread_pool, predict_batch_sync, self.model_path, features_list
                )

            processing_time = time.time() - start_time
            self.cache.set(cache_key, probabilities.tolist())
            self._update_metrics(processing_time, len(features_list))
            return [PredictionResult(float(prob), processing_time / len(features_list)) for prob in probabilities]

        except asyncio.TimeoutError:
            self.metrics['errors'] += 1
            return [PredictionResult(0.5, time.time() - start_time, error="Prediction timeout") for _ in features_list]
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"❌ Ошибка предсказания: {e}")
            return [PredictionResult(0.5, time.time() - start_time, error=str(e)) for _ in features_list]

    async def predict_single(self, features: Dict) -> PredictionResult:
        results = await self.predict_batch([features])
        return results[0] if results else PredictionResult(0.5, 0.0, error="No results")

    async def predict_price_acceptance(self, base_features: Dict, test_prices: List[float]) -> List[PredictionResult]:
        scenarios = [{**base_features, 'price_bid_local': price} for price in test_prices]
        return await self.predict_batch(scenarios)

    def _generate_cache_key(self, features_list: List[Dict]) -> str:
        try:
            key_parts = []
            for features in features_list:
                key_data = {
                    'price_start': round(features.get('price_start_local', 0), 2),
                    'price_bid': round(features.get('price_bid_local', 0), 2),
                    'distance': round(features.get('distance_in_meters', 0)),
                    'driver_id': str(features.get('driver_id', '')),
                    'user_id': str(features.get('user_id', ''))
                }
                key_parts.append(json.dumps(key_data, sort_keys=True))
            return hashlib.md5("|".join(key_parts).encode()).hexdigest()
        except Exception as e:
            logger.warning(f"⚠️ Ошибка генерации ключа кэша: {e}")
            return hashlib.md5(str(len(features_list)).encode()).hexdigest()

    def _update_metrics(self, processing_time: float, batch_size: int):
        self.metrics['total_predictions'] += batch_size
        self.metrics['total_processing_time'] += processing_time
        self.metrics['batch_sizes'].append(batch_size)
        if len(self.metrics['batch_sizes']) > 1000:
            self.metrics['batch_sizes'] = self.metrics['batch_sizes'][-1000:]

    def get_stats(self) -> Dict:
        cache_stats = self.cache.stats()
        total_predictions = self.metrics['total_predictions']
        avg_processing_time = (self.metrics['total_processing_time'] / total_predictions) if total_predictions else 0
        avg_batch_size = np.mean(self.metrics['batch_sizes']) if self.metrics['batch_sizes'] else 0
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
        logger.info("🛑 Завершаем работу AsyncMLPredictor...")
        self.cache._cleanup_expired()
        try:
            if self.process_pool:
                self.process_pool.shutdown(wait=True)
                logger.info("✅ ProcessPool завершен")
            self.thread_pool.shutdown(wait=True)
            logger.info("✅ ThreadPool завершен")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка при завершении пулов: {e}")
        logger.info("✅ AsyncMLPredictor остановлен")
