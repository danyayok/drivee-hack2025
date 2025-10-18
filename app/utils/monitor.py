import time
import asyncio
from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Prometheus метрики
PREDICTION_REQUESTS = Counter(
    'prediction_requests_total',
    'Total prediction requests',
    ['endpoint', 'status']
)

PREDICTION_DURATION = Histogram(
    'prediction_duration_seconds',
    'Prediction processing time',
    ['endpoint']
)

ACTIVE_REQUESTS = Gauge(
    'active_requests',
    'Currently active requests'
)

CACHE_HITS = Counter(
    'cache_hits_total',
    'Total cache hits'
)

CACHE_MISSES = Counter(
    'cache_misses_total',
    'Total cache misses'
)


class MetricsCollector:
    """Сборщик метрик для мониторинга"""

    def __init__(self):
        self.start_time = time.time()

    async def track_request(self, endpoint: str):
        """Трекинг начала запроса"""
        ACTIVE_REQUESTS.inc()
        return time.time()

    async def track_completion(self, endpoint: str, start_time: float, success: bool = True):
        """Трекинг завершения запроса"""
        ACTIVE_REQUESTS.dec()
        duration = time.time() - start_time

        PREDICTION_REQUESTS.labels(endpoint=endpoint, status='success' if success else 'error').inc()
        PREDICTION_DURATION.labels(endpoint=endpoint).observe(duration)

    async def track_cache_hit(self):
        """Трекинг попадания в кэш"""
        CACHE_HITS.inc()

    async def track_cache_miss(self):
        """Трекинг промаха кэша"""
        CACHE_MISSES.inc()

    def get_uptime(self) -> float:
        """Время работы сервиса"""
        return time.time() - self.start_time


# Глобальный сборщик метрик
metrics = MetricsCollector()