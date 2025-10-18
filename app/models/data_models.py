from pydantic import BaseModel, Field, validator, model_validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class DriverPlatform(str, Enum):
    IOS = "iOS"
    ANDROID = "Android"

    @classmethod
    def _missing_(cls, value):
        """Обработка значений в разных регистрах"""
        if isinstance(value, str):
            # Приводим к нижнему регистру для сравнения
            value_lower = value.lower()
            for member in cls:
                if member.value.lower() == value_lower:
                    return member
        return None


class OrderRequest(BaseModel):
    """Запрос на получение оптимальных цен"""
    distance_in_meters: float = Field(..., gt=0, description="Дистанция поездки в метрах")
    duration_in_seconds: float = Field(..., gt=0, description="Длительность поездки в секундах")
    pickup_in_meters: float = Field(..., gt=0, description="Дистанция до пассажира в метрах")
    pickup_in_seconds: float = Field(..., gt=0, description="Время до пассажира в секундах")
    driver_rating: float = Field(..., ge=0, le=5, description="Рейтинг водителя")
    user_rating: float = Field(..., ge=0, le=5, description="Рейтинг пассажира")
    price_start_local: float = Field(..., gt=0, description="Начальная цена пассажира")
    order_timestamp: str = Field(..., description="Время создания заказа")
    driver_platform: DriverPlatform = Field(default=DriverPlatform.ANDROID, description="Платформа водителя")
    driver_reg_date: str = Field(default="2023-01-01", description="Дата регистрации водителя")
    carname: Optional[str] = Field(None, description="Марка автомобиля")
    carmodel: Optional[str] = Field(None, description="Модель автомобиля")

    # 🔥 НОВЫЕ ПОЛЯ - ID для исторических данных
    driver_id: Optional[str] = Field(None, description="ID водителя для исторических данных")
    user_id: Optional[str] = Field(None, description="ID пользователя для исторических данных")

    @model_validator(mode='before')
    @classmethod
    def validate_and_normalize_data(cls, values):
        """Нормализация данных перед валидацией"""
        # Нормализация driver_platform
        if 'driver_platform' in values and values['driver_platform']:
            platform = values['driver_platform']
            if isinstance(platform, str):
                platform_lower = platform.lower()
                if platform_lower == 'ios':
                    values['driver_platform'] = DriverPlatform.IOS
                elif platform_lower in ['android', 'андроид']:
                    values['driver_platform'] = DriverPlatform.ANDROID

        # Нормализация дат
        date_fields = ['order_timestamp', 'driver_reg_date']
        for field in date_fields:
            if field in values and values[field]:
                try:
                    # Если это timestamp как число
                    if isinstance(values[field], (int, float)):
                        values[field] = datetime.fromtimestamp(values[field]).isoformat()
                    # Если это строка с timestamp
                    elif isinstance(values[field], str) and values[field].isdigit():
                        values[field] = datetime.fromtimestamp(int(values[field])).isoformat()
                    # Если это строка в формате 'YYYY-MM-DD HH:MM:SS'
                    elif isinstance(values[field], str) and ' ' in values[field]:
                        dt = datetime.strptime(values[field], '%Y-%m-%d %H:%M:%S')
                        values[field] = dt.isoformat()
                except (ValueError, TypeError):
                    # Оставляем как есть, если не получается преобразовать
                    pass

        return values

    @model_validator(mode='after')
    def validate_business_logic(self):
        """Бизнес-логика валидации"""
        # Проверка что время подачи не больше времени поездки
        if self.pickup_in_seconds > self.duration_in_seconds:
            raise ValueError('Время подачи не может быть больше времени поездки')

        # Проверка что дистанция подачи не больше дистанции поездки
        if self.pickup_in_meters > self.distance_in_meters:
            raise ValueError('Дистанция подачи не может быть больше дистанции поездки')

        return self


class PricePoint(BaseModel):
    """Одна точка цены с вероятностью, доходом и распределением"""
    price: float = Field(..., description="Цена", examples=[350.0])
    probability: float = Field(..., ge=0, le=1, description="Вероятность принятия", examples=[0.85])
    expected_revenue: float = Field(..., description="Ожидаемый доход", examples=[297.5])
    service_commission: float = Field(..., description="Комиссия сервиса (12.8%)", examples=[38.08])
    driver_earnings: float = Field(..., description="Заработок водителя", examples=[259.42])


class OptimalPricesResponse(BaseModel):
    """Ответ с оптимальными ценами"""
    price_curve: List[PricePoint] = Field(..., description="Кривая цена-вероятность-доход")
    processing_time_ms: float = Field(..., description="Время обработки в миллисекундах", examples=[245.5])
    analysis: Dict[str, Any] = Field(..., description="Анализ результатов")


class HealthResponse(BaseModel):
    """Статус здоровья сервиса"""
    status: str = Field(..., description="Статус сервиса", examples=["healthy"])
    model_loaded: bool = Field(..., description="Модель загружена", examples=[True])
    total_predictions: int = Field(..., description="Общее количество предсказаний", examples=[1500])
    cache_hit_rate: float = Field(..., description="Процент попаданий в кэш", examples=[0.65])
    avg_processing_time: float = Field(..., description="Среднее время обработки", examples=[0.245])
    uptime_seconds: float = Field(..., description="Время работы сервиса", examples=[86400.5])


class ServiceStats(BaseModel):
    """Статистика сервиса"""
    total_requests: int = Field(..., description="Всего обработано запросов")
    successful_requests: int = Field(..., description="Успешных запросов")
    failed_requests: int = Field(..., description="Неудачных запросов")
    avg_processing_time_ms: float = Field(..., description="Среднее время обработки")
    cache_hit_rate: float = Field(..., description="Процент попаданий в кэш")

    # Системные метрики
    memory_usage_mb: float = Field(..., description="Использование памяти (МБ)")
    memory_available_mb: float = Field(..., description="Свободная память (МБ)")
    cpu_usage_percent: float = Field(..., description="Загрузка CPU (%)")

    # Финансовая статистика
    total_service_revenue: float = Field(..., description="Общая выручка сервиса")
    total_driver_earnings: float = Field(..., description="Общий заработок водителей")
    total_commission: float = Field(..., description="Общая комиссия сервиса")

    # Активность по водителям
    active_drivers: int = Field(..., description="Активные водители")
    top_drivers: List[Dict[str, Any]] = Field(..., description="Топ водители по заработку")


class DriverEarnings(BaseModel):
    """Заработок водителя"""
    driver_id: str = Field(..., description="ID водителя")
    total_earnings: float = Field(..., description="Общий заработок")
    completed_orders: int = Field(..., description="Выполненные заказы")
    avg_earning_per_order: float = Field(..., description="Средний заработок за заказ")


class ErrorResponse(BaseModel):
    """Стандартный ответ об ошибке"""
    detail: str = Field(..., description="Описание ошибки")
    error_code: Optional[str] = Field(None, description="Код ошибки")
    timestamp: str = Field(..., description="Время возникновения ошибки")

    @model_validator(mode='after')
    def set_timestamp(self):
        """Установка временной метки"""
        self.timestamp = datetime.now().isoformat()
        return self