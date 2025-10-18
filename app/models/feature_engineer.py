import pandas as pd
import numpy as np
from datetime import datetime
from typing import List


class FeatureEngineer:
    """Инженерия фичей для ML модели"""

    def __init__(self):
        self.feature_columns = []

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание всех фичей из сырых данных
        """
        df = df.copy()

        # === ВРЕМЕННЫЕ ФИЧИ ===
        if 'order_timestamp' in df.columns:
            df['order_timestamp'] = pd.to_datetime(df['order_timestamp'])
            df['order_hour'] = df['order_timestamp'].dt.hour
            df['order_dayofweek'] = df['order_timestamp'].dt.dayofweek
            df['order_month'] = df['order_timestamp'].dt.month
            df['is_weekend'] = (df['order_dayofweek'] >= 5).astype(int)

            # Время суток
            df['is_night'] = ((df['order_hour'] >= 23) | (df['order_hour'] <= 5)).astype(int)
            df['is_morning_rush'] = ((df['order_hour'] >= 7) & (df['order_hour'] <= 9)).astype(int)
            df['is_evening_rush'] = ((df['order_hour'] >= 17) & (df['order_hour'] <= 20)).astype(int)

        # === ЦЕНОВЫЕ ФИЧИ ===
        if 'price_bid_local' in df.columns and 'price_start_local' in df.columns:
            df['price_increase_pct'] = (
                    (df['price_bid_local'] - df['price_start_local']) /
                    df['price_start_local'] * 100
            )
            df['price_per_km'] = df['price_bid_local'] / (df['distance_in_meters'] / 1000)
            df['price_start_per_km'] = df['price_start_local'] / (df['distance_in_meters'] / 1000)

        # === ГЕОГРАФИЧЕСКИЕ ФИЧИ ===
        if 'distance_in_meters' in df.columns:
            df['distance_km'] = df['distance_in_meters'] / 1000

        if 'duration_in_seconds' in df.columns:
            df['duration_minutes'] = df['duration_in_seconds'] / 60

        if 'pickup_in_meters' in df.columns:
            df['pickup_km'] = df['pickup_in_meters'] / 1000

        if 'pickup_in_seconds' in df.columns:
            df['pickup_minutes'] = df['pickup_in_seconds'] / 60

        # Средняя скорость
        if 'distance_km' in df.columns and 'duration_minutes' in df.columns:
            df['avg_speed_kmh'] = (df['distance_km'] / (df['duration_minutes'] / 60))
            df['avg_speed_kmh'] = df['avg_speed_kmh'].replace([np.inf, -np.inf], 0)

        # === ФИЧИ ВОДИТЕЛЯ ===
        if 'driver_reg_date' in df.columns:
            df['driver_reg_date'] = pd.to_datetime(df['driver_reg_date'])
            df['driver_experience_days'] = (datetime.now() - df['driver_reg_date']).dt.days

        if 'driver_platform' in df.columns:
            df['is_ios'] = (df['driver_platform'] == 'iOS').astype(int)

        # === ДОПОЛНИТЕЛЬНЫЕ ФИЧИ ===
        # Эффективность поездки
        if 'pickup_minutes' in df.columns and 'duration_minutes' in df.columns:
            df['waiting_efficiency'] = df['pickup_minutes'] / (df['duration_minutes'] + 1)

        # Тип поездки
        if 'distance_km' in df.columns:
            df['is_short_trip'] = (df['distance_km'] < 3).astype(int)
            df['is_medium_trip'] = ((df['distance_km'] >= 3) & (df['distance_km'] <= 15)).astype(int)
            df['is_long_trip'] = (df['distance_km'] > 15).astype(int)

        return df

    def get_feature_columns(self) -> List[str]:
        """
        Возвращает список фичей для обучения модели
        """
        features = [
            # Ценовые
            'price_start_local', 'price_increase_pct', 'price_per_km', 'price_start_per_km',

            # Временные
            'order_hour', 'order_dayofweek', 'order_month', 'is_weekend',
            'is_night', 'is_morning_rush', 'is_evening_rush',

            # Гео
            'distance_km', 'duration_minutes', 'pickup_km', 'pickup_minutes',
            'avg_speed_kmh', 'waiting_efficiency',

            # Тип поездки
            'is_short_trip', 'is_medium_trip', 'is_long_trip',

            # Социальные
            'driver_rating', 'user_rating', 'driver_experience_days', 'is_ios'
        ]

        self.feature_columns = [f for f in features if f in [
            'price_start_local', 'price_increase_pct', 'price_per_km',
            'order_hour', 'order_dayofweek', 'is_weekend', 'is_night',
            'distance_km', 'duration_minutes', 'pickup_km', 'pickup_minutes',
            'driver_rating', 'user_rating', 'driver_experience_days', 'is_ios'
        ]]

        return self.feature_columns