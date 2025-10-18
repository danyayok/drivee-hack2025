import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import joblib
import os
from app.utils.logger import get_logger

logger = get_logger(__name__)


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
        stats = self.driver_stats.get(driver_id, {})
        return {
            'driver_acceptance_rate': stats.get('acceptance_rate', 0.35),
            'driver_total_orders': stats.get('total_orders', 0),
            'driver_avg_rating': stats.get('avg_rating', 4.0),
            'driver_avg_bid_price': stats.get('avg_bid_price', 200.0)
        }

    def get_user_stats(self, user_id) -> Dict:
        """Получаем статистику пользователя"""
        stats = self.user_stats.get(user_id, {})
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