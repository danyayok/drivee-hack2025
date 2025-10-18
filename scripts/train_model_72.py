import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import joblib
from datetime import datetime
import warnings
import logging
import os
import sys

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class TaxiModelTrainer:
    """Тренировщик модели такси"""

    def __init__(self):
        self.model = None
        self.feature_names = []

    def analyze_data(self, df):
        """Анализ данных перед обучением"""
        logger.info("🔍 Анализируем данные...")

        # Анализ целевой переменной
        target_ratio = df['is_done'].mean()
        logger.info(f"📊 Баланс классов: {target_ratio:.1%} positive")

        # Анализ корреляций
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_with_target = df[numeric_cols].corr()['is_done'].abs().sort_values(ascending=False)
            logger.info("📈 Корреляции с целевой переменной:")
            for feature, corr in corr_with_target.iloc[1:11].items():  # Топ-10 кроме самой target
                logger.info(f"  {feature}: {corr:.3f}")

        # Проверка на мультиколлинеарность
        if len(numeric_cols) > 2:
            corr_matrix = df[numeric_cols].corr().abs()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.8:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

            if high_corr_pairs:
                logger.warning("⚠️ Высокие корреляции между фичами:")
                for pair in high_corr_pairs[:5]:  # Покажем первые 5
                    logger.warning(f"  {pair[0]} - {pair[1]}: {pair[2]:.3f}")

    def load_and_prepare_data(self, data_path: str):
        """Загрузка и подготовка данных"""
        logger.info(f"📂 Загружаем данные из {data_path}")

        df = pd.read_csv(data_path)
        logger.info(f"📊 Исходные данные: {df.shape}")

        # Удаляем мусорные колонки
        for col in df.columns:
            if 'Unnamed' in col:
                df = df.drop(columns=[col])
                logger.info(f"✅ Удалили мусорную колонку {col}")

        # Преобразуем целевую переменную
        df['is_done'] = df['is_done'].map({'done': 1, 'cancel': 0})
        logger.info(f"📊 Распределение целевой переменной: {df['is_done'].value_counts().to_dict()}")

        # Анализ данных
        self.analyze_data(df)

        # Базовые фильтры
        initial_size = len(df)
        df = df[df['distance_in_meters'] > 0]
        df = df[df['duration_in_seconds'] > 0]
        df = df[df['price_start_local'] > 0]
        logger.info(f"✅ Базовая очистка: {initial_size} -> {len(df)}")

        # Создаем фичи
        df_processed = self.create_features(df)

        # УДАЛЯЕМ ID-КОЛОНКИ И ТЕКСТОВЫЕ КОЛОНКИ
        columns_to_remove = [
            'order_id', 'tender_id', 'driver_id', 'user_id',  # ID колонки
            'order_timestamp', 'tender_timestamp', 'driver_reg_date',  # Временные метки
            'carmodel', 'carname', 'platform'  # Текстовые колонки
        ]

        for col in columns_to_remove:
            if col in df_processed.columns:
                df_processed = df_processed.drop(columns=[col])
                logger.info(f"🗑️ Удалили колонку: {col}")

        # Убедимся что остались только числовые колонки
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns if col != 'is_done']

        X = df_processed[feature_columns]
        y = df_processed['is_done']

        # Удаляем пропуски
        mask = ~X.isna().any(axis=1)
        X_clean = X[mask]
        y_clean = y[mask]

        logger.info(f"✅ Финальные данные: {X_clean.shape[0]} samples, {X_clean.shape[1]} features")
        logger.info(f"📋 Используемые фичи: {list(X_clean.columns)}")

        return X_clean, y_clean

    def create_features(self, df):
        """УЛУЧШЕННОЕ создание фичей"""
        df = df.copy()

        # === ВРЕМЕННЫЕ ФИЧИ ===
        if 'order_timestamp' in df.columns:
            df['order_timestamp'] = pd.to_datetime(df['order_timestamp'])
            df['order_hour'] = df['order_timestamp'].dt.hour
            df['order_dayofweek'] = df['order_timestamp'].dt.dayofweek
            df['order_month'] = df['order_timestamp'].dt.month

            # Детальное время суток
            df['is_night'] = ((df['order_hour'] >= 23) | (df['order_hour'] <= 5)).astype(int)
            df['is_morning_rush'] = ((df['order_hour'] >= 7) & (df['order_hour'] <= 9)).astype(int)
            df['is_evening_rush'] = ((df['order_hour'] >= 17) & (df['order_hour'] <= 20)).astype(int)
            df['is_weekend'] = (df['order_dayofweek'] >= 5).astype(int)

            # Синус/косинус для циклических фичей
            df['hour_sin'] = np.sin(2 * np.pi * df['order_hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['order_hour'] / 24)
            df['dayofweek_sin'] = np.sin(2 * np.pi * df['order_dayofweek'] / 7)
            df['dayofweek_cos'] = np.cos(2 * np.pi * df['order_dayofweek'] / 7)

        # === ЦЕНОВЫЕ ФИЧИ ===
        if 'price_bid_local' in df.columns and 'price_start_local' in df.columns:
            df['price_increase_pct'] = ((df['price_bid_local'] - df['price_start_local']) / df[
                'price_start_local']) * 100
            df['price_per_km'] = df['price_bid_local'] / (df['distance_in_meters'] / 1000)
            df['price_start_per_km'] = df['price_start_local'] / (df['distance_in_meters'] / 1000)
            df['price_ratio'] = df['price_bid_local'] / df['price_start_local']

            # Ценовые категории
            df['price_bid_category'] = pd.cut(df['price_bid_local'], bins=10, labels=False)
            df['price_start_category'] = pd.cut(df['price_start_local'], bins=10, labels=False)

        # === ГЕО-ФИЧИ ===
        if 'distance_in_meters' in df.columns:
            df['distance_km'] = df['distance_in_meters'] / 1000
            df['log_distance_km'] = np.log1p(df['distance_km'])


        if 'pickup_in_meters' in df.columns:
            df['pickup_km'] = df['pickup_in_meters'] / 1000
            df['log_pickup_km'] = np.log1p(df['pickup_km'])

        # Скорость и эффективность
        if 'distance_km' in df.columns and 'duration_minutes' in df.columns:
            df['speed_kmh'] = (df['distance_km'] / (df['duration_minutes'] / 60)).replace([np.inf, -np.inf], 0)
            df['log_speed_kmh'] = np.log1p(df['speed_kmh'])

        if 'pickup_km' in df.columns and 'pickup_minutes' in df.columns:
            df['pickup_speed_kmh'] = (df['pickup_km'] / (df['pickup_minutes'] / 60)).replace([np.inf, -np.inf], 0)
            df['log_pickup_speed_kmh'] = np.log1p(df['pickup_speed_kmh'])

        # Общее время и эффективность


        # === ФИЧИ ВОДИТЕЛЯ ===
        if 'driver_rating' in df.columns:
            df['driver_rating_squared'] = df['driver_rating'] ** 2
            df['driver_rating_category'] = pd.cut(df['driver_rating'],
                                                  bins=[0, 3.5, 4.0, 4.5, 5.0],
                                                  labels=[0, 1, 2, 3])

        # Платформа (если нужна)
        if 'platform' in df.columns:
            platform_dummies = pd.get_dummies(df['platform'], prefix='platform')
            df = pd.concat([df, platform_dummies], axis=1)

        # === ВЗАИМОДЕЙСТВИЯ ===
        if 'price_per_km' in df.columns and 'distance_km' in df.columns:
            df['price_distance_interaction'] = df['price_per_km'] * df['distance_km']

        if 'price_increase_pct' in df.columns and 'driver_experience_days' in df.columns:
            df['price_exp_interaction'] = df['price_increase_pct'] * df['driver_experience_days']

        if 'is_night' in df.columns and 'is_weekend' in df.columns:
            df['night_weekend_interaction'] = df['is_night'] * df['is_weekend']

        if 'driver_rating' in df.columns and 'price_increase_pct' in df.columns:
            df['rating_price_interaction'] = df['driver_rating'] * df['price_increase_pct']

        # === СТАТИСТИЧЕСКИЕ ФИЧИ ===
        # Отношения
        if 'pickup_km' in df.columns and 'distance_km' in df.columns:
            df['pickup_distance_ratio'] = df['pickup_km'] / (df['distance_km'] + 1e-8)

        if 'pickup_minutes' in df.columns and 'duration_minutes' in df.columns:
            df['pickup_duration_ratio'] = df['pickup_minutes'] / (df['duration_minutes'] + 1e-8)

        # Бинарные фичи
        if 'distance_km' in df.columns:
            df['is_short_trip'] = (df['distance_km'] < 3).astype(int)
            df['is_long_trip'] = (df['distance_km'] > 15).astype(int)

        if 'duration_minutes' in df.columns:
            df['is_quick_trip'] = (df['duration_minutes'] < 10).astype(int)
            df['is_long_duration'] = (df['duration_minutes'] > 30).astype(int)

        return df

    def get_feature_columns(self):
        """Список фичей для модели"""
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
            'driver_rating', 'driver_experience_days', 'is_ios'
        ]
        return features

    def try_alternative_models(self, X, y):
        """Пробуем другие алгоритмы если LightGBM плохо работает"""
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }

        for name, model in models.items():
            try:
                scores = cross_val_score(model, X, y, cv=3, scoring='roc_auc', n_jobs=-1)
                logger.info(f"🔍 {name} - Mean AUC: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            except Exception as e:
                logger.warning(f"⚠️ {name} не сработал: {e}")

    def train_model(self, X, y, test_size: float = 0.2):
        """УЛУЧШЕННОЕ обучение модели"""
        import warnings
        from lightgbm import LGBMClassifier, log_evaluation, early_stopping

        # Подавляем предупреждения
        warnings.filterwarnings('ignore', category=UserWarning)

        logger.info("🎯 Начинаем УЛУЧШЕННОЕ обучение...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        logger.info(f"📊 Обучающая выборка: {X_train.shape}")
        logger.info(f"📊 Тестовая выборка: {X_test.shape}")

        # Сохраняем имена фичей
        self.feature_names = X_train.columns.tolist()

        # ОПТИМИЗИРОВАННЫЕ НАСТРОЙКИ
        self.model = LGBMClassifier(
            n_estimators=20000,  # Больше деревьев
            learning_rate=0.005,  # Очень маленький learning rate
            max_depth=8,  # Оптимальная глубина
            num_leaves=63,  # Оптимальное количество листьев
            min_child_samples=30,  # Защита от переобучения
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.01,  # Слабая регуляризация
            reg_lambda=0.01,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            metric='auc',
            boosting_type='gbdt',
            importance_type='gain'  # Более информативная важность
        )

        logger.info("🔥 Запускаем оптимизированное обучение...")

        try:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric='auc',
                callbacks=[
                    early_stopping(stopping_rounds=200, verbose=False),  # Больше терпения
                    log_evaluation(500)  # Логируем реже
                ]
            )
        except Exception as e:
            logger.warning(f"⚠️ Проблема с обучением: {e}")
            logger.info("🔄 Используем простое обучение...")
            self.model.fit(X_train, y_train)

        # Оценка качества
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)

        logger.info(f"📈 ROC-AUC на тесте: {auc_score:.4f}")

        # Детальный анализ
        self._detailed_analysis(X_test, y_test, y_pred_proba)

        # Важность фичей
        self._log_feature_importance()

        return auc_score

    def _detailed_analysis(self, X_test, y_test, y_pred_proba):
        """Детальный анализ результатов"""
        from sklearn.metrics import precision_recall_curve, average_precision_score

        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)

        logger.info(f"📊 Precision-Recall AUC: {pr_auc:.4f}")

        # Найдем оптимальный threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = recall[best_idx]

        logger.info(f"🎯 Оптимальный threshold: {best_threshold:.3f}")
        logger.info(f"📈 Лучшая F1-score: {f1_scores[best_idx]:.3f}")
    def _log_feature_importance(self):
        """Логирование важности фичей"""
        importance_data = list(zip(self.feature_names, self.model.feature_importances_))
        importance_data.sort(key=lambda x: x[1], reverse=True)

        logger.info("🔝 Топ-10 самых важных фичей:")
        for feature, importance in importance_data[:10]:
            logger.info(f"  {feature}: {importance}")

    def save_model(self, output_path: str = 'models/taxi_model.joblib'):
        """Сохранение модели"""
        os.makedirs('models', exist_ok=True)

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'training_date': datetime.now(),
            'model_type': 'LightGBM'
        }

        joblib.dump(model_data, output_path)
        logger.info(f"💾 Модель сохранена: {output_path}")


def main():
    """Основная функция"""
    trainer = TaxiModelTrainer()

    try:
        if not os.path.exists('train.csv'):
            logger.error("❌ Файл train.csv не найден!")
            return

        # Загрузка с улучшенной подготовкой
        X, y = trainer.load_and_prepare_data('train.csv')

        if len(X) == 0:
            logger.error("❌ Нет данных для обучения!")
            return

        logger.info("🚀 ЗАПУСКАЕМ УЛУЧШЕННУЮ ВЕРСИЮ!")
        auc = trainer.train_model(X, y)

        trainer.save_model()

        if auc > 0.75:
            logger.info(f"🎉 ОТЛИЧНО! AUC: {auc:.4f}")
        elif auc > 0.72:
            logger.info(f"✅ Нормально! AUC: {auc:.4f}")
        else:
            logger.warning(f"⚠️ Слабовато! AUC: {auc:.4f}")
            logger.info("🔄 Пробуем альтернативные модели...")
            trainer.try_alternative_models(X, y)

    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        raise


if __name__ == "__main__":
    main()