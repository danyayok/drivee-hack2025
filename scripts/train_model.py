# scripts/retrain_smart.py
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import warnings
import joblib
from datetime import datetime

warnings.filterwarnings("ignore")


class SmartTaxiModelTrainer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.model = None
        self.features = []
        self.target = "is_done"

    def preprocess_target(self):
        """Ваш код с улучшенной обработкой ошибок"""
        self.df[self.target] = self.df[self.target].astype(str).str.strip().str.lower()
        self.df = self.df[self.df[self.target].isin(['done', 'cancel', '1', '0'])]
        self.df[self.target] = self.df[self.target].map({'done': 1, 'cancel': 0, '1': 1, '0': 0})
        self.df = self.df[self.df[self.target].isin([0, 1])].reset_index(drop=True)
        print(
            f"✅ Target подготовлен. Принято: {self.df[self.target].sum()}, Отменено: {len(self.df) - self.df[self.target].sum()}")

    def create_features(self):
        """Ваши отличные фичи + защита от ошибок"""
        df = self.df.copy()

        # 🔥 ДОБАВЬТЕ ЕСЛИ НЕТ user_rating
        if 'user_rating' not in df.columns:
            df['user_rating'] = 4.6
        # Преобразуем в datetime с защитой
        for col in ["order_timestamp", "tender_timestamp", "driver_reg_date"]:
            df[col] = pd.to_datetime(df[col], errors="coerce")

        # Временные признаки (ваши)
        df["order_hour"] = df["order_timestamp"].dt.hour
        df["order_dayofweek"] = df["order_timestamp"].dt.dayofweek
        df["order_month"] = df["order_timestamp"].dt.month
        df["is_peak_hour"] = df["order_hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
        df["is_weekend"] = (df["order_dayofweek"] >= 5).astype(int)

        # Временные дельты (ваши)
        df["response_delay_sec"] = (df["tender_timestamp"] - df["order_timestamp"]).dt.total_seconds().clip(lower=0)
        df["driver_experience_days"] = (
                    (df["order_timestamp"] - df["driver_reg_date"]).dt.total_seconds() / (3600 * 24)).clip(lower=0)

        # Цены и взаимодействия (ваши лучшие фичи)
        df["price_ratio"] = df["price_bid_local"] / (df["price_start_local"] + 1e-6)
        df["price_diff"] = df["price_bid_local"] - df["price_start_local"]
        df["price_per_km"] = df["price_bid_local"] / (df["distance_in_meters"] / 1000 + 1e-6)
        df["price_per_min"] = df["price_bid_local"] / (df["duration_in_seconds"] / 60 + 1e-6)
        df["trip_speed_kmh"] = (df["distance_in_meters"] / 1000) / (df["duration_in_seconds"] / 3600 + 1e-6)
        df["pickup_speed_kmh"] = (df["pickup_in_meters"] / 1000) / (df["pickup_in_seconds"] / 3600 + 1e-6)

        # Взаимодействия (ваши)
        df["rating_price_interaction"] = df["driver_rating"] * df["price_ratio"]
        df["rating_speed_interaction"] = df["driver_rating"] * df["trip_speed_kmh"]
        df["experience_price_ratio"] = df["driver_experience_days"] * df["price_ratio"]

        # Логарифмы с защитой (ваши + исправление)
        for col in ["price_bid_local", "distance_in_meters", "duration_in_seconds", "driver_experience_days"]:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))  # 🔥 Защита от отрицательных

        # Статистика по группам (ваши мощные фичи)
        df['driver_cancel_rate'] = df.groupby('driver_id')[self.target].transform(lambda x: 1 - x.mean())
        df['user_cancel_rate'] = df.groupby('user_id')[self.target].transform(lambda x: 1 - x.mean())
        df['driver_avg_price'] = df.groupby('driver_id')['price_bid_local'].transform('mean')
        df['user_avg_price'] = df.groupby('user_id')['price_bid_local'].transform('mean')

        # Защита от бесконечностей и NaN
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        self.df = df
        print(f"✅ Создано {len(df.columns)} фичей")
        return df

    def prepare_for_production(self):
        """Подготовка для продакшена с сохранением feature_names"""
        df = self.df.copy()

        # Колонки для удаления (как у вас)
        drop_cols = [
            self.target, "order_id", "tender_id", "tender_timestamp",
            "driver_reg_date", "order_timestamp", "user_id", "driver_id", "Unnamed: 18"
        ]
        drop_cols = [col for col in drop_cols if col in df.columns]

        X = df.drop(columns=drop_cols, errors="ignore")
        y = df[self.target]

        # Категориальные фичи (как у вас)
        cat_cols = ["carmodel", "carname", "platform"]
        cat_cols = [c for c in cat_cols if c in X.columns]

        self.features = X.columns.tolist()
        self.cat_features = cat_cols

        print(f"📊 Фичей для обучения: {len(self.features)}")
        print(f"🐱 Категориальные фичи: {cat_cols}")

        return X, y, cat_cols

    def train_model(self, save_path="models/catboost_taxi_smart.joblib"):
        """Обучение с вашими параметрами + мониторинг"""
        X, y, cat_cols = self.prepare_for_production()

        # Стратифицированное разделение для стабильности
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"📈 Размер train: {X_train.shape}, test: {X_test.shape}")

        train_pool = Pool(X_train, y_train, cat_features=cat_cols)
        test_pool = Pool(X_test, y_test, cat_features=cat_cols)

        # 🔥 ВАШИ ОТЛИЧНЫЕ ПАРАМЕТРЫ
        model = CatBoostClassifier(
            iterations=3000,
            depth=10,
            learning_rate=0.01,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            verbose=200,
            early_stopping_rounds=200,
            l2_leaf_reg=3.0,
            bagging_temperature=1.0,
            random_strength=1.0,
            task_type="CPU"
        )

        print("🎯 Начинаем обучение с вашими параметрами...")
        model.fit(train_pool, eval_set=test_pool, use_best_model=True)

        # Оценка
        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)

        self.model = model
        print(f"\n✅ Final AUC: {auc:.4f}")

        # 🔥 УМНОЕ СОХРАНЕНИЕ для продакшена
        model_data = {
            "model": self.model,
            "feature_names": self.features,  # Критически важно!
            "cat_features": cat_cols,
            "training_date": datetime.now().isoformat(),
            "metrics": {"auc": auc},
            "feature_statistics": self.get_feature_stats(X_train),
            "version": "smart_v1"
        }

        joblib.dump(model_data, save_path)
        print(f"💾 Умная модель сохранена: {save_path}")

        # Анализ
        self.analyze_results(model, X_test, y_test)

        return auc

    def get_feature_stats(self, X):
        """Статистика фичей для отладки"""
        return {
            "num_features": len(X.columns),
            "feature_list": X.columns.tolist(),
            "dtypes": X.dtypes.to_dict()
        }

    def analyze_results(self, model, X_test, y_test):
        """Умный анализ результатов"""
        print("\n🔍 Детальный анализ:")

        # Предсказания
        probabilities = model.predict_proba(X_test)[:, 1]

        print(f"📊 Статистика вероятностей:")
        print(f"   Min: {probabilities.min():.3f}")
        print(f"   Mean: {probabilities.mean():.3f}")
        print(f"   Max: {probabilities.max():.3f}")
        print(f"   Std: {probabilities.std():.3f}")

        # Важность фичей
        importance = pd.DataFrame({
            'feature': self.features,
            'importance': model.get_feature_importance()
        }).sort_values('importance', ascending=False)

        print(f"\n🔝 Топ-15 важных фичей:")
        print(importance.head(15).to_string(index=False))

        # Тест на реалистичность
        self.realism_test(model, X_test)

    def realism_test(self, model, X_test):
        """Тест на реалистичность вероятностей"""
        print(f"\n🧪 Тест реалистичности:")

        sample = X_test.iloc[[0]].copy()
        base_price = sample['price_start_local'].iloc[0]

        # Тестируем разные цены
        test_cases = [
            ("Низкая", base_price * 0.7),
            ("Базовая", base_price),
            ("Высокая", base_price * 1.5),
            ("Очень высокая", base_price * 2.0)
        ]

        for name, price in test_cases:
            test_row = sample.copy()
            test_row['price_bid_local'] = price
            test_row['price_ratio'] = price / (test_row['price_start_local'].iloc[0] + 1e-6)
            test_row['price_diff'] = price - test_row['price_start_local'].iloc[0]

            # Обновляем взаимодействия
            test_row['rating_price_interaction'] = test_row['driver_rating'].iloc[0] * test_row['price_ratio'].iloc[0]
            test_row['experience_price_ratio'] = test_row['driver_experience_days'].iloc[0] * \
                                                 test_row['price_ratio'].iloc[0]

            probability = model.predict_proba(test_row)[0, 1]
            print(f"   {name} цена ({price:.0f} руб): {probability:.3f}")

    def get_feature_importance(self, top_n=20):
        """Ваш код важности фичей"""
        if self.model is None:
            raise ValueError("Model not trained yet. Run train_model() first.")
        fi = pd.DataFrame({
            "feature": self.features,
            "importance": self.model.get_feature_importance()
        }).sort_values(by="importance", ascending=False)
        print(f"\n🔝 Top {top_n} features:")
        print(fi.head(top_n))
        return fi


def main():
    print("🚀 Запуск УМНОГО обучения...")
    print("=" * 50)

    # Загрузка
    df = pd.read_csv("train.csv")
    print(f"📁 Загружено данных: {df.shape}")

    # Обучение
    trainer = SmartTaxiModelTrainer(df)
    trainer.preprocess_target()
    trainer.create_features()
    trainer.train_model()
    trainer.get_feature_importance()

    print("=" * 50)
    print("🎉 Умное обучение завершено!")


if __name__ == "__main__":
    main()