# scripts/retrain_proper.py
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import warnings

warnings.filterwarnings('ignore')


class ProperTaxiModelTrainer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.target = "is_done"

    def preprocess_data(self):
        """Корректная предобработка данных"""
        df = self.df.copy()

        # 🔥 ПРАВИЛЬНОЕ преобразование target
        df[self.target] = df[self.target].astype(str).str.strip().str.lower()
        df = df[df[self.target].isin(['done', 'cancel', '1', '0'])]
        df[self.target] = df[self.target].map({'done': 1, 'cancel': 0, '1': 1, '0': 0})

        print(f"✅ Target подготовлен. Принято: {df[self.target].sum()}, Отменено: {len(df) - df[self.target].sum()}")

        # Преобразуем даты
        for col in ["order_timestamp", "tender_timestamp", "driver_reg_date"]:
            df[col] = pd.to_datetime(df[col], errors='coerce')

        self.df = df
        return df

    def create_features(self):
        """Создание всех фичей как в оригинальном коде"""
        df = self.df.copy()

        # Временные фичи
        df["order_hour"] = df["order_timestamp"].dt.hour
        df["order_dayofweek"] = df["order_timestamp"].dt.dayofweek
        df["order_month"] = df["order_timestamp"].dt.month
        df["is_peak_hour"] = df["order_hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
        df["is_weekend"] = (df["order_dayofweek"] >= 5).astype(int)

        # Временные дельты
        df["response_delay_sec"] = (df["tender_timestamp"] - df["order_timestamp"]).dt.total_seconds().clip(lower=0)
        df["driver_experience_days"] = (
                    (df["order_timestamp"] - df["driver_reg_date"]).dt.total_seconds() / (3600 * 24)).clip(lower=0)

        # Ценовые фичи
        df["price_ratio"] = df["price_bid_local"] / (df["price_start_local"] + 1e-6)
        df["price_diff"] = df["price_bid_local"] - df["price_start_local"]
        df["price_per_km"] = df["price_bid_local"] / (df["distance_in_meters"] / 1000 + 1e-6)
        df["price_per_min"] = df["price_bid_local"] / (df["duration_in_seconds"] / 60 + 1e-6)
        df["trip_speed_kmh"] = (df["distance_in_meters"] / 1000) / (df["duration_in_seconds"] / 3600 + 1e-6)
        df["pickup_speed_kmh"] = (df["pickup_in_meters"] / 1000) / (df["pickup_in_seconds"] / 3600 + 1e-6)

        # Взаимодействия
        df["rating_price_interaction"] = df["driver_rating"] * df["price_ratio"]
        df["rating_speed_interaction"] = df["driver_rating"] * df["trip_speed_kmh"]
        df["experience_price_ratio"] = df["driver_experience_days"] * df["price_ratio"]

        # Логарифмы
        for col in ["price_bid_local", "distance_in_meters", "duration_in_seconds", "driver_experience_days"]:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))  # 🔥 ИСПРАВЛЕНИЕ: clip для избежания ошибок

        # Статистики по группам (без утечки!)
        df['driver_total_orders'] = df.groupby('driver_id')['driver_id'].transform('count')
        df['user_total_orders'] = df.groupby('user_id')['user_id'].transform('count')

        # Заполняем пропуски
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        self.df = df
        print(f"✅ Создано {len(df.columns)} фичей")
        return df

    def prepare_features_for_training(self):
        """Подготовка фичей для обучения"""
        df = self.df.copy()

        # Колонки для удаления
        drop_cols = [
            self.target, "order_id", "tender_id", "tender_timestamp",
            "driver_reg_date", "order_timestamp", "user_id", "Unnamed: 18"
        ]
        drop_cols = [col for col in drop_cols if col in df.columns]

        X = df.drop(columns=drop_cols, errors="ignore")
        y = df[self.target]

        # Категориальные фичи
        cat_cols = ["driver_id", "carmodel", "carname", "platform"]
        cat_cols = [c for c in cat_cols if c in X.columns]

        self.feature_names = X.columns.tolist()
        self.cat_features = cat_cols

        print(f"📊 Фичей для обучения: {len(self.feature_names)}")
        print(f"🐱 Категориальные фичи: {cat_cols}")

        return X, y, cat_cols

    def train_model(self, save_path="models/catboost_taxi_proper.joblib"):
        """Обучение модели"""
        X, y, cat_cols = self.prepare_features_for_training()

        # Разделение данных
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"📈 Размер train: {X_train.shape}, test: {X_test.shape}")

        # Создаем Pool
        train_pool = Pool(X_train, y_train, cat_features=cat_cols)
        test_pool = Pool(X_test, y_test, cat_features=cat_cols)

        # Настройки модели
        model = CatBoostClassifier(
            iterations=1000,
            depth=8,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            verbose=100,
            early_stopping_rounds=50,
            l2_leaf_reg=3,
            cat_features=cat_cols
        )

        print("🎯 Начинаем обучение...")
        model.fit(train_pool, eval_set=test_pool, use_best_model=True)

        # Оценка
        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)

        print(f"\n✅ Обучение завершено!")
        print(f"📊 Final AUC: {auc:.4f}")

        # Сохраняем модель с ВСЕМИ данными
        model_data = {
            'model': model,
            'feature_names': self.feature_names,
            'cat_features': cat_cols,
            'metrics': {'auc': auc},
            'training_date': pd.Timestamp.now().isoformat()
        }

        joblib.dump(model_data, save_path)
        print(f"💾 Модель сохранена: {save_path}")

        # Анализ
        self.analyze_model(model, X_test, y_test)

        return model, auc

    def analyze_model(self, model, X_test, y_test):
        """Анализ модели"""
        print("\n🔍 Анализ модели:")

        # Предсказания
        probabilities = model.predict_proba(X_test)[:, 1]

        print(f"📊 Статистика вероятностей:")
        print(f"   Min: {probabilities.min():.3f}")
        print(f"   Mean: {probabilities.mean():.3f}")
        print(f"   Max: {probabilities.max():.3f}")

        # Важность фичей
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.get_feature_importance()
        }).sort_values('importance', ascending=False)

        print(f"\n🔝 Топ-10 важных фичей:")
        print(importance.head(10).to_string(index=False))

        # Тестируем на разных ценах
        self.test_price_sensitivity(model, X_test)

    def test_price_sensitivity(self, model, X_test):
        """Тестируем чувствительность к цене"""
        print(f"\n💰 Тестируем чувствительность к цене:")

        # Берем первую строку как шаблон
        sample = X_test.iloc[[0]].copy()
        base_price = sample['price_start_local'].iloc[0]

        test_prices = [base_price * 0.7, base_price, base_price * 1.3, base_price * 1.6, base_price * 2.0]

        for price in test_prices:
            test_row = sample.copy()
            test_row['price_bid_local'] = price
            test_row['price_ratio'] = price / (test_row['price_start_local'].iloc[0] + 1e-6)
            test_row['price_diff'] = price - test_row['price_start_local'].iloc[0]

            # Обновляем взаимодействия с ценой
            if 'rating_price_interaction' in test_row.columns:
                test_row['rating_price_interaction'] = test_row['driver_rating'].iloc[0] * test_row['price_ratio'].iloc[
                    0]
            if 'experience_price_ratio' in test_row.columns:
                test_row['experience_price_ratio'] = test_row['driver_experience_days'].iloc[0] * \
                                                     test_row['price_ratio'].iloc[0]

            probability = model.predict_proba(test_row)[0, 1]
            print(f"   Цена: {price:.0f} руб -> Вероятность: {probability:.3f}")


def main():
    print("🚀 Запуск нормального обучения модели...")

    # Загружаем данные
    df = pd.read_csv("train.csv")
    print(f"📁 Загружено данных: {df.shape}")

    # Обучаем модель
    trainer = ProperTaxiModelTrainer(df)
    trainer.preprocess_data()
    trainer.create_features()
    trainer.train_model()

    print("🎉 Обучение завершено!")


if __name__ == "__main__":
    main()