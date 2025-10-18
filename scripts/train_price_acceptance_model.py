import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import warnings
import joblib
from datetime import datetime
import os

warnings.filterwarnings("ignore")


class PriceAcceptanceModelTrainer:
    """
    Модель для предсказания вероятности принятия ЛЮБОЙ цены
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.model = None
        self.feature_names = []
        self.target = "is_done"

    def preprocess_target(self):
        """Преобразуем целевую переменную"""
        self.df[self.target] = self.df[self.target].astype(str).str.lower().map({"done": 1, "cancel": 0})
        self.df = self.df[self.df[self.target].isin([0, 1])].reset_index(drop=True)
        print(f"✅ Target подготовлен. Баланс классов: {self.df[self.target].value_counts().to_dict()}")

    def create_price_features(self, df):
        """
        Создаем фичи, которые зависят от price_bid_local
        Эти фичи будут пересчитываться для каждой тестовой цены
        """
        # Базовые ценовые фичи (будут меняться для каждой тестовой цены)
        df["price_diff"] = df["price_bid_local"] - df["price_start_local"]
        df["price_ratio"] = df["price_bid_local"] / (df["price_start_local"] + 1e-6)
        df["price_increase_pct"] = ((df["price_bid_local"] - df["price_start_local"]) /
                                    (df["price_start_local"] + 1e-6)) * 100

        df["price_per_km"] = df["price_bid_local"] / (df["distance_in_meters"] / 1000 + 1e-6)
        df["price_per_min"] = df["price_bid_local"] / (df["duration_in_seconds"] / 60 + 1e-6)

        # Взаимодействия с рейтингом (будут меняться)
        df["rating_price_interaction"] = df["driver_rating"] * df["price_ratio"]
        df["experience_price_ratio"] = df["driver_experience_days"] * df["price_ratio"]

        return df

    def create_static_features(self, df):
        """
        Создаем статические фичи, которые не зависят от цены
        Эти фичи остаются постоянными для одного заказа
        """
        # Временные фичи
        df["order_timestamp"] = pd.to_datetime(df["order_timestamp"], errors="coerce")
        df["order_hour"] = df["order_timestamp"].dt.hour
        df["order_dayofweek"] = df["order_timestamp"].dt.dayofweek
        df["is_peak_hour"] = df["order_hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
        df["is_weekend"] = (df["order_dayofweek"] >= 5).astype(int)
        df["is_night"] = ((df["order_hour"] >= 23) | (df["order_hour"] <= 5)).astype(int)

        # Гео фичи
        df["distance_km"] = df["distance_in_meters"] / 1000
        df["duration_minutes"] = df["duration_in_seconds"] / 60
        df["pickup_km"] = df["pickup_in_meters"] / 1000
        df["pickup_minutes"] = df["pickup_in_seconds"] / 60

        # Скорости
        df["trip_speed_kmh"] = (df["distance_km"]) / (df["duration_minutes"] / 60 + 1e-6)
        df["pickup_speed_kmh"] = (df["pickup_km"]) / (df["pickup_minutes"] / 60 + 1e-6)

        # Опыт водителя
        df["driver_reg_date"] = pd.to_datetime(df["driver_reg_date"], errors="coerce")
        df["driver_experience_days"] = (
                (df["order_timestamp"] - df["driver_reg_date"]).dt.total_seconds() / (3600 * 24)).clip(lower=0)

        # Логарифмические преобразования
        for col in ["distance_in_meters", "duration_in_seconds", "driver_experience_days"]:
            df[f"log_{col}"] = np.log1p(df[col])

        # Статистики по водителю и пользователю
        if len(df) > 1 and 'driver_id' in df.columns and 'user_id' in df.columns:
            # В обучающих данных - считаем реальные статистики
            df['driver_acceptance_rate'] = df.groupby('driver_id')[self.target].transform('mean')
            df['user_acceptance_rate'] = df.groupby('user_id')[self.target].transform('mean')
        else:
            # В тестовых данных - используем общее среднее
            overall_rate = self.df[self.target].mean() if hasattr(self, 'df') else 0.35
            df['driver_acceptance_rate'] = overall_rate
            df['user_acceptance_rate'] = overall_rate

        return df

    def prepare_categorical_features(self, df):
        """ВАЖНО: Преобразуем категориальные фичи в строки"""
        categorical_cols = ['platform', 'carmodel', 'carname']
        for col in categorical_cols:
            if col in df.columns:
                # Преобразуем ВСЕ значения в строки и заменяем NaN
                df[col] = df[col].fillna('unknown').astype(str)
                # Если значение числовое, добавляем префикс
                df[col] = df[col].apply(lambda x: f"cat_{x}" if str(x).replace('.', '').isdigit() else x)
        return df

    def prepare_features(self):
        """Подготовка всех фичей"""
        print("🔄 Создаем фичи...")

        # Сначала статические фичи
        df = self.create_static_features(self.df)

        # Затем ценовые фичи (которые зависят от price_bid_local)
        df = self.create_price_features(df)

        # ОБЯЗАТЕЛЬНО: преобразуем категориальные фичи
        df = self.prepare_categorical_features(df)

        # Заполняем пропуски
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        self.df = df
        return df

    def get_feature_columns(self):
        """Определяем фичи для модели"""
        # Статические фичи (не зависят от цены)
        static_features = [
            'distance_km', 'duration_minutes', 'pickup_km', 'pickup_minutes',
            'order_hour', 'order_dayofweek', 'is_peak_hour', 'is_weekend', 'is_night',
            'driver_rating', 'user_rating',
            'trip_speed_kmh', 'pickup_speed_kmh',
            'driver_experience_days', 'log_distance_in_meters',
            'log_duration_in_seconds', 'log_driver_experience_days',
            'driver_acceptance_rate', 'user_acceptance_rate'
        ]

        # Динамические фичи (зависят от price_bid_local)
        dynamic_features = [
            'price_bid_local', 'price_diff', 'price_ratio', 'price_increase_pct',
            'price_per_km', 'price_per_min',
            'rating_price_interaction', 'experience_price_ratio'
        ]

        # Категориальные фичи
        categorical_features = ['platform', 'carmodel', 'carname']

        all_features = static_features + dynamic_features + categorical_features
        # Оставляем только те, что есть в данных
        available_features = [f for f in all_features if f in self.df.columns]

        print(f"📊 Всего фичей: {len(available_features)}")
        print(f"  - Статические: {len([f for f in available_features if f in static_features])}")
        print(f"  - Динамические: {len([f for f in available_features if f in dynamic_features])}")
        print(f"  - Категориальные: {len([f for f in available_features if f in categorical_features])}")

        return available_features, categorical_features

    def train_model(self, test_size=0.2, random_state=42, save_path="models/price_acceptance_model.joblib"):
        """Обучение модели"""
        print("🚀 Начинаем обучение модели...")

        # Подготовка данных
        df = self.prepare_features()
        features, cat_features = self.get_feature_columns()

        X = df[features]
        y = df[self.target]

        # Убедимся, что категориальные фичи есть в данных
        cat_features = [f for f in cat_features if f in X.columns]

        print(f"🎯 Размер данных: {X.shape}")
        print(f"📈 Баланс целевой переменной: {y.value_counts().to_dict()}")
        print(f"🐱 Категориальные фичи: {cat_features}")

        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Создаем пулы для CatBoost
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        test_pool = Pool(X_test, y_test, cat_features=cat_features)

        # Настройки модели
        model = CatBoostClassifier(
            iterations=1500,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=random_state,
            verbose=100,
            early_stopping_rounds=100,
            l2_leaf_reg=5.0,
            border_count=128
        )

        # Обучение
        print("📚 Обучаем модель...")
        model.fit(train_pool, eval_set=test_pool, use_best_model=True)

        # Предсказания и метрики
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        auc = roc_auc_score(y_test, y_pred_proba)

        print(f"\n✅ Final AUC: {auc:.4f}")
        print("📊 Classification Report:")
        print(classification_report(y_test, y_pred))

        self.model = model
        self.feature_names = features
        self.cat_features = cat_features  # Сохраняем для тестирования

        # Сохраняем модель
        model_data = {
            "model": model,
            "feature_names": features,
            "categorical_features": cat_features,
            "model_type": "price_acceptance",
            "training_date": datetime.now().isoformat(),
            "auc_score": auc
        }

        # Создаем директорию если не существует
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model_data, save_path)
        print(f"💾 Модель сохранена: {save_path}")

        return auc

    def test_price_sensitivity(self, sample_data, price_range):
        """Тестируем чувствительность модели к цене"""
        if self.model is None:
            raise ValueError("Модель не обучена!")

        print("\n🔍 Тестируем чувствительность к цене...")

        # Сначала создаем все статические фичи для тестовых данных
        sample_df = pd.DataFrame([sample_data])
        sample_df = self.create_static_features(sample_df)
        sample_df = self.prepare_categorical_features(sample_df)  # ВАЖНО: преобразуем категориальные фичи

        results = []
        for price in price_range:
            test_data = sample_df.iloc[0].to_dict().copy()
            test_data['price_bid_local'] = price

            # Пересчитываем только динамические фичи
            test_data_df = pd.DataFrame([test_data])
            test_data_df = self.create_price_features(test_data_df)
            test_data_df = self.prepare_categorical_features(test_data_df)  # ВАЖНО: снова преобразуем

            # Убедимся, что все фичи присутствуют
            for feature in self.feature_names:
                if feature not in test_data_df.columns:
                    if feature in self.cat_features:
                        test_data_df[feature] = 'unknown'  # Для категориальных - строка
                    else:
                        test_data_df[feature] = 0.0

            # Подготавливаем фичи в правильном порядке
            features_df = test_data_df[self.feature_names]

            # ВАЖНО: Создаем Pool с указанием категориальных фичей
            prediction_pool = Pool(features_df, cat_features=self.cat_features)
            probability = self.model.predict_proba(prediction_pool)[0, 1]

            results.append((price, probability))

            print(f"  💰 {price:.1f} руб → {probability:.3f} вероятности")

        return results

    def get_feature_importance(self, top_n=20):
        """Важность фич"""
        if self.model is None:
            raise ValueError("Model not trained yet. Run train_model() first.")

        fi = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.get_feature_importance()
        }).sort_values(by="importance", ascending=False)

        print("\n🔝 Top features:")
        print(fi.head(top_n).to_string(index=False))
        return fi

    def calculate_acceptance_rates(self):
        """Рассчитываем acceptance rates из обучающих данных"""
        print("📊 Расчет acceptance rates...")

        driver_acceptance = self.df.groupby('driver_id')[self.target].mean().to_dict()
        user_acceptance = self.df.groupby('user_id')[self.target].mean().to_dict()

        overall_driver_rate = self.df[self.target].mean()
        overall_user_rate = self.df[self.target].mean()

        print(f"✅ Рассчитано rates для {len(driver_acceptance)} водителей и {len(user_acceptance)} пользователей")
        print(f"📈 Общий acceptance rate: {overall_driver_rate:.3f}")

        # Сохраняем в файл
        acceptance_data = {
            'driver_acceptance': driver_acceptance,
            'user_acceptance': user_acceptance,
            'overall_driver_rate': overall_driver_rate,
            'overall_user_rate': overall_user_rate,
            'calculated_at': datetime.now().isoformat()
        }

        os.makedirs("models", exist_ok=True)
        joblib.dump(acceptance_data, "models/acceptance_rates.joblib")
        print("💾 Acceptance rates сохранены в models/acceptance_rates.joblib")

        return acceptance_data


def simple_test():
    """Простой тест модели"""
    print("\n🧪 ПРОСТОЙ ТЕСТ МОДЕЛИ")
    print("=" * 40)

    try:
        # Загружаем модель
        model_data = joblib.load("models/price_acceptance_model.joblib")
        model = model_data['model']
        feature_names = model_data['feature_names']
        cat_features = model_data['categorical_features']

        print("✅ Модель загружена!")

        # Простой тест
        test_data = {
            "distance_in_meters": 3404.0,
            "duration_in_seconds": 486.0,
            "pickup_in_meters": 790.0,
            "pickup_in_seconds": 169.0,
            "driver_rating": 5.0,
            "user_rating": 4.8,
            "price_start_local": 180.0,
            "price_bid_local": 200.0,
            "order_timestamp": "2020-05-01T00:05:14",
            "driver_platform": "Android",
            "driver_reg_date": "2019-09-22",
            "carmodel": "Logan",
            "carname": "Renault"
        }

        # Создаем DataFrame
        df = pd.DataFrame([test_data])

        # Простые фичи (только основные)
        df["distance_km"] = df["distance_in_meters"] / 1000
        df["duration_minutes"] = df["duration_in_seconds"] / 60
        df["price_diff"] = df["price_bid_local"] - df["price_start_local"]
        df["price_ratio"] = df["price_bid_local"] / df["price_start_local"]

        # Заполняем остальные фичи нулями
        for feature in feature_names:
            if feature not in df.columns:
                if feature in cat_features:
                    df[feature] = 'unknown'
                else:
                    df[feature] = 0.0

        # Предсказание
        features_df = df[feature_names]
        pool = Pool(features_df, cat_features=cat_features)
        probability = model.predict_proba(pool)[0, 1]

        print(f"🎯 Тестовое предсказание:")
        print(f"   Цена: {test_data['price_bid_local']} руб")
        print(f"   Вероятность принятия: {probability:.3f}")
        print(f"   Ожидаемый доход: {test_data['price_bid_local'] * probability:.1f} руб")

    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")


if __name__ == "__main__":
    # Загрузка данных
    print("📥 Загружаем данные...")
    df = pd.read_csv("train.csv")

    # Создаем и обучаем модель
    trainer = PriceAcceptanceModelTrainer(df)
    trainer.preprocess_target()

    # Рассчитываем acceptance rates
    acceptance_data = trainer.calculate_acceptance_rates()

    # Обучаем модель
    auc = trainer.train_model(save_path="models/price_acceptance_model.joblib")
    trainer.get_feature_importance()

    # Простой тест вместо сложного
    simple_test()

    print("\n" + "=" * 50)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО! МОДЕЛЬ ГОТОВА К ИСПОЛЬЗОВАНИЮ!")
    print("=" * 50)
    print("🎯 Дальнейшие действия:")
    print("   1. python scripts/rename_model.py")
    print("   2. python run_tests.py")
    print("   3. python run.py")