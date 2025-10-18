import requests
import json


def debug_api_features():
    """Проверяем какие фичи реально передаются в API"""

    # Тестовые данные
    test_data = {
        "distance_in_meters": 3404.0,
        "duration_in_seconds": 486.0,
        "pickup_in_meters": 790.0,
        "pickup_in_seconds": 169.0,
        "driver_rating": 5.0,
        "user_rating": 4.8,
        "price_start_local": 180.0,
        "order_timestamp": "2020-05-01T00:05:14",
        "driver_platform": "Android",
        "driver_reg_date": "2019-09-22",
        "carmodel": "Logan",
        "carname": "Renault"
    }

    print("🔧 ДЕБАГ API ФИЧЕЙ")
    print("=" * 50)

    try:
        response = requests.post(
            "http://localhost:8000/api/v1/get_optimal_prices",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ API ответ получен")
            print(f"📊 Вероятности: {[p['probability'] for p in result['price_curve']]}")
            print(f"💰 Цены: {[p['price'] for p in result['price_curve']]}")
        else:
            print(f"❌ Ошибка API: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"❌ Ошибка запроса: {e}")


def test_direct_model():
    """Прямой тест модели с фичами из API"""
    import pandas as pd
    import numpy as np
    import joblib
    from datetime import datetime

    print("\n🔧 ПРЯМОЙ ТЕСТ МОДЕЛИ")
    print("=" * 50)

    # Загружаем модель
    model_data = joblib.load("models/price_acceptance_model.joblib")
    model = model_data['model']
    feature_names = model_data['feature_names']
    cat_features = model_data['categorical_features']

    # Тестовые данные
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

    # Подготавливаем фичи как в API (должны быть acceptance rates 0.8/0.7)
    df = pd.DataFrame([test_data])

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

    # Ценовые фичи
    df["price_diff"] = df["price_bid_local"] - df["price_start_local"]
    df["price_ratio"] = df["price_bid_local"] / (df["price_start_local"] + 1e-6)
    df["price_increase_pct"] = (df["price_diff"] / (df["price_start_local"] + 1e-6)) * 100
    df["price_per_km"] = df["price_bid_local"] / (df["distance_km"] + 1e-6)
    df["price_per_min"] = df["price_bid_local"] / (df["duration_minutes"] + 1e-6)
    df["rating_price_interaction"] = df["driver_rating"] * df["price_ratio"]
    df["experience_price_ratio"] = df["driver_experience_days"] * df["price_ratio"]

    # Категориальные фичи
    df['platform'] = df['driver_platform'].astype(str)
    df['carmodel'] = df['carmodel'].astype(str)
    df['carname'] = df['carname'].astype(str)

    # ВАЖНО: acceptance rates которые ДОЛЖНЫ быть в API
    df['driver_acceptance_rate'] = 0.8
    df['user_acceptance_rate'] = 0.7

    # Заполняем пропущенные фичи
    for feature in feature_names:
        if feature not in df.columns:
            if feature in cat_features:
                df[feature] = 'unknown'
            else:
                df[feature] = 0.0

    # Проверяем ключевые фичи
    print("📊 КЛЮЧЕВЫЕ ФИЧИ:")
    print(f"  driver_acceptance_rate: {df['driver_acceptance_rate'].iloc[0]}")
    print(f"  user_acceptance_rate: {df['user_acceptance_rate'].iloc[0]}")
    print(f"  driver_rating: {df['driver_rating'].iloc[0]}")

    # Предсказание
    features_df = df[feature_names]
    pool = pd.DataFrame(features_df)
    probability = model.predict_proba(pool)[0, 1]

    print(f"\n🎯 ПРЕДСКАЗАНИЕ:")
    print(f"   Вероятность: {probability:.3f}")
    print(f"   Ожидаемый доход: {test_data['price_bid_local'] * probability:.1f} руб")

    # Должна быть вероятность ~0.95 с rates 0.8/0.7


if __name__ == "__main__":
    debug_api_features()
    test_direct_model()