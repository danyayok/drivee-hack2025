import joblib
import pandas as pd
import numpy as np


def debug_current_model():
    """Диагностика текущей модели"""

    # Загружаем модель
    try:
        model_data = joblib.load("models/catboost_taxi_model.joblib")
        model = model_data['model']
        feature_names = model_data['features']

        print("✅ Модель загружена")
        print(f"📊 Ожидаемые фичи ({len(feature_names)}):")
        for i, feature in enumerate(feature_names[:20]):  # Покажем первые 20
            print(f"  {i + 1}. {feature}")

        # Проверим, есть ли price_bid_local в фичах
        has_price_bid = 'price_bid_local' in feature_names
        print(f"\n🔍 price_bid_local в фичах: {has_price_bid}")

        if has_price_bid:
            # Проверим важность фичи
            importance_dict = dict(zip(feature_names, model.get_feature_importance()))
            price_importance = importance_dict.get('price_bid_local', 0)
            print(f"📈 Важность price_bid_local: {price_importance}")

        return model, feature_names

    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return None, None


if __name__ == "__main__":
    debug_current_model()