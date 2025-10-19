# scripts/diagnose_smart_model_text.py
import pandas as pd
import numpy as np
import joblib
from catboost import Pool
import warnings

warnings.filterwarnings('ignore')


class SmartTaxiModelDiagnostics:
    def __init__(self, model_path="models/catboost_taxi_smart.joblib"):
        print("🔄 Загружаем модель...")
        model_data = joblib.load(model_path)
        if isinstance(model_data, dict) and 'model' in model_data:
            self.model = model_data['model']
            self.feature_names = model_data.get('feature_names', [])
            self.cat_features = model_data.get('cat_features', [])
        else:
            self.model = model_data
            self.feature_names = []
            self.cat_features = ["carmodel", "carname", "platform"]
        print(f"✅ Модель загружена. Фичей: {len(self.feature_names)}, Категориальные: {self.cat_features}")

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()
        X["price_ratio"] = X["price_bid_local"] / (X["price_start_local"] + 1e-6)
        X["price_diff"] = X["price_bid_local"] - X["price_start_local"]
        X["distance_km"] = X["distance_in_meters"] / 1000
        X["duration_min"] = X["duration_in_seconds"] / 60
        X["trip_speed_kmh"] = X["distance_km"] / (X["duration_min"] / 60 + 1e-6)
        X["price_per_km"] = X["price_bid_local"] / (X["distance_km"] + 1e-6)
        X["price_per_min"] = X["price_bid_local"] / (X["duration_min"] + 1e-6)
        X["rating_price_interaction"] = X["driver_rating"] * X["price_ratio"]
        X["rating_speed_interaction"] = X["driver_rating"] * X["trip_speed_kmh"]
        X["order_timestamp"] = pd.to_datetime(X["order_timestamp"])
        X["order_hour"] = X["order_timestamp"].dt.hour
        X["order_dayofweek"] = X["order_timestamp"].dt.dayofweek
        X["order_month"] = X["order_timestamp"].dt.month
        X["is_peak_hour"] = X["order_hour"].isin([7,8,9,17,18,19]).astype(int)
        X["is_weekend"] = (X["order_dayofweek"] >= 5).astype(int)
        X["pickup_speed_kmh"] = (X["pickup_in_meters"] / 1000) / (X["pickup_in_seconds"] / 3600 + 1e-6)
        X["driver_reg_date"] = pd.to_datetime(X["driver_reg_date"])
        X["driver_experience_days"] = ((X["order_timestamp"] - X["driver_reg_date"]).dt.total_seconds() / (3600*24)).clip(lower=0)
        X["tender_timestamp"] = pd.to_datetime(X["tender_timestamp"])
        X["response_delay_sec"] = (X["tender_timestamp"] - X["order_timestamp"]).dt.total_seconds().clip(lower=0)
        for col in ["price_bid_local","distance_in_meters","duration_in_seconds","driver_experience_days"]:
            X[f"log_{col}"] = np.log1p(X[col].clip(lower=0))
        X["experience_price_ratio"] = X["driver_experience_days"] * X["price_ratio"]
        for col in ["platform","carmodel","carname"]:
            if col in X.columns:
                X[col] = X[col].astype(str)
        if 'user_rating' not in X.columns:
            X['user_rating'] = 4.6
        X['driver_cancel_rate'] = 0.35
        X['user_cancel_rate'] = 0.4
        X['driver_avg_price'] = 200.0
        X['user_avg_price'] = 200.0

        if self.feature_names:
            missing_features = set(self.feature_names) - set(X.columns)
            for feature in missing_features:
                X[feature] = 0.0
            X_final = X[self.feature_names]
        else:
            X_final = X
        return X_final

    def predict(self, df: pd.DataFrame):
        X = self.prepare_features(df)
        pool = Pool(X, cat_features=self.cat_features)
        probabilities = self.model.predict_proba(pool)[:,1]
        df["predicted_probability"] = probabilities
        return df, probabilities

    def show_feature_importance(self, top_n=20):
        fi = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.get_feature_importance()
        }).sort_values('importance', ascending=False)
        print(f"\n🔝 Топ-{top_n} важных фичей:")
        print(fi.head(top_n).to_string(index=False))
        return fi

    def recommend_threshold(self, probabilities, target_done_ratio=0.7):
        thresholds = np.linspace(0.3, 0.7, 50)
        best_threshold = 0.5
        best_diff = float('inf')
        for threshold in thresholds:
            predicted_done = (probabilities > threshold).sum()
            current_ratio = predicted_done / len(probabilities)
            diff = abs(current_ratio - target_done_ratio)
            if diff < best_diff:
                best_diff = diff
                best_threshold = threshold
        print(f"\n🎯 Рекомендованный порог для ~{target_done_ratio*100:.0f}% done: {best_threshold:.3f}")
        return best_threshold

    def what_if_analysis(self, df: pd.DataFrame):
        df_sample = df.iloc[[0]].copy()
        base_price = df_sample['price_start_local'].iloc[0]
        scenarios = [
            ("Низкая цена", base_price*0.7),
            ("Высокая цена", base_price*1.5),
            ("Плохой рейтинг", df_sample['driver_rating'].iloc[0]*0.7),
            ("Долгий pickup", df_sample['pickup_in_seconds'].iloc[0]*3),
            ("Очень короткая поездка", 100)
        ]
        print("\n🔍 What-if анализ:")
        for name, value in scenarios:
            test_row = df_sample.copy()
            if "цена" in name.lower():
                test_row['price_bid_local'] = value
                test_row['price_ratio'] = test_row['price_bid_local'] / (test_row['price_start_local'] + 1e-6)
                test_row['price_diff'] = test_row['price_bid_local'] - test_row['price_start_local']
                test_row['rating_price_interaction'] = test_row['driver_rating'].iloc[0] * test_row['price_ratio'].iloc[0]
                test_row['experience_price_ratio'] = test_row['driver_experience_days'].iloc[0] * test_row['price_ratio'].iloc[0]
            elif "рейтинг" in name.lower():
                test_row['driver_rating'] = value
                test_row['rating_price_interaction'] = test_row['driver_rating'].iloc[0] * test_row['price_ratio'].iloc[0]
                test_row['rating_speed_interaction'] = test_row['driver_rating'].iloc[0] * test_row['trip_speed_kmh'].iloc[0]
            elif "pickup" in name.lower():
                test_row['pickup_in_seconds'] = value
                test_row['pickup_speed_kmh'] = (test_row['pickup_in_meters'] / 1000) / (test_row['pickup_in_seconds'] / 3600 + 1e-6)
            elif "короткая поездка" in name.lower():
                test_row['distance_in_meters'] = value
                test_row['distance_km'] = value / 1000
                test_row['trip_speed_kmh'] = test_row['distance_km'].iloc[0] / (test_row['duration_min'].iloc[0]/60 + 1e-6)
                test_row['price_per_km'] = test_row['price_bid_local'] / (test_row['distance_km'] + 1e-6)
                test_row['rating_speed_interaction'] = test_row['driver_rating'].iloc[0] * test_row['trip_speed_kmh'].iloc[0]
            pool = Pool(test_row[self.feature_names], cat_features=self.cat_features)
            prob = self.model.predict_proba(pool)[:,1][0]
            print(f"   {name}: вероятность done = {prob:.3f}")


def main():
    df = pd.read_csv("test.csv")
    diag = SmartTaxiModelDiagnostics()
    df_pred, probabilities = diag.predict(df)

    print("\n📊 Статистика вероятностей:")
    print(f"   Min: {probabilities.min():.3f}")
    print(f"   Mean: {probabilities.mean():.3f}")
    print(f"   Max: {probabilities.max():.3f}")
    print(f"   Std: {probabilities.std():.3f}")

    diag.show_feature_importance(top_n=20)
    diag.recommend_threshold(probabilities)
    diag.what_if_analysis(df_pred)


if __name__ == "__main__":
    main()
