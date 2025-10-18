import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
import joblib  # для сохранения модели

warnings.filterwarnings("ignore")


class TaxiModelTrainerCatBoost:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.model = None
        self.features = []
        self.target = "is_done"

    def preprocess_target(self):
        """Преобразуем целевую переменную"""
        self.df[self.target] = self.df[self.target].astype(str).str.lower().map({"done": 1, "cancel": 0})
        self.df = self.df[self.df[self.target].isin([0, 1])].reset_index(drop=True)

    def create_features(self):
        df = self.df.copy()
        df["order_timestamp"] = pd.to_datetime(df["order_timestamp"], errors="coerce")
        df["tender_timestamp"] = pd.to_datetime(df["tender_timestamp"], errors="coerce")
        df["driver_reg_date"] = pd.to_datetime(df["driver_reg_date"], errors="coerce")

        df["order_hour"] = df["order_timestamp"].dt.hour
        df["order_dayofweek"] = df["order_timestamp"].dt.dayofweek
        df["order_month"] = df["order_timestamp"].dt.month
        df["is_peak_hour"] = df["order_hour"].isin([7,8,9,17,18,19]).astype(int)
        df["is_weekend"] = (df["order_dayofweek"] >= 5).astype(int)
        df["response_delay_sec"] = (df["tender_timestamp"] - df["order_timestamp"]).dt.total_seconds().clip(lower=0)
        df["driver_experience_days"] = ((df["order_timestamp"] - df["driver_reg_date"]).dt.total_seconds() / (3600*24)).clip(lower=0)

        df["price_diff"] = df["price_bid_local"] - df["price_start_local"]
        df["price_ratio"] = df["price_bid_local"] / (df["price_start_local"] + 1e-6)
        df["price_per_km"] = df["price_bid_local"] / (df["distance_in_meters"]/1000 + 1e-6)
        df["price_per_min"] = df["price_bid_local"] / (df["duration_in_seconds"]/60 + 1e-6)

        df["trip_speed_kmh"] = (df["distance_in_meters"]/1000) / (df["duration_in_seconds"]/3600 + 1e-6)
        df["pickup_speed_kmh"] = (df["pickup_in_meters"]/1000) / (df["pickup_in_seconds"]/3600 + 1e-6)
        df["speed_ratio"] = df["trip_speed_kmh"] / (df["pickup_speed_kmh"] + 1e-6)

        df["rating_price_interaction"] = df["driver_rating"] * df["price_ratio"]
        df["rating_speed_interaction"] = df["driver_rating"] * df["trip_speed_kmh"]
        df["experience_price_ratio"] = df["driver_experience_days"] * df["price_ratio"]

        for col in ["price_bid_local", "distance_in_meters", "duration_in_seconds", "driver_experience_days"]:
            df[f"log_{col}"] = np.log1p(df[col])

        df['driver_cancel_rate'] = df.groupby('driver_id')[self.target].transform(lambda x: 1 - x.mean())
        df['user_cancel_rate'] = df.groupby('user_id')[self.target].transform(lambda x: 1 - x.mean())

        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        self.df = df
        return df

    def preprocess_features(self):
        df = self.df.copy()
        drop_cols = [
            self.target, "order_id", "tender_id", "tender_timestamp",
            "driver_reg_date", "order_timestamp", "user_id"
        ]
        X = df.drop(columns=drop_cols, errors="ignore")
        y = df[self.target]

        cat_cols = ["driver_id", "carmodel", "carname", "platform"]
        cat_cols = [c for c in cat_cols if c in X.columns]

        self.features = X.columns.tolist()
        return X, y, cat_cols

    def train_model(self, test_size=0.2, random_state=42, save_path="catboost_taxi_model.joblib"):
        X, y, cat_cols = self.preprocess_features()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        train_pool = Pool(X_train, y_train, cat_features=cat_cols)
        test_pool = Pool(X_test, y_test, cat_features=cat_cols)

        model = CatBoostClassifier(
            iterations=2000,
            depth=8,
            learning_rate=0.03,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=random_state,
            verbose=200,
            early_stopping_rounds=150,
            l2_leaf_reg=3.0
        )

        model.fit(train_pool, eval_set=test_pool, use_best_model=True)
        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)

        self.model = model
        print(f"\n✅ Final AUC: {auc:.4f}")
        print(f"📊 Features used: {len(self.features)}")
        print("Категориальные:", cat_cols)

        # Сохраняем модель на диск вместе с признаками
        joblib.dump({"model": self.model, "features": self.features, "cat_features": cat_cols}, save_path)
        print(f"💾 Модель сохранена: {save_path}")

        return auc

    def get_feature_importance(self, top_n=20):
        if self.model is None:
            raise ValueError("Model not trained yet. Run train_model() first.")
        fi = pd.DataFrame({
            "feature": self.features,
            "importance": self.model.get_feature_importance()
        }).sort_values(by="importance", ascending=False)
        print("\n🔝 Top features:")
        print(fi.head(top_n))
        return fi


if __name__ == "__main__":
    df = pd.read_csv("train.csv")
    trainer = TaxiModelTrainerCatBoost(df)
    trainer.preprocess_target()
    trainer.create_features()
    trainer.train_model()
    trainer.get_feature_importance()
