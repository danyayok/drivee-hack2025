# scripts/diagnose_data_issues.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def diagnose_data_issues():
    print("🔍 Диагностика проблем данных...")

    # Загружаем данные
    df = pd.read_csv("train.csv")
    print(f"📊 Размер данных: {df.shape}")

    # Анализируем целевой признак
    df['is_done'] = df['is_done'].astype(str).str.lower().map({"done": 1, "cancel": 0})
    df = df[df['is_done'].isin([0, 1])]

    print(f"🎯 Распределение целевой переменной:")
    print(f"   Принято: {df['is_done'].sum()} ({df['is_done'].mean() * 100:.1f}%)")
    print(f"   Отменено: {len(df) - df['is_done'].sum()} ({(1 - df['is_done'].mean()) * 100:.1f}%)")

    # Проверяем утечку данных
    check_data_leakage(df)

    # Анализируем фичи
    analyze_features(df)


def check_data_leakage(df):
    """Проверяем утечку данных"""
    print("\n🔎 Проверка утечки данных:")

    # Проверяем корреляции
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()['is_done'].abs().sort_values(ascending=False)

    print("📈 Корреляции с целевой переменной:")
    for feature, corr in correlations.head(10).items():
        if feature != 'is_done':
            print(f"   {feature}: {corr:.3f}")

    # Подозрительные корреляции (> 0.5)
    suspicious = correlations[correlations > 0.5].index.tolist()
    if 'is_done' in suspicious:
        suspicious.remove('is_done')
    if suspicious:
        print(f"🚨 Подозрительные фичи (возможная утечка): {suspicious}")


def analyze_features(df):
    """Анализируем проблемные фичи"""
    print("\n📊 Анализ фичей:")

    # Проверяем driver_cancel_rate и user_cancel_rate
    if 'driver_cancel_rate' in df.columns:
        print(f"📊 driver_cancel_rate:")
        print(f"   Min: {df['driver_cancel_rate'].min():.3f}")
        print(f"   Max: {df['driver_cancel_rate'].max():.3f}")
        print(f"   Mean: {df['driver_cancel_rate'].mean():.3f}")

        # Проверяем как считается эта фича
        sample_driver = df['driver_id'].iloc[0]
        driver_data = df[df['driver_id'] == sample_driver]
        print(f"   Пример для driver {sample_driver}:")
        print(f"     Всего заказов: {len(driver_data)}")
        print(f"     Принято: {driver_data['is_done'].sum()}")
        print(f"     cancel_rate: {driver_data['driver_cancel_rate'].iloc[0]:.3f}")


if __name__ == "__main__":
    diagnose_data_issues()