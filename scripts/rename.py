import os
import shutil
import joblib


def rename_model():
    """Переименовываем модель для использования в API"""

    source = "models/price_acceptance_model.joblib"
    target = "models/catboost_taxi_model.joblib"

    if not os.path.exists(source):
        print("❌ Новая модель не найдена!")
        return

    # Создаем backup старой модели
    if os.path.exists(target):
        backup = "models/catboost_taxi_model_backup.joblib"
        shutil.copy2(target, backup)
        print(f"📦 Backup создан: {backup}")

    # Копируем новую модель
    shutil.copy2(source, target)
    print(f"✅ Модель переименована: {source} -> {target}")

    # Проверяем
    if os.path.exists(target):
        print("🎯 API теперь использует новую модель!")

        # Проверяем модель
        try:
            model_data = joblib.load(target)
            print(f"📊 Модель проверена: {len(model_data['feature_names'])} фичей")
        except Exception as e:
            print(f"❌ Ошибка проверки модели: {e}")


if __name__ == "__main__":
    rename_model()