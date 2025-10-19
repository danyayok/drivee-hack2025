#!/usr/bin/env python3
"""
Скрипт для запуска тестов
"""
import subprocess
import sys
import os


def run_tests():
    """Запуск тестов с разными опциями"""

    print("🚀 Запуск тестов Price Optimizer API...")
    print("=" * 50)

    # Тесты моделей данных (самые стабильные)
    print("\n📋 Запуск тестов моделей данных...")
    result_models = subprocess.run([
        "pytest",
        "tests/test_data_models.py",
        "-v",
        "--tb=short"
    ], cwd=os.path.dirname(os.path.abspath(__file__)))

    if result_models.returncode != 0:
        print("\n❌ Тесты моделей данных не прошли!")
        return result_models.returncode

    # Тесты API
    print("\n🌐 Запуск API тестов...")
    result_api = subprocess.run([
        "pytest",
        "tests/test_api.py",
        "-v",
        "--tb=short"
    ], cwd=os.path.dirname(os.path.abspath(__file__)))

    if result_api.returncode != 0:
        print("\n❌ API тесты не прошли!")
        return result_api.returncode

    # Тесты статистики (могут быть проблемными)
    print("\n📊 Запуск тестов статистики...")
    result_stats = subprocess.run([
        "pytest",
        "tests/test_statistics.py",
        "-v",
        "--tb=short"
    ], cwd=os.path.dirname(os.path.abspath(__file__)))

    if result_stats.returncode != 0:
        print("\n⚠️ Тесты статистики имеют проблемы, но продолжаем...")

    # Тесты ML (опционально)
    print("\n🤖 Запуск ML тестов...")
    try:
        result_ml = subprocess.run([
            "pytest",
            "tests/test_ml_model.py",
            "-v",
            "--tb=short"
        ], cwd=os.path.dirname(os.path.abspath(__file__)))

        if result_ml.returncode != 0:
            print("\n⚠️ ML тесты имеют проблемы")
    except FileNotFoundError:
        print("\n⚠️ ML тесты не найдены, пропускаем...")

    print("=" * 50)
    print("✅ Основное тестирование завершено!")

    # Возвращаем код самой серьезной ошибки
    return max(result_models.returncode, result_api.returncode, result_stats.returncode)


if __name__ == "__main__":
    sys.exit(run_tests())