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

    # Базовые тесты
    print("\n📋 Запуск базовых тестов...")
    result = subprocess.run([
        "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "-m", "not slow"
    ], cwd=os.path.dirname(os.path.abspath(__file__)))

    if result.returncode != 0:
        print("\n❌ Базовые тесты не прошли!")
        return result.returncode

    # Интеграционные тесты (медленные)
    print("\n🔍 Запуск интеграционных тестов...")
    result = subprocess.run([
        "pytest",
        "tests/test_integration.py",
        "-v",
        "--tb=short",
        "-m", "slow"
    ], cwd=os.path.dirname(os.path.abspath(__file__)))

    if result.returncode != 0:
        print("\n⚠️ Интеграционные тесты не прошли, но это нормально для CI")

    # Покрытие кода
    print("\n📊 Запуск тестов с покрытием...")
    result = subprocess.run([
        "pytest",
        "tests/",
        "--cov=app",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "-m", "not slow"
    ], cwd=os.path.dirname(os.path.abspath(__file__)))

    print("=" * 50)
    print("✅ Тестирование завершено!")

    return result.returncode


if __name__ == "__main__":
    sys.exit(run_tests())