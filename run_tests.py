#!/usr/bin/env python3
"""
Запуск всех тестов
"""
import pytest
import sys
import os

if __name__ == "__main__":
    # Добавляем корневую директорию в PYTHONPATH
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Запускаем pytest
    exit_code = pytest.main([
        "tests/",
        "-v",  # verbose
        "--tb=short",  # короткий traceback
        "--cov=app",  # покрытие кода
        "--cov-report=term-missing",
        "-x"  # остановиться при первой ошибке
    ])

    sys.exit(exit_code)