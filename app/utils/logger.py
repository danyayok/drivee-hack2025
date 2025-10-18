import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Создает и настраивает логгер

    Args:
        name: Имя логгера (обычно __name__)
        level: Уровень логирования (по умолчанию INFO)

    Returns:
        Настроенный логгер
    """
    if level is None:
        level = logging.INFO

    # Создаем логгер
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Если уже есть обработчики, не добавляем новые
    if logger.handlers:
        return logger

    # Создаем форматтер
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)'
    )

    # Обработчик для stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(formatter)

    # Добавляем обработчики к логгеру
    logger.addHandler(stdout_handler)

    # Предотвращаем дублирование логов
    logger.propagate = False

    return logger