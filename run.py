#!/usr/bin/env python3
"""
Точка входа для запуска приложения с обработкой ошибок
"""
import uvicorn
import sys
import os
import signal
import asyncio

# Добавляем корневую директорию в PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class Application:
    """Класс для управления жизненным циклом приложения"""

    def __init__(self):
        self.shutdown_event = asyncio.Event()

    def handle_shutdown(self, sig, frame):
        """Обработчик сигналов завершения"""
        logger.info(f"🛑 Получен сигнал {sig}, начинаем graceful shutdown...")
        self.shutdown_event.set()

    async def run(self):
        """Запуск приложения"""
        try:
            logger.info("🚀 Запускаем Price Optimizer API...")
            logger.info(f"📊 Настройки: {settings.PROCESS_POOL_WORKERS} процессов, "
                        f"{settings.THREAD_POOL_WORKERS} потоков")

            # Настройка обработчиков сигналов
            signal.signal(signal.SIGINT, self.handle_shutdown)
            signal.signal(signal.SIGTERM, self.handle_shutdown)

            # Конфигурация сервера
            config = uvicorn.Config(
                "app.main:app",
                host=settings.HOST,
                port=settings.PORT,
                reload=settings.DEBUG,
                log_level="info",
                access_log=settings.DEBUG,
                workers=1,
                timeout_keep_alive=5,
                timeout_notify=30,
                timeout_graceful_shutdown=30
            )

            server = uvicorn.Server(config)

            # Запуск сервера с возможностью graceful shutdown
            server_task = asyncio.create_task(server.serve())

            # Ожидание сигнала завершения
            await self.shutdown_event.wait()

            # Graceful shutdown
            logger.info("🛑 Останавливаем сервер...")
            server.should_exit = True
            await server_task

            logger.info("✅ Сервер остановлен")

        except KeyboardInterrupt:
            logger.info("🛑 Остановка по запросу пользователя...")
        except Exception as e:
            logger.error(f"❌ Критическая ошибка запуска: {e}")
            sys.exit(1)


def main():
    """Основная функция запуска"""
    app = Application()
    asyncio.run(app.run())


if __name__ == "__main__":
    main()