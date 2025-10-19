import uvicorn
import sys
import os
import signal
import asyncio
import platform

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

    def setup_linux_optimizations(self):
        """Оптимизации для Linux"""
        if platform.system() != "Windows":
            try:
                # Увеличиваем лимиты файлов
                import resource
                soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
                logger.info(f"✅ Установлен лимит файлов: {min(65536, hard)}")
            except:
                logger.warning("⚠️ Не удалось установить лимиты файлов")

    async def run(self):
        """Запуск приложения"""
        try:
            logger.info("🚀 Запускаем Price Optimizer API...")

            # Оптимизации для Linux
            self.setup_linux_optimizations()

            logger.info(f"📊 Настройки: {settings.THREAD_POOL_WORKERS} потоков, "
                        f"ProcessPool: {'включен' if settings.PROCESS_POOL_WORKERS > 0 else 'отключен'}")

            # Настройка обработчиков сигналов
            if platform.system() != "Windows":
                signal.signal(signal.SIGTERM, self.handle_shutdown)
                signal.signal(signal.SIGINT, self.handle_shutdown)
            else:
                # Windows signal handling
                try:
                    import win32api
                    win32api.SetConsoleCtrlHandler(self.handle_shutdown, True)
                except ImportError:
                    pass

            # Конфигурация сервера
            config = uvicorn.Config(
                "app.main:app",
                host=settings.HOST,
                port=settings.PORT,
                reload=settings.DEBUG,
                log_level="info",
                access_log=settings.DEBUG,
                workers=1,
            )

            # Linux-specific optimizations
            if platform.system() != "Windows":
                config.timeout_keep_alive = 5
                config.timeout_notify = 30
                config.timeout_graceful_shutdown = 30

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