import logging
import os
import platform
from logging.handlers import RotatingFileHandler, SysLogHandler


def setup_linux_logging():
    """Настройка логирования для Linux"""
    logger = logging.getLogger()

    if platform.system() != "Windows":
        try:
            log_dir = os.getenv("LOG_DIR", "/var/log/price-optimizer")

            # Создаем директорию с правильными правами
            os.makedirs(log_dir, exist_ok=True, mode=0o755)

            # Rotating File Handler
            file_handler = RotatingFileHandler(
                filename=os.path.join(log_dir, "app.log"),
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )

            # Syslog Handler для systemd
            try:
                syslog_handler = SysLogHandler(address='/dev/log')
                syslog_handler.setLevel(logging.INFO)
                logger.addHandler(syslog_handler)
            except:
                pass  # Syslog не доступен

            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            return file_handler

        except PermissionError:
            print("⚠️ Нет прав для записи логов в /var/log/, используем текущую директорию")
            # Fallback to current directory
            fallback_handler = logging.FileHandler("app.log", encoding='utf-8')
            logger.addHandler(fallback_handler)
            return fallback_handler

    return None