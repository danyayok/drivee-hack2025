FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Создание пользователя для безопасности
RUN groupadd -r app && useradd -r -g app app

# Создание директорий
RUN mkdir -p /app/models /app/data /var/log/price-optimizer
RUN chown -R app:app /app /var/log/price-optimizer

WORKDIR /app

# Копирование requirements и установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование приложения
COPY . .

# Права доступа
RUN chown -R app:app /app
USER app

# Создание симлинков для Linux
RUN ln -sf /dev/stdout /var/log/price-optimizer/app.log

# Переменные окружения для Linux
ENV MODEL_PATH=/app/models/catboost_taxi_smart.joblib
ENV DATA_PATH=/app/data/train.csv
ENV LOG_DIR=/var/log/price-optimizer
ENV PROCESS_POOL_WORKERS=0
ENV THREAD_POOL_WORKERS=16

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/live || exit 1

CMD ["python", "run.py"]