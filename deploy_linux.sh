#!/bin/bash

echo "🚀 Деплой Price Optimizer API на Linux..."

# Создание директорий
sudo mkdir -p /opt/price-optimizer/{models,data,logs}
sudo chown -R $USER:$USER /opt/price-optimizer

# Копирование файлов
cp models/catboost_taxi_smart.joblib /opt/price-optimizer/models/
cp data/train.csv /opt/price-optimizer/data/

# Создание systemd service
sudo tee /etc/systemd/system/price-optimizer.service > /dev/null <<EOF
[Unit]
Description=Price Optimizer API
After=network.target

[Service]
Type=exec
User=$USER
WorkingDirectory=/opt/price-optimizer
Environment=MODEL_PATH=/opt/price-optimizer/models/catboost_taxi_smart.joblib
Environment=DATA_PATH=/opt/price-optimizer/data/train.csv
Environment=LOG_DIR=/var/log/price-optimizer
Environment=PROCESS_POOL_WORKERS=0
Environment=THREAD_POOL_WORKERS=16
Environment=DEBUG=false
ExecStart=/usr/bin/python3 run.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Перезагрузка systemd и запуск
sudo systemctl daemon-reload
sudo systemctl enable price-optimizer
sudo systemctl start price-optimizer

echo "✅ Деплой завершен! Сервис запущен."
echo "📊 Проверка статуса: sudo systemctl status price-optimizer"
echo "📝 Логи: sudo journalctl -u price-optimizer -f"