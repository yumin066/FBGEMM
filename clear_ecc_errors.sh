#!/bin/bash
# 每 30 秒执行一次 sudo nvidia-smi -p 0 清除 ECC 错误

echo "开始循环清除 ECC 错误（每 30 秒一次），按 Ctrl+C 停止..."

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 执行 sudo nvidia-smi -p 0"
    sudo nvidia-smi -p 0
    sleep 30
done
