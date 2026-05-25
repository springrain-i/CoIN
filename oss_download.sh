#!/bin/bash

# ================= Configuration =================
LOG_DIR=./logs
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/oss_download_$(date +"%Y%m%d_%H%M%S").log"

echo "=========================================="
echo "Script started at: $(date)"
echo "Log file: $LOG_FILE"
echo "=========================================="

# Destination directory
DEST_DIR="/hy-tmp"

# ================= Functions =================

# 自定义日志函数：同时打印到屏幕和日志文件
log_msg() {
    local msg="[$(date +'%Y-%m-%d %T')] $1"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

process_dataset() {
    local oss_path="$1"
    local filename=$(basename "$oss_path")
    
    echo "------------------------------------------"
    log_msg "Starting process for: $filename"
    
    # 1. Download
    log_msg "Downloading $filename..."
    oss cp "$oss_path" "$DEST_DIR"
    
    # 检查下载是否成功
    if [ $? -ne 0 ]; then
        log_msg "ERROR: Download failed for $filename"
        return 1
    fi
    
    # 2. Extract
    log_msg "Extracting $filename..."
    # 修复：去掉了 -v 选项，防止刷屏
    if [[ "$filename" == *.tar.gz ]]; then
        tar -zxf "$DEST_DIR/$filename" -C "$DEST_DIR"
    elif [[ "$filename" == *.tar ]]; then
        tar -xf "$DEST_DIR/$filename" -C "$DEST_DIR"
    else
        tar -xf "$DEST_DIR/$filename" -C "$DEST_DIR"
    fi
    
    # 3. Clean up
    log_msg "Removing archive $filename..."
    rm -f "$DEST_DIR/$filename"
    log_msg "Finished $filename"
}

# ================= Execution =================

# --- Model Weights & Projectors ---
# process_dataset "oss://Vicuna.tar"
# process_dataset "oss://clip-vit-large-patch14-336.tar"
# process_dataset "oss://llava_projectors.tar"

# --- Standard Datasets ---
# process_dataset "oss://COCO2014.tar.gz"
# process_dataset "oss://GQA.tar.gz"
process_dataset "oss://ImageNet_withlabel.tar.gz"
# process_dataset "oss://OCR-VQA.tar.gz"
# process_dataset "oss://RefCOCO.tar.gz"
# process_dataset "oss://ScienceQA.tar.gz"
# process_dataset "oss://TextVQA.tar.gz"
# process_dataset "oss://VizWiz.tar.gz"

# --- Playground Dataset (Special Logic) ---
# process_dataset "oss://playground.tar.gz"

log_msg "All tasks completed."