#!/usr/bin/env bash
# 列出 CSI/USB 摄像头（新版系统用 rpicam-*，旧版用 libcamera-*）
set -euo pipefail

if command -v rpicam-hello >/dev/null 2>&1; then
  echo "使用 rpicam-hello（当前 Raspberry Pi OS 推荐）"
  exec rpicam-hello --list-cameras
fi

if command -v libcamera-hello >/dev/null 2>&1; then
  echo "使用 libcamera-hello（较旧系统）"
  exec libcamera-hello --list-cameras
fi

echo "未找到 rpicam-hello / libcamera-hello。请安装：sudo apt install rpicam-apps"
exit 1
