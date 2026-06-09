#!/usr/bin/env bash
# 树莓派 Bookworm/Trixie：GStreamer + libcamera 插件 + 系统 OpenCV（推荐用于阶段 0）
set -euo pipefail

echo "==> 若 apt 报 raspbian 源 SHA1/OpenPGP 校验失败，请检查 /etc/apt/sources.list.d/ 是否混用失效镜像（可暂时注释 bookworm 的 raspbian 行后重试 sudo apt update）"
echo "==> 安装 GStreamer / libcamera 插件与系统 Python OpenCV（需 sudo）"
sudo apt-get update
sudo apt-get install -y \
  python3-venv \
  python3-full \
  python3-opencv \
  python3-numpy \
  python3-flask \
  gstreamer1.0-libcamera \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-tools

echo ""
echo "==> Python 包（Flask / onnxruntime）：Debian Trixie 受 PEP 668 约束，勿对系统 Python 直接 pip。"
echo "    请运行项目根目录 ./install.sh，会自动创建 .venv 并安装依赖。"
