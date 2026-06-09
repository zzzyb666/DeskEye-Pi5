#!/usr/bin/env bash
# 在桌面环境下验证 GStreamer + libcamera（不含 OpenCV）。
# 用法：在 ~/deskeye 下 export DESKEYE_CAMERA_NAME='...' 后执行本脚本。
set -euo pipefail
NAME="${DESKEYE_CAMERA_NAME:-}"
if [[ -z "$NAME" ]]; then
  echo "请先: export DESKEYE_CAMERA_NAME='/base/axi/pcie@...'"
  exit 1
fi
echo "使用 camera-name=$NAME ，按 Ctrl+C 结束"
exec gst-launch-1.0 -v \
  libcamerasrc "camera-name=${NAME}" ! \
  "video/x-raw,width=1296,height=972,format=NV12" ! queue ! \
  videoconvert ! videoscale ! \
  "video/x-raw,width=640,height=480,format=BGR" ! \
  autovideosink
