#!/usr/bin/env bash
# DeskEye 一键安装（阶段 6）：依赖、venv、模型、systemd 自启
#
# 用法：chmod +x install.sh && ./install.sh
#       或：bash install.sh
#
# Debian Trixie / Pi OS 受 PEP 668 约束，禁止系统 Python 直接 pip install。
# 本脚本自动创建 .venv（--system-site-packages）以复用 apt 的 opencv，再 pip 安装 onnxruntime。
set -euo pipefail

DESKEYE_DIR="$(cd "$(dirname "$0")" && pwd)"
DESKEYE_USER="$(whoami)"
VENV_DIR="${DESKEYE_DIR}/.venv"
SYSTEM_PYTHON="$(command -v python3)"

# piwheels 大包易超时：pip 默认超时与重试
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-180}"
export PIP_RETRIES="${PIP_RETRIES:-5}"

_pip_install_with_retry() {
  local pip_bin="$1"
  shift
  local attempt
  for attempt in 1 2 3; do
    echo "pip 安装 (第 ${attempt}/3 次): $*"
    if "${pip_bin}" install --retries "${PIP_RETRIES}" "$@"; then
      return 0
    fi
    echo "pip 下载失败，10 秒后重试…" >&2
    sleep 10
  done
  echo "尝试改用 PyPI 官方源（绕过 piwheels）…" >&2
  "${pip_bin}" install --retries "${PIP_RETRIES}" \
    --index-url https://pypi.org/simple \
    --extra-index-url https://www.piwheels.org/simple \
    "$@"
}

echo "==> DeskEye 安装目录: ${DESKEYE_DIR}"
echo "==> 运行用户: ${DESKEYE_USER}"
echo "==> 系统 Python: ${SYSTEM_PYTHON}"
echo "==> 用法提示: chmod +x install.sh && ./install.sh  （或 bash install.sh）"

if [[ ! -f "${DESKEYE_DIR}/main.py" ]]; then
  echo "错误：未找到 main.py，请在项目根目录运行本脚本。" >&2
  exit 1
fi

echo ""
echo "==> [1/5] 安装系统依赖（GStreamer / libcamera / OpenCV）"
if [[ -f "${DESKEYE_DIR}/scripts/pi_install_deps.sh" ]]; then
  bash "${DESKEYE_DIR}/scripts/pi_install_deps.sh" || {
    echo ""
    echo "【提示】若 apt 报 raspbian bookworm 源 SHA1 签名校验失败，可暂时注释失效源后重试：" >&2
    echo "  sudo nano /etc/apt/sources.list.d/raspi.list  # 或相关 list 文件" >&2
    echo "  注释含 mirrors.tuna.tsinghua.edu.cn/raspbian/raspbian bookworm 的行" >&2
    echo "  sudo apt update && ./install.sh" >&2
    exit 1
  }
else
  sudo apt-get update
  sudo apt-get install -y \
    python3-opencv python3-numpy python3-pip python3-venv python3-full \
    gstreamer1.0-libcamera gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-tools
fi

echo ""
echo "==> [2/5] 创建虚拟环境并安装 Python 依赖（规避 PEP 668）"
if [[ ! -x "${VENV_DIR}/bin/python3" ]]; then
  echo "创建 ${VENV_DIR}（--system-site-packages，复用 apt 的 python3-opencv）"
  "${SYSTEM_PYTHON}" -m venv --system-site-packages "${VENV_DIR}"
else
  echo "复用已有虚拟环境: ${VENV_DIR}"
fi

PYTHON3="${VENV_DIR}/bin/python3"
PIP="${VENV_DIR}/bin/pip"

# 不升级 pip：venv 自带 pip 已可用；piwheels 上 pip _wheel 约 1.8MB 易因网络超时失败
echo "跳过 pip 自升级（避免 piwheels 大包下载超时）"

# Flask 优先 apt（--system-site-packages 可复用），仅 pip 安装 onnxruntime 减小下载量
if ! "${PYTHON3}" -c "import flask" 2>/dev/null; then
  echo "尝试 apt 安装 python3-flask …"
  sudo apt-get install -y python3-flask || true
fi

if "${PYTHON3}" -c "import flask" 2>/dev/null; then
  echo "Flask 已由系统包提供"
  _pip_install_with_retry "${PIP}" "onnxruntime>=1.16.0"
else
  echo "系统无 Flask，将通过 pip 安装 Flask + onnxruntime"
  _pip_install_with_retry "${PIP}" "Flask>=3.0.0" "onnxruntime>=1.16.0"
fi

# 验证关键包可导入
"${PYTHON3}" -c "import cv2; import flask; import onnxruntime; print('Python 依赖 OK: cv2', cv2.__version__, 'flask', flask.__version__, 'ort', onnxruntime.__version__)"

echo ""
echo "==> [3/5] 下载检测模型"
"${PYTHON3}" "${DESKEYE_DIR}/scripts/download_face_models.py"
"${PYTHON3}" "${DESKEYE_DIR}/scripts/download_object_model.py"

echo ""
echo "==> [4/5] 准备环境文件 deskeye.env"
ENV_FILE="${DESKEYE_DIR}/deskeye.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  cp "${DESKEYE_DIR}/deskeye.env.example" "${ENV_FILE}"
  echo "已创建 ${ENV_FILE}，请确认 DESKEYE_CAMERA_NAME 是否正确。"
else
  echo "保留已有 ${ENV_FILE}"
fi
mkdir -p "${DESKEYE_DIR}/data" "${DESKEYE_DIR}/models"

echo ""
echo "==> [5/5] 安装 systemd 服务 deskeye"
SERVICE_DST="/etc/systemd/system/deskeye.service"
TMP_SERVICE="$(mktemp)"
sed \
  -e "s|__DESKEYE_DIR__|${DESKEYE_DIR}|g" \
  -e "s|__DESKEYE_USER__|${DESKEYE_USER}|g" \
  -e "s|__PYTHON3__|${PYTHON3}|g" \
  "${DESKEYE_DIR}/deskeye.service" > "${TMP_SERVICE}"
sudo cp "${TMP_SERVICE}" "${SERVICE_DST}"
rm -f "${TMP_SERVICE}"
sudo systemctl daemon-reload
sudo systemctl enable deskeye.service

echo ""
echo "安装完成。"
echo "  Python 解释器: ${PYTHON3}"
echo "  下一步："
echo "  1) 检查摄像头路径: nano ${ENV_FILE}"
echo "  2) 启动服务:       sudo systemctl start deskeye"
echo "  3) 查看状态:       sudo systemctl status deskeye"
echo "  4) 查看日志:       journalctl -u deskeye -n 30 --no-pager"
echo "     实时跟踪日志:   journalctl -u deskeye -f   # 按 Ctrl+C 退出"
echo "  5) 浏览器打开:     http://$(hostname -I 2>/dev/null | awk '{print $1}'):5000/"
