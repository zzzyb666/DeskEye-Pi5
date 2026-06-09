"""
阶段 0 验证：捕获一帧 640x480 BGR 图像并保存。
用法（在项目根目录）：python tests/test_camera.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

import config  # noqa: E402
from src.camera import Camera  # noqa: E402
from src.utils import ensure_dir  # noqa: E402


def main() -> int:
    ensure_dir(config.DATA_DIR)
    out_path = config.DATA_DIR / "test_frame.jpg"

    # 调试多管线尝试：DESKEYE_GST_DEBUG=1 python3 tests/test_camera.py
    cam = Camera()
    if not cam.open():
        print(
            "错误：无法打开摄像头。\n"
            "- 树莓派：执行 bash scripts/pi_install_deps.sh；确认用户在 video 组。\n"
            "  若仍失败，在 config.py 设置 CAMERA_LIBCAMERA_NAME 或导出 DESKEYE_CAMERA_NAME 为\n"
            "  rpicam-hello --list-cameras 中括号内完整路径。\n"
            "  列出设备：bash scripts/pi_list_cameras.sh\n"
            "- 开发机：可在 config.py 中设置 CAMERA_SOURCE_OVERRIDE = 0 使用本机摄像头调试。"
        )
        return 1

    ok, frame = cam.read()
    cam.release()

    if not ok or frame is None:
        print(
            "错误：读取帧失败（常见于 GStreamer caps 未协商成功）。\n"
            "- 确认已安装：sudo apt install gstreamer1.0-libcamera python3-opencv\n"
            "- 在 config.py 中设置 CAMERA_LIBCAMERA_NAME=你的设备路径，或：\n"
            "  export DESKEYE_CAMERA_NAME='/base/axi/pcie@1000120000/rp1/i2c@80000/ov5647@36'\n"
            "  （路径以 rpicam-hello --list-cameras 为准）\n"
            "- 勿对系统 Python 强行 pip install opencv（PEP 668）；优先 apt 的 python3-opencv 或 venv+--system-site-packages。"
        )
        return 1

    h, w = frame.shape[:2]
    if w != config.CAMERA_WIDTH or h != config.CAMERA_HEIGHT:
        print(f"警告：期望 {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}，实际 {w}x{h}，仍保存")

    cv2.imwrite(str(out_path), frame)
    print(f"已保存一帧到: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
