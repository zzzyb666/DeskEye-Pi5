"""
阶段 3 验证：手机检测（与 DeskEye_Cursor_Prompt.md 一致）。

规范要点（摘要）：
- 摄像头：与项目其余模块相同，OpenCV + GStreamer **libcamerasrc**（见 config / Camera）。
- 推理：**ONNX Runtime**，`providers=ort.get_available_providers()`（与规范「自动选最优 backend」一致）。
- 模型：规范要求 **YOLOv5n INT8 ONNX**；当前脚本配套下载为 release **FP16 输入** ONNX，生产请换 INT8。
- 验证：画面中出现手机时控制台输出 **PHONE_DETECTED** 及置信度；本脚本另提供 **实时预览**（OpenCV / 浏览器 MJPEG）。

用法（项目根）：
  python3 scripts/download_object_model.py   # 首次下载（若缺模型，本脚本默认也会尝试自动下载）
  export DESKEYE_CAMERA_NAME='/base/axi/pcie@.../ov5647@36'   # 填 rpicam-hello --list-cameras 括号内真实路径，勿用字面 ...
  python3 tests/test_object.py

环境变量：
  DESKEYE_YOLO_ONNX=/path/to/model.onnx   覆盖默认 models/yolov5n.onnx
  DESKEYE_AUTO_DOWNLOAD_YOLO=0           关闭缺模型时的自动下载
  DESKEYE_DETECT_EVERY=2                 每 N 帧跑一次 YOLO（默认见 config.OBJECT_DETECT_EVERY_N）
  DESKEYE_OBJECT_HOLD_SEC=1.2            检测框保持秒数（默认见 config.OBJECT_DISPLAY_HOLD_SEC）

实时预览：
  - 无 DISPLAY/WAYLAND（如纯 SSH）且未设置 DESKEYE_PREVIEW_HTTP 时，默认 **自动开启 MJPEG**
    （可用 `DESKEYE_AUTO_MJPEG=0` 关闭自动，再手动设 DESKEYE_PREVIEW_HTTP=1）。
  - `DESKEYE_PREVIEW_HTTP=1`：浏览器 http://<Pi-IP>:8765/
  - `DESKEYE_SHOW=1`：OpenCV 窗口（无 DISPLAY 时会尝试 DISPLAY=:0）
  - 每 N 帧写入 `data/object_preview.jpg`（环境变量 `DESKEYE_OBJECT_PREVIEW_EVERY`，默认 15）

按 q 或 Ctrl+C 退出。
"""
from __future__ import annotations

import os
import time

# 须在 import onnxruntime 之前；4=FATAL，尽量压制 GetGpuDevices 类告警
os.environ.setdefault("ORT_LOGGING_LEVEL", "4")

import socket
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

import config  # noqa: E402
from src.camera import Camera  # noqa: E402
from src.mjpeg_preview import MjpegPreviewServer  # noqa: E402
from src.object_detector import ObjectDetector, PhoneDetection  # noqa: E402
from src.utils import ensure_dir  # noqa: E402

PHONE_BGR = (0, 0, 255)


def _parse_detect_every() -> int:
    raw = os.environ.get("DESKEYE_DETECT_EVERY", "").strip()
    if not raw:
        return max(1, int(config.OBJECT_DETECT_EVERY_N))
    try:
        return max(1, int(raw))
    except ValueError:
        return max(1, int(config.OBJECT_DETECT_EVERY_N))


def _parse_hold_sec() -> float:
    raw = os.environ.get("DESKEYE_OBJECT_HOLD_SEC", "").strip()
    if not raw:
        return float(config.OBJECT_DISPLAY_HOLD_SEC)
    try:
        return max(0.0, float(raw))
    except ValueError:
        return float(config.OBJECT_DISPLAY_HOLD_SEC)


def _draw_phones(frame, phones: List[PhoneDetection]) -> None:
    for p in phones:
        cv2.rectangle(frame, (p.x1, p.y1), (p.x2, p.y2), PHONE_BGR, 2)
        cv2.putText(
            frame,
            f"phone {p.confidence:.2f}",
            (p.x1, max(0, p.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            PHONE_BGR,
            2,
            cv2.LINE_AA,
        )


def _parse_object_preview_every() -> int:
    raw = os.environ.get("DESKEYE_OBJECT_PREVIEW_EVERY", "15").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 15


def _has_gui_session() -> bool:
    if os.name == "nt":
        return True
    return bool(
        os.environ.get("DISPLAY", "").strip()
        or os.environ.get("WAYLAND_DISPLAY", "").strip()
    )


def _maybe_auto_mjpeg_headless() -> None:
    """纯 SSH 无图形会话时默认开 MJPEG，便于验证（可用 DESKEYE_AUTO_MJPEG=0 关闭）。"""
    if os.name == "nt":
        return
    if "DESKEYE_PREVIEW_HTTP" in os.environ:
        return
    if _has_gui_session():
        return
    if os.environ.get("DESKEYE_AUTO_MJPEG", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return
    os.environ["DESKEYE_PREVIEW_HTTP"] = "1"


def _bootstrap_display_env() -> None:
    if os.name == "nt":
        return
    raw = os.environ.get("DESKEYE_SHOW", "").strip().lower()
    if raw not in ("1", "true", "yes", "on"):
        return
    if not os.environ.get("DISPLAY", "").strip():
        os.environ["DISPLAY"] = ":0"
    xa = Path.home() / ".Xauthority"
    if xa.is_file():
        os.environ.setdefault("XAUTHORITY", str(xa))


def _want_show() -> bool:
    raw = os.environ.get("DESKEYE_SHOW", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return _has_gui_session()


def _parse_preview_http_port() -> int | None:
    raw = os.environ.get("DESKEYE_PREVIEW_HTTP", "").strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        return None
    port_default = 8765
    try:
        port_default = int(os.environ.get("DESKEYE_PREVIEW_HTTP_PORT", "8765").strip())
    except ValueError:
        port_default = 8765
    if raw in ("1", "true", "yes", "on"):
        return max(1, min(65535, port_default))
    try:
        return max(1, min(65535, int(raw)))
    except ValueError:
        return None


def _resolve_onnx_path() -> Path | None:
    raw = os.environ.get("DESKEYE_YOLO_ONNX", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _onnx_file_plausible(path: Path) -> bool:
    """yolov5n.onnx 正常约 4MB+；过小多为错误页或损坏。"""
    if not path.is_file():
        return False
    return path.stat().st_size >= 2 * 1024 * 1024


def _ensure_onnx_model() -> Path | None:
    """
    确保存在 YOLO ONNX。默认路径为 config.OBJECT_ONNX_PATH；
    缺文件时默认自动执行 scripts/download_object_model.py（可用 DESKEYE_AUTO_DOWNLOAD_YOLO=0 关闭）。
    """
    custom = _resolve_onnx_path()
    target = custom if custom is not None else config.OBJECT_ONNX_PATH
    if target.is_file() and not _onnx_file_plausible(target):
        print(f"现有 ONNX 体积异常（{target.stat().st_size} 字节），已删除并将重新获取: {target}")
        try:
            target.unlink()
        except OSError:
            pass

    if ObjectDetector.model_path_exists(custom) and _onnx_file_plausible(target):
        return target

    if custom is not None:
        print(
            f"未找到指定 ONNX 文件: {target}\n"
            "请检查 DESKEYE_YOLO_ONNX，或删除该变量后使用默认路径并运行：\n"
            "  python3 scripts/download_object_model.py"
        )
        return None

    auto = os.environ.get("DESKEYE_AUTO_DOWNLOAD_YOLO", "1").strip().lower()
    if auto not in ("0", "false", "no", "off"):
        print(f"未找到默认模型: {target}\n正在运行: python3 scripts/download_object_model.py …")
        script = ROOT / "scripts" / "download_object_model.py"
        r = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(ROOT),
            timeout=900,
        )
        if r.returncode != 0:
            print("自动下载失败（网络或权限）。请手动执行上述 download 脚本。")
            return None

    if not ObjectDetector.model_path_exists(None) or not _onnx_file_plausible(config.OBJECT_ONNX_PATH):
        print(
            f"仍未找到 YOLO ONNX: {config.OBJECT_ONNX_PATH}\n"
            "请手动运行: python3 scripts/download_object_model.py\n"
            "或设置 DESKEYE_YOLO_ONNX 指向已有 .onnx 文件。"
        )
        return None
    return config.OBJECT_ONNX_PATH


def _warn_bad_camera_name() -> None:
    cn = os.environ.get("DESKEYE_CAMERA_NAME", "").strip()
    if cn in ("...", "''", '""', ".", ".."):
        print(
            "【警告】DESKEYE_CAMERA_NAME 不能是占位符 …，请改为 rpicam-hello --list-cameras 里括号中的完整路径。\n"
            "例如: export DESKEYE_CAMERA_NAME='/base/axi/pcie@1000120000/rp1/i2c@80000/ov5647@36'"
        )


def _guess_lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return str(s.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main() -> int:
    _warn_bad_camera_name()
    onnx_path = _ensure_onnx_model()
    if onnx_path is None:
        return 1

    _maybe_auto_mjpeg_headless()
    _bootstrap_display_env()
    ensure_dir(config.DATA_DIR)
    preview_path = config.DATA_DIR / "object_preview.jpg"
    preview_every = _parse_object_preview_every()
    show = _want_show()
    http_port = _parse_preview_http_port()
    mjpeg: MjpegPreviewServer | None = None

    if http_port is not None:
        mjpeg = MjpegPreviewServer(host="0.0.0.0", port=http_port, target_fps=20)
        try:
            mjpeg.start()
        except OSError as e:
            print(f"MJPEG 预览端口 {http_port} 启动失败: {e}")
            mjpeg = None

    cam = Camera()
    if not cam.open():
        print("无法打开摄像头，参见 tests/test_camera.py。")
        if mjpeg:
            mjpeg.stop()
        return 1

    det = ObjectDetector(onnx_path=onnx_path)
    detect_every = _parse_detect_every()
    hold_sec = _parse_hold_sec()
    if show:
        try:
            cv2.startWindowThread()
        except cv2.error:
            pass
        cv2.namedWindow("DeskEye phone", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("DeskEye phone", config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

    ip_hint = _guess_lan_ip()
    print(
        "手机检测运行中（DeskEye 阶段 3）。红色框为手机检测；控制台输出 PHONE_DETECTED。\n"
        f"推理输入: {config.OBJECT_INPUT_SIZE}px，置信度阈: {config.OBJECT_CONF_THRESHOLD}，"
        f"每 {detect_every} 帧推理一次，框保持 {hold_sec}s。\n"
        f"每 {preview_every} 帧写入: {preview_path}\n"
        + (
            "已开启 OpenCV 预览窗口，按 q 退出。\n"
            if show
            else "未检测到本机图形会话：已尝试自动 MJPEG（见下）。也可 export DESKEYE_SHOW=1。\n"
        )
        + (
            f"浏览器实时预览: http://{ip_hint}:{http_port}/  （本机 http://127.0.0.1:{http_port}/）\n"
            if mjpeg
            else "手动开启浏览器预览: DESKEYE_PREVIEW_HTTP=1 python3 tests/test_object.py\n"
        )
        + "关闭自动 MJPEG: DESKEYE_AUTO_MJPEG=0\n"
        + "Ctrl+C 退出。"
    )

    frame_i = 0
    last_printed = False
    held_phones: List[PhoneDetection] = []
    hold_until = 0.0
    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                print("读帧失败")
                return 1

            if frame_i % detect_every == 0:
                new_phones = det.detect_phones(frame)
                if new_phones:
                    held_phones = new_phones
                    hold_until = time.monotonic() + hold_sec
                elif time.monotonic() >= hold_until:
                    held_phones = []

            phones = held_phones
            _draw_phones(frame, phones)

            frame_i += 1
            if frame_i % preview_every == 0:
                cv2.imwrite(str(preview_path), frame)
            if mjpeg:
                mjpeg.set_frame_bgr(frame, jpeg_quality=68)

            if phones:
                best = phones[0]
                if not last_printed:
                    print(f"PHONE_DETECTED confidence={best.confidence:.3f}")
                last_printed = True
            else:
                if last_printed:
                    print("(无手机)")
                    last_printed = False

            if show:
                cv2.imshow("DeskEye phone", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\n已退出。")
    finally:
        if mjpeg:
            mjpeg.stop()
        if show:
            cv2.destroyAllWindows()
        cam.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
