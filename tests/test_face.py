"""
阶段 2 验证：人脸检测，在画面上绘制 bbox 并标注朝向（front / left / right）。

用法（项目根）：
  python3 scripts/download_face_models.py   # 首次下载模型
  export DESKEYE_CAMERA_NAME='...'          # 树莓派建议设置
  python3 tests/test_face.py

实时预览（三选一或组合）：
  1) 本机桌面终端：直接运行，有 DISPLAY/WAYLAND 时会自动弹出 OpenCV 窗口（每帧刷新）。
  2) SSH 但画面要显示在接显示器的树莓派上：
       export DESKEYE_SHOW=1
       （脚本会在无 DISPLAY 时尝试 DISPLAY=:0 + ~/.Xauthority）
       若仍无窗口：在 Pi 的「图形界面终端」里运行同一命令，或配置 X11 权限。
  3) 任意机器用浏览器看实时流（推荐 SSH 调试用）：
       DESKEYE_PREVIEW_HTTP=1 python3 tests/test_face.py
       然后浏览器打开 http://<树莓派局域网IP>:8765/ （端口可用 DESKEYE_PREVIEW_HTTP=9000 指定）

环境变量：
  DESKEYE_SHOW=0              强制不弹 OpenCV 窗口
  DESKEYE_SHOW=1              强制尝试弹窗，并尝试绑定本地 :0
  DESKEYE_PREVIEW_HTTP=1      开启 HTTP MJPEG（默认端口 8765）
  DESKEYE_PREVIEW_HTTP=9000   同上，端口 9000
  DESKEYE_PREVIEW_HTTP_PORT=8765   与 DESKEYE_PREVIEW_HTTP=1 联用改端口
  DESKEYE_FACE_PREVIEW_EVERY=15   每 N 帧写入 data/face_preview.jpg

按 q（焦点在 OpenCV 窗口上）或 Ctrl+C 退出。
"""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

import config  # noqa: E402
from src.camera import Camera  # noqa: E402
from src.face_detector import FaceDetector  # noqa: E402
from src.mjpeg_preview import MjpegPreviewServer  # noqa: E402
from src.utils import ensure_dir  # noqa: E402

ORIENTATION_BGR = {
    "front": (0, 200, 0),
    "left": (255, 128, 0),
    "right": (0, 128, 255),
}


def _parse_preview_every() -> int:
    raw = os.environ.get("DESKEYE_FACE_PREVIEW_EVERY", "15").strip()
    try:
        n = int(raw)
        return max(1, n)
    except ValueError:
        return 15


def _has_gui_session() -> bool:
    if os.name == "nt":
        return True
    return bool(
        os.environ.get("DISPLAY", "").strip()
        or os.environ.get("WAYLAND_DISPLAY", "").strip()
    )


def _bootstrap_display_env() -> None:
    """DESKEYE_SHOW=1 且无 DISPLAY 时，尝试把窗口画到本机 X11 :0（接显示器会话）。"""
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
    """
    返回 MJPEG 监听端口；不需要 HTTP 预览时返回 None。
    DESKEYE_PREVIEW_HTTP=1 → 端口来自 DESKEYE_PREVIEW_HTTP_PORT 或 8765
    DESKEYE_PREVIEW_HTTP=9000 → 端口 9000
    """
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
        p = int(raw)
        return max(1, min(65535, p))
    except ValueError:
        return None


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
    if not FaceDetector.model_paths_exist():
        print(
            "未找到人脸模型文件。请先运行：\n"
            "  python3 scripts/download_face_models.py\n"
            f"期望路径：\n  {config.FACE_PROTO_PATH}\n  {config.FACE_CAFFEMODEL_PATH}"
        )
        return 1

    _bootstrap_display_env()
    ensure_dir(config.DATA_DIR)
    preview_path = config.DATA_DIR / "face_preview.jpg"
    preview_every = _parse_preview_every()
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
        print("无法打开摄像头，参见 tests/test_camera.py 中的说明。")
        if mjpeg:
            mjpeg.stop()
        return 1

    det = FaceDetector()
    if show:
        try:
            cv2.startWindowThread()
        except cv2.error:
            pass
        cv2.namedWindow("DeskEye face", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("DeskEye face", config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

    ip_hint = _guess_lan_ip()
    print(
        "人脸检测运行中… 脸部 bbox 上会标注 front/left/right。\n"
        f"每 {preview_every} 帧写入: {preview_path}\n"
        + (
            "已尝试开启 OpenCV 实时预览窗口（每帧 imshow）。按 q 退出。\n"
            if show
            else "当前未开启 OpenCV 窗口（无 DISPLAY/WAYLAND 或未要求弹窗）。\n"
            "可: export DESKEYE_SHOW=1 后重试；或见下方浏览器预览。\n"
        )
        + (
            f"浏览器实时预览: http://{ip_hint}:{http_port}/  （本机可试 http://127.0.0.1:{http_port}/）\n"
            if mjpeg
            else "开启浏览器预览: DESKEYE_PREVIEW_HTTP=1 python3 tests/test_face.py\n"
        )
        + "Ctrl+C 退出。"
    )

    frame_i = 0
    last_summary = ""

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                print("读帧失败")
                return 1

            faces = det.detect(frame)
            for f in faces:
                color = ORIENTATION_BGR.get(f.orientation, (200, 200, 200))
                cv2.rectangle(frame, (f.x1, f.y1), (f.x2, f.y2), color, 2)
                label = f"{f.orientation} {f.confidence:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (f.x1, max(0, f.y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            frame_i += 1
            if frame_i % preview_every == 0:
                cv2.imwrite(str(preview_path), frame)

            if mjpeg:
                mjpeg.set_frame_bgr(frame)

            if faces:
                summary = " | ".join(
                    f"{ff.orientation}({ff.confidence:.2f})" for ff in faces[:3]
                )
                if summary != last_summary:
                    print(summary)
                    last_summary = summary
            else:
                if last_summary != "":
                    print("(无人脸)")
                    last_summary = ""

            if show:
                cv2.imshow("DeskEye face", frame)
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
