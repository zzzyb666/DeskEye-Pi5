"""
DeskEye 主程序（阶段 6）：专注检测守护 + Flask Web 仪表盘。

默认无头运行（适合 systemd）：仅写库与 Web；预览需显式设置环境变量。

用法（项目根）：
  export DESKEYE_CAMERA_NAME='/base/axi/pcie@.../ov5647@36'
  python3 main.py

环境变量（常用）：
  DESKEYE_CAMERA_NAME          libcamera 设备路径（必填，除非写在 config.py）
  DESKEYE_DETECT_EVERY=3       YOLO 每 N 帧推理
  DESKEYE_SHOW=1               开启 OpenCV 预览窗口
  DESKEYE_PREVIEW_HTTP=1       开启 MJPEG 实时预览（端口 8765，与 Web 5000 独立）
  DESKEYE_AUTO_MJPEG=0         生产环境建议关闭自动 MJPEG
"""
from __future__ import annotations

import gc
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("ORT_LOGGING_LEVEL", "4")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

import config  # noqa: E402
from src.camera import Camera  # noqa: E402
from src.database import DeskEyeDatabase  # noqa: E402
from src.face_detector import FaceDetection, FaceDetector  # noqa: E402
from src.focus_scorer import FocusInput, FocusScorer  # noqa: E402
from src.motion_detector import MotionDetector  # noqa: E402
from src.object_detector import ObjectDetector, PhoneDetection  # noqa: E402
from src.utils import ensure_dir  # noqa: E402
from web.app import app as flask_app  # noqa: E402

PREVIEW_EVERY_N_FRAMES = 15


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


def _parse_tick_interval() -> float:
    raw = os.environ.get("DESKEYE_FOCUS_TICK_SEC", "").strip()
    if not raw:
        return float(config.FOCUS_TICK_INTERVAL_SEC)
    try:
        return max(0.2, float(raw))
    except ValueError:
        return float(config.FOCUS_TICK_INTERVAL_SEC)


def _best_face_orientation(faces: List[FaceDetection]) -> Optional[str]:
    if not faces:
        return None
    return faces[0].orientation


def _state_label(inp: FocusInput) -> str:
    parts = ["PRESENT" if inp.present else "ABSENT"]
    if inp.face_orientation:
        parts.append(inp.face_orientation)
    else:
        parts.append("no_face")
    if inp.phone_detected:
        parts.append(f"phone({inp.phone_confidence:.2f})")
    return " ".join(parts)


def _warn_bad_camera_name() -> None:
    cn = os.environ.get("DESKEYE_CAMERA_NAME", "").strip()
    if cn in ("...", "''", '""', ".", ".."):
        print(
            "【警告】DESKEYE_CAMERA_NAME 不能是占位符，请设为 rpicam-hello --list-cameras 括号内完整路径。"
        )


def _check_models() -> bool:
    if not FaceDetector.model_paths_exist():
        print(
            "未找到人脸模型。请运行：python3 scripts/download_face_models.py\n"
            f"期望：{config.FACE_PROTO_PATH}\n      {config.FACE_CAFFEMODEL_PATH}"
        )
        return False
    if not ObjectDetector.model_path_exists():
        print(
            "未找到 YOLO ONNX。请运行：python3 scripts/download_object_model.py\n"
            f"期望：{config.OBJECT_ONNX_PATH}"
        )
        return False
    return True


def run_focus_loop(stop: threading.Event) -> None:
    """后台线程：摄像头采集、检测、计分、写库。"""
    tick_interval = _parse_tick_interval()
    detect_every = _parse_detect_every()
    hold_sec = _parse_hold_sec()

    ensure_dir(config.DATA_DIR)
    preview_path = config.DATA_DIR / "focus_preview.jpg"
    db = DeskEyeDatabase()
    cam = Camera()

    if not cam.open():
        print("【错误】无法打开摄像头，DeskEye 检测线程退出。")
        db.close()
        stop.set()
        return

    motion = MotionDetector()
    face_det = FaceDetector()
    obj_det = ObjectDetector()
    scorer = FocusScorer()

    last_tick_mono = time.monotonic()
    frame_i = 0
    held_phones: List[PhoneDetection] = []
    hold_until = 0.0
    focused_seconds = 0
    phone_ticks = 0
    distraction_count = 0

    print(
        f"DeskEye 检测线程已启动（计分 {tick_interval}s，YOLO 每 {detect_every} 帧，"
        f"数据库 {config.DATABASE_PATH}）"
    )

    try:
        while not stop.is_set():
            ok, frame = cam.read()
            if not ok or frame is None:
                print("读帧失败，1 秒后重试…")
                if stop.wait(1.0):
                    break
                continue

            if stop.is_set():
                break

            for line in motion.update(frame):
                print(line)

            faces: List[FaceDetection] = []
            if not stop.is_set():
                faces = face_det.detect(frame)

            if stop.is_set():
                break

            if frame_i % detect_every == 0 and not stop.is_set():
                new_phones = obj_det.detect_phones(frame)
                if new_phones:
                    held_phones = new_phones
                    hold_until = time.monotonic() + hold_sec
                elif time.monotonic() >= hold_until:
                    held_phones = []

            phone_conf = held_phones[0].confidence if held_phones else 0.0
            state = FocusInput(
                present=motion.is_present,
                face_orientation=_best_face_orientation(faces),
                phone_detected=bool(held_phones),
                phone_confidence=phone_conf,
            )

            now = time.monotonic()
            if now - last_tick_mono >= tick_interval:
                elapsed = now - last_tick_mono
                last_tick_mono = now

                for tr in scorer.detect_transitions(state):
                    db.insert_focus_event(
                        tr.event_type,
                        confidence=tr.confidence,
                        notes=tr.notes,
                    )
                    if tr.event_type == "distracted":
                        distraction_count += 1

                tick = scorer.apply_tick(state, elapsed)
                db.insert_focus_event(
                    "score_tick",
                    score_delta=tick.score_delta,
                    notes=_state_label(state),
                )

                if (
                    tick.rate_per_sec > 0
                    and state.present
                    and state.face_orientation == "front"
                ):
                    focused_seconds += int(round(elapsed))
                if state.phone_detected:
                    phone_ticks += 1

                print(
                    f"score={tick.total_score:4d} rate={tick.rate_per_sec:+2d}/s "
                    f"state={_state_label(state)}"
                )

            frame_i += 1
            if frame_i % PREVIEW_EVERY_N_FRAMES == 0:
                cv2.imwrite(str(preview_path), frame)
            if frame_i % max(1, int(config.MAIN_GC_EVERY_N_FRAMES)) == 0:
                gc.collect()

    except Exception as e:
        print(f"【错误】检测线程异常：{e}")
        stop.set()
    finally:
        if (
            focused_seconds > 0
            or phone_ticks > 0
            or distraction_count > 0
            or scorer.total_score > 0
        ):
            db.upsert_daily_stats(
                add_focus_seconds=focused_seconds,
                add_distraction=distraction_count,
                add_phone_detect=phone_ticks,
                score_sample=float(scorer.total_score),
            )
        cam.release()
        db.close()
        print("DeskEye 检测线程已停止。")


def run_web_server(stop: threading.Event) -> None:
    """后台线程：Flask Web 仪表盘。"""
    print(
        f"DeskEye Web 启动：http://{config.WEB_HOST}:{config.WEB_PORT}/ "
        f"（数据 {config.DATABASE_PATH}）"
    )
    try:
        flask_app.run(
            host=config.WEB_HOST,
            port=int(config.WEB_PORT),
            debug=False,
            threaded=True,
            use_reloader=False,
        )
    except Exception as e:
        print(f"【错误】Web 线程异常：{e}")
        stop.set()


def main() -> int:
    stop = threading.Event()

    def handle_signal(signum, _frame) -> None:
        print(f"\n收到信号 {signum}，正在停止 DeskEye…")
        stop.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    _warn_bad_camera_name()
    if not _check_models():
        return 1

    if not os.environ.get("DESKEYE_CAMERA_NAME", "").strip() and not config.CAMERA_LIBCAMERA_NAME:
        print(
            "【提示】未设置 DESKEYE_CAMERA_NAME，将尝试 libcamerasrc 默认设备。\n"
            "Pi 上建议在 deskeye.env 或 systemd 中配置完整 camera-name 路径。"
        )

    focus_thread = threading.Thread(
        target=run_focus_loop,
        args=(stop,),
        name="deskeye-focus",
        daemon=True,
    )
    web_thread = threading.Thread(
        target=run_web_server,
        args=(stop,),
        name="deskeye-web",
        daemon=True,
    )

    print("DeskEye 主程序启动（阶段 6）…")
    focus_thread.start()
    web_thread.start()

    try:
        while not stop.is_set():
            if not focus_thread.is_alive() or not web_thread.is_alive():
                print("某工作线程已退出，正在停止 DeskEye…")
                stop.set()
                break
            stop.wait(1.0)
    except KeyboardInterrupt:
        stop.set()

    focus_thread.join(timeout=20.0)
    print("DeskEye 已退出。")
    return 0 if not focus_thread.is_alive() else 1


if __name__ == "__main__":
    raise SystemExit(main())
