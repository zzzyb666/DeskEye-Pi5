"""
阶段 4 验证：专注度评分 + SQLite 持久化 + 实时预览。

综合运动检测（PRESENT/ABSENT）、人脸朝向、手机检测，按秒计分并写入 data/deskeye.db。
画面上绘制人脸框（按朝向着色）与手机红框，叠加 score/rate 状态。

用法（项目根）：
  python3 scripts/download_face_models.py
  python3 scripts/download_object_model.py
  export DESKEYE_CAMERA_NAME='/base/axi/pcie@.../ov5647@36'
  python3 tests/test_focus.py

环境变量：
  DESKEYE_FOCUS_DURATION=0      运行秒数（默认 0=无限，Ctrl+C 或 q 退出；设 60 可自动结束）
  DESKEYE_DETECT_EVERY=3        手机 YOLO 每 N 帧推理一次
  DESKEYE_PREVIEW_HTTP=1        浏览器 MJPEG 预览（端口 8765）
  DESKEYE_SHOW=1                强制 OpenCV 窗口（SSH 可配合 DISPLAY=:0）
  DESKEYE_AUTO_MJPEG=0          关闭 SSH 无 DISPLAY 时自动 MJPEG

退出：Ctrl+C、按 q（OpenCV 窗口焦点）、或 DESKEYE_FOCUS_DURATION 到期。
"""
from __future__ import annotations

import importlib.util
import os
import signal
import socket
import sys
import time
from pathlib import Path
from typing import List, Optional

# 须在 import onnxruntime 之前
os.environ.setdefault("ORT_LOGGING_LEVEL", "4")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

import config  # noqa: E402
from src.camera import Camera  # noqa: E402
from src.database import DeskEyeDatabase  # noqa: E402
from src.face_detector import FaceDetection, FaceDetector  # noqa: E402
from src.focus_scorer import FocusInput, FocusScorer, FocusTickResult  # noqa: E402
from src.mjpeg_preview import MjpegPreviewServer  # noqa: E402
from src.motion_detector import MotionDetector  # noqa: E402
from src.object_detector import ObjectDetector, PhoneDetection  # noqa: E402
from src.utils import ensure_dir  # noqa: E402

# 复用 test_object 的 ONNX 引导
_to_spec = importlib.util.spec_from_file_location(
    "deskeye_test_object", ROOT / "tests" / "test_object.py"
)
_to_mod = importlib.util.module_from_spec(_to_spec)
assert _to_spec.loader is not None
_to_spec.loader.exec_module(_to_mod)
_ensure_onnx_model = _to_mod._ensure_onnx_model
_parse_detect_every = _to_mod._parse_detect_every
_parse_hold_sec = _to_mod._parse_hold_sec
_warn_bad_camera_name = _to_mod._warn_bad_camera_name
_draw_phones = _to_mod._draw_phones

ORIENTATION_BGR = {
    "front": (0, 200, 0),
    "left": (255, 128, 0),
    "right": (0, 128, 255),
}
PHONE_BGR = (0, 0, 255)
HUD_BGR = (240, 240, 240)

_stop_requested = False


def _request_stop(*_args) -> None:
    global _stop_requested
    _stop_requested = True


def _parse_duration_sec() -> Optional[float]:
    raw = os.environ.get("DESKEYE_FOCUS_DURATION", "0").strip()
    try:
        sec = float(raw)
    except ValueError:
        sec = 0.0
    if sec <= 0:
        return None
    return sec


def _parse_tick_interval() -> float:
    raw = os.environ.get("DESKEYE_FOCUS_TICK_SEC", "").strip()
    if not raw:
        return float(config.FOCUS_TICK_INTERVAL_SEC)
    try:
        return max(0.2, float(raw))
    except ValueError:
        return float(config.FOCUS_TICK_INTERVAL_SEC)


def _parse_focus_preview_every() -> int:
    raw = os.environ.get("DESKEYE_FOCUS_PREVIEW_EVERY", "15").strip()
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


def _guess_lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return str(s.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


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


def _draw_faces(frame, faces: List[FaceDetection]) -> None:
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


def _draw_hud(
    frame,
    tick: Optional[FocusTickResult],
    state: FocusInput,
) -> None:
    lines = [
        f"score={tick.total_score if tick else 0:4d}  "
        f"rate={(tick.rate_per_sec if tick else 0):+2d}/s",
        _state_label(state),
    ]
    y = 22
    for line in lines:
        cv2.putText(
            frame,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            HUD_BGR,
            1,
            cv2.LINE_AA,
        )
        y += 22


def _print_db_summary(db: DeskEyeDatabase) -> None:
    print(f"\n数据库: {db.path}")
    events = db.get_recent_events(limit=15)
    if not events:
        print("（尚无 focus_events 记录）")
    else:
        print("最近 focus_events（最新在上）:")
        for e in reversed(events):
            delta = f" delta={e.score_delta:+d}" if e.score_delta else ""
            conf = f" conf={e.confidence:.2f}" if e.confidence else ""
            note = f" ({e.notes})" if e.notes else ""
            print(f"  [{e.timestamp}] {e.event_type}{delta}{conf}{note}")

    today = db.get_today_stats()
    if today:
        print(
            f"今日 daily_stats: 专注秒={today.total_focus_seconds} "
            f"分心次={today.distraction_count} "
            f"手机次={today.phone_detect_count} "
            f"均分={today.avg_score:.1f}"
        )
    else:
        print("今日 daily_stats: （尚无记录）")


def main() -> int:
    global _stop_requested

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    _warn_bad_camera_name()
    _maybe_auto_mjpeg_headless()
    _bootstrap_display_env()

    if not FaceDetector.model_paths_exist():
        print(
            "未找到人脸模型。请先运行：\n"
            "  python3 scripts/download_face_models.py"
        )
        return 1

    onnx_path = _ensure_onnx_model()
    if onnx_path is None:
        return 1

    duration = _parse_duration_sec()
    tick_interval = _parse_tick_interval()
    detect_every = _parse_detect_every()
    hold_sec = _parse_hold_sec()
    preview_every = _parse_focus_preview_every()
    show = _want_show()
    http_port = _parse_preview_http_port()

    ensure_dir(config.DATA_DIR)
    preview_path = config.DATA_DIR / "focus_preview.jpg"
    db = DeskEyeDatabase()
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
        db.close()
        return 1

    motion = MotionDetector()
    face_det = FaceDetector()
    obj_det = ObjectDetector(onnx_path=onnx_path)
    scorer = FocusScorer()

    if show:
        try:
            cv2.startWindowThread()
        except cv2.error:
            pass
        cv2.namedWindow("DeskEye focus", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("DeskEye focus", config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

    ip_hint = _guess_lan_ip()
    print(
        "专注度评分运行中（DeskEye 阶段 4）。人脸框按朝向着色，手机红框。\n"
        f"计分间隔: {tick_interval}s；YOLO 每 {detect_every} 帧；"
        f"手机框保持 {hold_sec}s。\n"
        f"数据库: {config.DATABASE_PATH}\n"
        f"每 {preview_every} 帧写入: {preview_path}\n"
        + (
            f"将运行 {duration:.0f} 秒后自动结束。\n"
            if duration is not None
            else "默认无限运行：Ctrl+C 或按 q 退出。\n"
        )
        + (
            "已开启 OpenCV 预览窗口，按 q 退出。\n"
            if show
            else "未检测到本机图形会话：已尝试自动 MJPEG（见下）。也可 export DESKEYE_SHOW=1。\n"
        )
        + (
            f"浏览器实时预览: http://{ip_hint}:{http_port}/  （本机 http://127.0.0.1:{http_port}/）\n"
            if mjpeg
            else "手动开启浏览器预览: DESKEYE_PREVIEW_HTTP=1 python3 tests/test_focus.py\n"
        )
        + "关闭自动 MJPEG: DESKEYE_AUTO_MJPEG=0\n"
        + "每秒输出: score=总分 rate=本秒变化率 state=当前状态"
    )

    start_mono = time.monotonic()
    last_tick_mono = start_mono
    frame_i = 0
    held_phones: List[PhoneDetection] = []
    hold_until = 0.0
    focused_seconds = 0
    phone_ticks = 0
    distraction_count = 0
    last_tick: Optional[FocusTickResult] = None

    try:
        while not _stop_requested:
            ok, frame = cam.read()
            if not ok or frame is None:
                print("读帧失败")
                return 1

            if _stop_requested:
                break

            for line in motion.update(frame):
                print(line)

            faces: List[FaceDetection] = []
            if not _stop_requested:
                faces = face_det.detect(frame)

            if _stop_requested:
                break

            if frame_i % detect_every == 0 and not _stop_requested:
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

                last_tick = scorer.apply_tick(state, elapsed)
                db.insert_focus_event(
                    "score_tick",
                    score_delta=last_tick.score_delta,
                    notes=_state_label(state),
                )

                if (
                    last_tick.rate_per_sec > 0
                    and state.present
                    and state.face_orientation == "front"
                ):
                    focused_seconds += int(round(elapsed))
                if state.phone_detected:
                    phone_ticks += 1

                print(
                    f"score={last_tick.total_score:4d} rate={last_tick.rate_per_sec:+2d}/s "
                    f"state={_state_label(state)}"
                )

            _draw_faces(frame, faces)
            _draw_phones(frame, held_phones)
            _draw_hud(frame, last_tick, state)

            frame_i += 1
            if frame_i % preview_every == 0:
                cv2.imwrite(str(preview_path), frame)
            if mjpeg:
                mjpeg.set_frame_bgr(frame, jpeg_quality=68)

            if show:
                cv2.imshow("DeskEye focus", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    _stop_requested = True
                    break

            if duration is not None and (now - start_mono) >= duration:
                break

    except KeyboardInterrupt:
        _stop_requested = True
        print("\n已中断（KeyboardInterrupt）。")
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
        if mjpeg:
            mjpeg.stop()
        if show:
            cv2.destroyAllWindows()
        cam.release()
        _print_db_summary(db)
        db.close()
        if _stop_requested:
            print("已退出。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
