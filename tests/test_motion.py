"""
阶段 1 验证：MOG2 运动检测，有人活动时输出 PRESENT，离开画面约 5 秒后输出 ABSENT。
用法（项目根）：export DESKEYE_CAMERA_NAME='...'   # Pi 上建议设置
            python3 tests/test_motion.py
Ctrl+C 结束。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.camera import Camera  # noqa: E402
from src.motion_detector import MotionDetector  # noqa: E402


def main() -> int:
    cam = Camera()
    if not cam.open():
        print("无法打开摄像头，参见 tests/test_camera.py 中的说明。")
        return 1

    det = MotionDetector()
    print("运动检测运行中… 在画面前走动应出现 PRESENT，静止离开约 5 秒后出现 ABSENT。Ctrl+C 退出。")

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                print("读帧失败")
                return 1
            for line in det.update(frame):
                print(line)
    except KeyboardInterrupt:
        print("\n已退出。")
    finally:
        cam.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
