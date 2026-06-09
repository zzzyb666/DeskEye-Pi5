"""
DeskEye 全局配置：摄像头、路径、阈值（后续阶段逐步补充）。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

# 项目根目录（config.py 所在目录）
PROJECT_ROOT = Path(__file__).resolve().parent

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
WEB_DIR = PROJECT_ROOT / "web"

# 摄像头输出（与规范一致）
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15

# 若 OpenCV 读帧报 GStreamer caps / internal data stream error：
# 1) 安装 gstreamer1.0-libcamera；2) 在此填入 rpicam-hello --list-cameras 里括号内完整路径；
#    或通过环境变量 DESKEYE_CAMERA_NAME 指定（优先级高于本变量）。
# OV5647 + Pi5 示例：/base/axi/pcie@1000120000/rp1/i2c@80000/ov5647@36
CAMERA_LIBCAMERA_NAME: Optional[str] = None


def _resolved_libcamera_device_name() -> Optional[str]:
    env = os.environ.get("DESKEYE_CAMERA_NAME", "").strip()
    if env:
        return env
    return CAMERA_LIBCAMERA_NAME


def _libcamerasrc_prefixes() -> List[str]:
    """返回要尝试的 libcamerasrc 前缀（先带 camera-name，再试默认首路）。"""
    name = _resolved_libcamera_device_name()
    if name:
        return [
            f'libcamerasrc camera-name="{name}"',
            "libcamerasrc",
        ]
    return ["libcamerasrc"]


def get_gstreamer_pipeline_candidates() -> List[str]:
    """
    多条 GStreamer 管线，按顺序尝试。

    Pi 5 + PiSP + OV5647 常见为 libcamera 先配成 1296×972，再经 ISP 输出 NV12；
    若 libcamerasrc 后无明确 caps，易出现 Internal data stream error / pipeline 无法 PLAYING。
    """
    bgr_tail = (
        "videoconvert ! videoscale ! "
        f"video/x-raw,width={CAMERA_WIDTH},height={CAMERA_HEIGHT},format=BGR ! "
        "appsink drop=true sync=false max-buffers=1"
    )
    appsink = "appsink drop=true sync=false max-buffers=1"

    candidates: List[str] = []
    seen: set[str] = set()

    def add(p: str) -> None:
        if p not in seen:
            seen.add(p)
            candidates.append(p)

    for src in _libcamerasrc_prefixes():
        # 与传感器常见配置一致：1296×972 NV12（见 libcamera 配置日志）
        add(f"{src} ! video/x-raw,width=1296,height=972,format=NV12 ! queue ! {bgr_tail}")
        add(
            f"{src} ! video/x-raw,width=1296,height=972,framerate=46/1,format=NV12 ! "
            f"queue ! {bgr_tail}"
        )
        # 双 queue（部分 OpenCV/GStreamer 组合更稳）
        add(
            f"{src} ! video/x-raw,width=1296,height=972,format=NV12 ! queue ! "
            f"videoconvert ! videoscale ! queue ! "
            f"video/x-raw,width={CAMERA_WIDTH},height={CAMERA_HEIGHT},format=BGR ! {appsink}"
        )
        # 传感器原生 640×480 NV12
        add(
            f"{src} ! video/x-raw,width=640,height=480,format=NV12 ! queue ! videoconvert ! "
            f"video/x-raw,width={CAMERA_WIDTH},height={CAMERA_HEIGHT},format=BGR ! {appsink}"
        )
        # RGBx 再转 BGR（论坛常见写法）
        add(f"{src} ! video/x-raw,width=1296,height=972,format=RGBx ! queue ! {bgr_tail}")
        # 不在首段强绑格式，由 libcamera 协商（旧默认）
        add(
            f"{src} ! queue ! videoconvert ! videoscale ! "
            f"video/x-raw,width={CAMERA_WIDTH},height={CAMERA_HEIGHT},format=BGR ! {appsink}"
        )

    return candidates


# 兼容旧代码：默认取候选列表第一条
CAMERA_PIPELINE = get_gstreamer_pipeline_candidates()[0]

# 文档/对照用：DeskEye 提示词中的 NV12 640×480 模板
CAMERA_PIPELINE_NV12_FORCED = (
    "libcamerasrc ! "
    "video/x-raw, width=640, height=480, framerate=15/1, format=NV12 ! "
    "queue ! videoconvert ! videoscale ! "
    "video/x-raw, width=640, height=480, format=BGR ! "
    "appsink drop=true"
)

# 可选：用文件或索引摄像头做离线调试（仅开发机；生产留 None 使用 CSI pipeline）
# 例如 Windows 调试可设为 0 或 "path/to/video.mp4"
CAMERA_SOURCE_OVERRIDE = None

# 阶段 1：运动 / 人物存在（MOG2）
MOG2_HISTORY = 500
MOG2_VAR_THRESHOLD = 16
ABSENT_SECONDS = 5
# 前景轮廓面积阈值（640×480 下人体走动通常远大于此，可按环境调大减小误报）
MOTION_MIN_AREA = 4000

# 阶段 2：人脸检测（OpenCV DNN Caffe Res10 SSD 300×300）
# 模型文件由 scripts/download_face_models.py 下载到 models/
FACE_PROTO_PATH = MODELS_DIR / "res10_300x300_ssd_deploy.prototxt"
FACE_CAFFEMODEL_PATH = MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
FACE_INPUT_SIZE = 300
# 侧脸召回：规范建议可降到 0.3
FACE_CONFIDENCE_THRESHOLD = 0.3
# 脸框中心相对画幅中心的归一化横向偏移超过此值则判 left/right（基线启发式，未做朝向微调）
FACE_YAW_OFFSET_THRESHOLD = 0.12
# 过滤过小的误检（像素面积）
FACE_MIN_BOX_AREA = 900

# 阶段 3：手机检测（YOLOv5n ONNX + ONNX Runtime；规范要求 INT8，默认脚本拉取官方 FP32 作开发基线，
# 生产请自行替换为 yolov5n INT8 并同名放在 OBJECT_ONNX_PATH）
OBJECT_ONNX_PATH = MODELS_DIR / "yolov5n.onnx"
# 推理输入边长：须与 ONNX 图一致。Ultralytics release 的 yolov5n.onnx 固定 640×640（不可改 320）
# ObjectDetector 会从模型元数据自动校正；此处默认 640。降负载请增大 OBJECT_DETECT_EVERY_N
OBJECT_INPUT_SIZE = 640
OBJECT_CONF_THRESHOLD = 0.25
OBJECT_IOU_THRESHOLD = 0.45
# Ultralytics COCO80 中「cell phone」类别索引
OBJECT_PHONE_CLASS_ID = 67
# test_object：每 N 帧跑一次 YOLO，中间帧复用上一结果画框（减轻 640 推理卡顿）
OBJECT_DETECT_EVERY_N = 3
# 检测到手机后，框与 PHONE_DETECTED 状态至少保持的秒数（抑制闪烁）
OBJECT_DISPLAY_HOLD_SEC = 1.2
# ONNX Runtime CPU 线程（Pi 5 四核，2～4 可按负载调整）
OBJECT_ORT_INTRA_THREADS = 2

# 阶段 4：专注度评分 + SQLite
DATABASE_PATH = DATA_DIR / "deskeye.db"
# 专注度算法 v1：每秒得分变化（座位无人不加分；手机检测优先扣分，最低 0）
FOCUS_RATE_FRONT = 1
FOCUS_RATE_SIDE = 0
FOCUS_RATE_ABSENT = 0
FOCUS_RATE_PHONE = -2
FOCUS_SCORE_MIN = 0
# test_focus 默认每秒结算一次分数并写入数据库
FOCUS_TICK_INTERVAL_SEC = 1.0

# 阶段 5：Flask Web 仪表盘
WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
# 仪表盘页面自动刷新间隔（秒）
WEB_REFRESH_SEC = 30

# 阶段 6：主程序内存回收（规范：长时间运行每 N 帧 gc.collect）
MAIN_GC_EVERY_N_FRAMES = 1000
