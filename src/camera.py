"""
CSI 摄像头封装：OpenCV + GStreamer（libcamerasrc），与 DeskEye 规范一致。
"""
from __future__ import annotations

import os
from typing import List, Optional

import cv2

import config


class Camera:
    """通过 GStreamer pipeline 打开 libcamera 源，读取 BGR 帧。"""

    def __init__(self, pipeline: Optional[str] = None) -> None:
        # None：使用 config 中多条候选管线依次尝试；非 None：仅该条
        self._pipelines: List[str] = (
            [pipeline] if pipeline is not None else config.get_gstreamer_pipeline_candidates()
        )
        self._pipeline: Optional[str] = None  # 实际打开成功的那条
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """打开摄像头；成功返回 True。"""
        if self._cap is not None and self._cap.isOpened():
            return True

        if config.CAMERA_SOURCE_OVERRIDE is not None:
            src = config.CAMERA_SOURCE_OVERRIDE
            self._cap = cv2.VideoCapture(src)
            self._pipeline = str(src)
            return bool(self._cap and self._cap.isOpened())

        gst_debug = os.environ.get("DESKEYE_GST_DEBUG", "").strip() in ("1", "true", "yes")

        for p in self._pipelines:
            cap = cv2.VideoCapture(p, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                if gst_debug:
                    print(f"[DeskEye] GStreamer 未打开，跳过管线片段: {p[:120]}...")
                cap.release()
                continue
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                self._cap = cap
                self._pipeline = p
                if gst_debug:
                    print(f"[DeskEye] 使用管线: {p[:160]}...")
                return True
            if gst_debug:
                print(f"[DeskEye] 已打开但读帧失败，尝试下一条...")
            cap.release()

        self._cap = None
        self._pipeline = None
        return False

    def read(self):
        """读取一帧 (ok, frame)。frame 为 BGR numpy 数组或 None。"""
        if not self._cap or not self._cap.isOpened():
            return False, None
        return self._cap.read()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        if not self.open():
            raise RuntimeError("无法打开摄像头，请检查 GStreamer pipeline 与 video 组权限")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
