"""
人脸检测与粗略朝向估计（阶段 2）。

使用 OpenCV DNN `readNetFromCaffe` 加载 Res10 SSD（300×300）。
朝向：SSD 仅 bbox；用脸框中心相对画幅中心的横向偏移作为粗略 yaw 代理（规范基线）。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

import config


@dataclass(frozen=True)
class FaceDetection:
    """单张人脸检测结果。"""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    orientation: str  # "front" | "left" | "right"


def _orientation_from_bbox(
    cx: float,
    frame_w: int,
    offset_threshold: float,
) -> str:
    """脸中心 x 相对画幅中心的归一化偏移：偏右→left，偏左→right。"""
    if frame_w <= 0:
        return "front"
    center = frame_w * 0.5
    rel = (cx - center) / max(center, 1e-3)
    if rel > offset_threshold:
        return "left"
    if rel < -offset_threshold:
        return "right"
    return "front"


class FaceDetector:
    """Res10 SSD 人脸检测 + bbox 级朝向启发式。"""

    def __init__(
        self,
        proto_path: Optional[Path] = None,
        caffemodel_path: Optional[Path] = None,
        conf_threshold: Optional[float] = None,
        input_size: Optional[int] = None,
        yaw_offset_threshold: Optional[float] = None,
        min_box_area: Optional[int] = None,
    ) -> None:
        self._proto = proto_path if proto_path is not None else config.FACE_PROTO_PATH
        self._caffe = (
            caffemodel_path if caffemodel_path is not None else config.FACE_CAFFEMODEL_PATH
        )
        self._conf = (
            conf_threshold
            if conf_threshold is not None
            else float(config.FACE_CONFIDENCE_THRESHOLD)
        )
        self._in_size = input_size if input_size is not None else int(config.FACE_INPUT_SIZE)
        self._yaw_thr = (
            yaw_offset_threshold
            if yaw_offset_threshold is not None
            else float(config.FACE_YAW_OFFSET_THRESHOLD)
        )
        self._min_area = (
            min_box_area if min_box_area is not None else int(config.FACE_MIN_BOX_AREA)
        )
        self._net = cv2.dnn.readNetFromCaffe(str(self._proto), str(self._caffe))

    @staticmethod
    def model_paths_exist(
        proto_path: Optional[Path] = None,
        caffemodel_path: Optional[Path] = None,
    ) -> bool:
        p = proto_path if proto_path is not None else config.FACE_PROTO_PATH
        c = caffemodel_path if caffemodel_path is not None else config.FACE_CAFFEMODEL_PATH
        return p.is_file() and c.is_file()

    def detect(self, frame_bgr: np.ndarray) -> List[FaceDetection]:
        """检测一帧中的所有人脸，按置信度从高到低排序。"""
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame_bgr, (self._in_size, self._in_size)),
            1.0,
            (self._in_size, self._in_size),
            (104.0, 117.0, 123.0),
        )
        self._net.setInput(blob)
        out = self._net.forward()

        faces: List[FaceDetection] = []
        if out is None or out.size == 0:
            return faces

        if out.ndim == 4:
            dets = out[0, 0]
        elif out.ndim == 3:
            dets = out[0]
        else:
            return faces

        for i in range(dets.shape[0]):
            conf = float(dets[i, 2])
            if conf < self._conf:
                continue
            x1 = int(dets[i, 3] * w)
            y1 = int(dets[i, 4] * h)
            x2 = int(dets[i, 5] * w)
            y2 = int(dets[i, 6] * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            area = (x2 - x1) * (y2 - y1)
            if area < self._min_area:
                continue
            cx = (x1 + x2) * 0.5
            ori = _orientation_from_bbox(cx, w, self._yaw_thr)
            faces.append(
                FaceDetection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=conf,
                    orientation=ori,
                )
            )

        faces.sort(key=lambda f: f.confidence, reverse=True)
        return faces
