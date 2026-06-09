"""
手机 / 干扰物检测（阶段 3）：YOLOv5n ONNX + ONNX Runtime。

后处理针对 Ultralytics 导出的单输出张量 [1, N, 85]（4 框 + 1 obj + 80 类）。
仅保留 COCO「cell phone」类（默认 class id = 67）；用手机类分数而非 argmax，
避免画面里有人时 top 类为 person 而漏检手机。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

import config
from src.utils import letterbox_bgr, xyxy640_to_original


def _resolve_input_size_from_onnx(session: ort.InferenceSession, fallback: int) -> int:
    """读取 ONNX 固定输入高宽（如 [1,3,640,640]）；动态维则退回 config/fallback。"""
    shape = session.get_inputs()[0].shape
    if len(shape) < 4:
        return fallback
    h, w = shape[2], shape[3]
    if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
        if h != w:
            return max(h, w)
        return h
    return fallback


def _ort_input_np_dtype(session: ort.InferenceSession) -> type:
    meta = session.get_inputs()[0]
    s = (getattr(meta, "type", "") or "").lower()
    if "float16" in s or "fp16" in s:
        return np.float16
    if "float64" in s or "double" in s:
        return np.float64
    return np.float32


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))


def _activate_obj_and_cls(obj_raw: np.ndarray, cls_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """若 ONNX 已输出概率则不再 sigmoid（避免双重激活压低分数）。"""
    if (
        obj_raw.size > 0
        and float(np.min(obj_raw)) >= 0.0
        and float(np.max(obj_raw)) <= 1.0
        and float(np.min(cls_raw)) >= 0.0
        and float(np.max(cls_raw)) <= 1.0
    ):
        return obj_raw.astype(np.float32), cls_raw.astype(np.float32)
    return _sigmoid(obj_raw), _sigmoid(cls_raw)


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + bb - inter
    return float(inter / union) if union > 0 else 0.0


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = np.array([_iou_xyxy(boxes[i], boxes[j]) for j in rest])
        order = rest[ious < iou_thr]
    return keep


@dataclass(frozen=True)
class PhoneDetection:
    """原图坐标系下的手机框。"""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


class ObjectDetector:
    """YOLOv5 ONNX：仅输出手机（cell phone）检测。"""

    def __init__(
        self,
        onnx_path: Optional[Path] = None,
        input_size: Optional[int] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        phone_class_id: Optional[int] = None,
        ort_intra_threads: Optional[int] = None,
    ) -> None:
        self._path = onnx_path if onnx_path is not None else config.OBJECT_ONNX_PATH
        self._size = int(input_size if input_size is not None else config.OBJECT_INPUT_SIZE)
        self._conf = float(
            conf_threshold if conf_threshold is not None else config.OBJECT_CONF_THRESHOLD
        )
        self._iou = float(
            iou_threshold if iou_threshold is not None else config.OBJECT_IOU_THRESHOLD
        )
        self._phone_cls = int(
            phone_class_id if phone_class_id is not None else config.OBJECT_PHONE_CLASS_ID
        )
        threads = int(
            ort_intra_threads
            if ort_intra_threads is not None
            else config.OBJECT_ORT_INTRA_THREADS
        )
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = max(1, threads)
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ort.get_available_providers()
        self._session = ort.InferenceSession(
            str(self._path),
            sess_options=sess_opts,
            providers=providers,
        )
        self._in_name = self._session.get_inputs()[0].name
        self._out_names = [o.name for o in self._session.get_outputs()]
        self._input_np_dtype = _ort_input_np_dtype(self._session)
        model_size = _resolve_input_size_from_onnx(self._session, self._size)
        if model_size != self._size:
            import warnings

            warnings.warn(
                f"ONNX 要求输入 {model_size}x{model_size}，已覆盖 config OBJECT_INPUT_SIZE={self._size}",
                stacklevel=2,
            )
            self._size = model_size

    @staticmethod
    def model_path_exists(onnx_path: Optional[Path] = None) -> bool:
        p = onnx_path if onnx_path is not None else config.OBJECT_ONNX_PATH
        return p.is_file()

    def detect_phones(self, frame_bgr: np.ndarray) -> List[PhoneDetection]:
        """检测一帧中的手机，NMS 后按置信度降序。"""
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        oh, ow = frame_bgr.shape[:2]

        lb, ratio, (pad_w, pad_h) = letterbox_bgr(frame_bgr, self._size, self._size)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
        batch = np.expand_dims(chw, axis=0).astype(self._input_np_dtype, copy=False)

        outs = self._session.run(self._out_names, {self._in_name: batch})
        pred = np.asarray(outs[0], dtype=np.float32)
        if pred.ndim == 3 and pred.shape[1] == 85:
            pred = np.transpose(pred, (0, 2, 1))
        if pred.ndim != 3 or pred.shape[-1] != 85:
            raise ValueError(f"不支持的 ONNX 输出形状: {pred.shape}，期望末维 85")

        rows = pred[0]
        coords = rows[:, 0:4].astype(np.float32)
        if float(np.max(np.abs(coords))) <= 2.5:
            coords = coords * float(self._size)

        obj, cls_prob = _activate_obj_and_cls(rows[:, 4], rows[:, 5:85])
        phone_scores = obj * cls_prob[:, self._phone_cls]
        valid = phone_scores >= self._conf
        if not np.any(valid):
            return []

        idx = np.where(valid)[0]
        scores = phone_scores[idx]
        if len(idx) > 400:
            top = np.argpartition(-scores, 400)[:400]
            idx = idx[top]
            scores = scores[top]

        xc = coords[idx, 0]
        yc = coords[idx, 1]
        bw = coords[idx, 2]
        bh = coords[idx, 3]
        x1 = np.clip(xc - bw * 0.5, 0, self._size - 1)
        y1 = np.clip(yc - bh * 0.5, 0, self._size - 1)
        x2 = np.clip(xc + bw * 0.5, 0, self._size - 1)
        y2 = np.clip(yc + bh * 0.5, 0, self._size - 1)
        ok = (x2 > x1) & (y2 > y1)
        if not np.any(ok):
            return []
        idx = idx[ok]
        scores = scores[ok]
        x1, y1, x2, y2 = x1[ok], y1[ok], x2[ok], y2[ok]

        b640 = np.stack([x1, y1, x2, y2], axis=1)
        keep = _nms_xyxy(b640, scores, self._iou)

        results: List[PhoneDetection] = []
        for k in keep:
            ox1, oy1, ox2, oy2 = xyxy640_to_original(
                float(b640[k, 0]),
                float(b640[k, 1]),
                float(b640[k, 2]),
                float(b640[k, 3]),
                ratio,
                pad_w,
                pad_h,
                ow,
                oh,
            )
            results.append(
                PhoneDetection(
                    x1=ox1,
                    y1=oy1,
                    x2=ox2,
                    y2=oy2,
                    confidence=float(scores[k]),
                )
            )
        results.sort(key=lambda p: p.confidence, reverse=True)
        return results
