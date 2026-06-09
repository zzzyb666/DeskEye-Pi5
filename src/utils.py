"""通用工具函数。"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def project_root() -> Path:
    """返回项目根目录（包含 config.py 的目录）。"""
    return Path(__file__).resolve().parent.parent


def ensure_dir(path: Path) -> Path:
    """确保目录存在并返回该路径。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def assert_bgr_frame_shape(
    frame: np.ndarray,
    width: int,
    height: int,
) -> Tuple[int, int, int]:
    """校验 BGR 帧形状，返回 (H, W, C)。"""
    if frame is None or frame.size == 0:
        raise ValueError("空帧")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"期望 BGR 三通道图像，得到 shape={frame.shape}")
    h, w = frame.shape[0], frame.shape[1]
    if h != height or w != width:
        raise ValueError(f"期望 {width}x{height}，实际 {w}x{h}")
    return h, w, 3


def letterbox_bgr(
    img_bgr: np.ndarray,
    new_w: int = 640,
    new_h: int = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    保持宽高比缩放并居中填充到 new_w×new_h。
    返回 (letterboxed_bgr, ratio, (pad_w, pad_h))；ratio = min(new_w/w0, new_h/h0)。
    """
    h0, w0 = img_bgr.shape[:2]
    r = min(new_w / w0, new_h / h0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w = (new_w - nw) // 2
    pad_h = (new_h - nh) // 2
    out = np.full((new_h, new_w, 3), color, dtype=np.uint8)
    out[pad_h : pad_h + nh, pad_w : pad_w + nw] = resized
    return out, r, (pad_w, pad_h)


def xyxy640_to_original(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    ratio: float,
    pad_w: int,
    pad_h: int,
    orig_w: int,
    orig_h: int,
) -> Tuple[int, int, int, int]:
    """将 letterbox 后 640 空间中的 xyxy 映射回原图。"""
    x1 = (x1 - pad_w) / ratio
    y1 = (y1 - pad_h) / ratio
    x2 = (x2 - pad_w) / ratio
    y2 = (y2 - pad_h) / ratio
    x1 = int(max(0, min(orig_w - 1, round(x1))))
    y1 = int(max(0, min(orig_h - 1, round(y1))))
    x2 = int(max(0, min(orig_w - 1, round(x2))))
    y2 = int(max(0, min(orig_h - 1, round(y2))))
    if x2 <= x1:
        x2 = min(orig_w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(orig_h - 1, y1 + 1)
    return x1, y1, x2, y2
