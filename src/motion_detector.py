"""
运动检测：MOG2 背景减除 + 轮廓，用于人物是否在画面前（阶段 1）。
"""
from __future__ import annotations

import time
from typing import List, Optional

import cv2
import numpy as np

import config


class MotionDetector:
    """
    基于 MOG2 的前景面积判断「有人活动」；连续无运动超过 ABSENT_SECONDS 视为离开。
    """

    def __init__(
        self,
        history: Optional[int] = None,
        var_threshold: Optional[float] = None,
        absent_seconds: Optional[float] = None,
        min_area: Optional[int] = None,
        warmup_frames: int = 45,
    ) -> None:
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=history if history is not None else config.MOG2_HISTORY,
            varThreshold=var_threshold
            if var_threshold is not None
            else float(config.MOG2_VAR_THRESHOLD),
            detectShadows=False,
        )
        self._absent_seconds = (
            absent_seconds
            if absent_seconds is not None
            else float(config.ABSENT_SECONDS)
        )
        self._min_area = min_area if min_area is not None else config.MOTION_MIN_AREA
        self._warmup_frames = warmup_frames
        self._frame_index = 0

        self._present = False
        self._last_motion_mono: float = 0.0
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    @property
    def is_present(self) -> bool:
        """当前是否判定为「有人在画面前」。"""
        return self._present

    def _foreground_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        fg = self._bg.apply(frame_bgr)
        _, fg_bin = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg_bin = cv2.morphologyEx(fg_bin, cv2.MORPH_OPEN, self._kernel, iterations=1)
        return fg_bin

    def _motion_area(self, mask: np.ndarray) -> int:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        total = 0
        for c in contours:
            a = int(cv2.contourArea(c))
            if a > 200:
                total += a
        return total

    def update(self, frame_bgr: np.ndarray) -> List[str]:
        """
        处理一帧 BGR 图像，返回本帧应打印的状态行（如 "PRESENT" / "ABSENT"），无变化则 []。
        """
        self._frame_index += 1
        now = time.monotonic()

        mask = self._foreground_mask(frame_bgr)
        area = self._motion_area(mask)
        motion = area >= self._min_area

        messages: List[str] = []

        if self._frame_index <= self._warmup_frames:
            if motion:
                self._last_motion_mono = now
            return messages

        if motion:
            self._last_motion_mono = now
            if not self._present:
                self._present = True
                messages.append("PRESENT")
            return messages

        if self._present and (now - self._last_motion_mono) >= self._absent_seconds:
            self._present = False
            messages.append("ABSENT")

        return messages
