"""
专注度评分（阶段 4）：综合座位、人脸朝向、手机检测计算分数。

算法 v1（DeskEye_Cursor_Prompt.md）：
- 座位有人 + 人脸正面 → +1 分/秒
- 座位有人 + 人脸侧转 / 无人脸 → 0 分/秒
- 检测到手机 → -2 分/秒（总分不低于 0）
- 座位无人 → 不加分
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import config


@dataclass(frozen=True)
class FocusInput:
    """一帧（或一秒）聚合后的检测状态。"""

    present: bool
    face_orientation: Optional[str]  # "front" | "left" | "right" | None
    phone_detected: bool
    phone_confidence: float = 0.0


@dataclass(frozen=True)
class FocusTransition:
    """状态变化事件，用于写入 focus_events。"""

    event_type: str
    confidence: float = 0.0
    notes: str = ""


@dataclass(frozen=True)
class FocusTickResult:
    """一次计分周期的结果。"""

    elapsed_sec: float
    rate_per_sec: int
    score_delta: int
    total_score: int
    input_state: FocusInput


class FocusScorer:
    """专注度评分器：维护总分、检测状态迁移、按秒结算。"""

    def __init__(
        self,
        rate_front: Optional[int] = None,
        rate_side: Optional[int] = None,
        rate_absent: Optional[int] = None,
        rate_phone: Optional[int] = None,
        score_min: Optional[int] = None,
    ) -> None:
        self._rate_front = (
            rate_front if rate_front is not None else int(config.FOCUS_RATE_FRONT)
        )
        self._rate_side = (
            rate_side if rate_side is not None else int(config.FOCUS_RATE_SIDE)
        )
        self._rate_absent = (
            rate_absent if rate_absent is not None else int(config.FOCUS_RATE_ABSENT)
        )
        self._rate_phone = (
            rate_phone if rate_phone is not None else int(config.FOCUS_RATE_PHONE)
        )
        self._score_min = (
            score_min if score_min is not None else int(config.FOCUS_SCORE_MIN)
        )

        self._total_score = 0
        self._prev_present: Optional[bool] = None
        self._prev_orientation: Optional[str] = None
        self._prev_phone: Optional[bool] = None

    @property
    def total_score(self) -> int:
        return self._total_score

    def rate_per_sec(self, state: FocusInput) -> int:
        """根据当前状态计算每秒得分变化率。"""
        if state.phone_detected:
            return self._rate_phone
        if not state.present:
            return self._rate_absent
        if state.face_orientation == "front":
            return self._rate_front
        return self._rate_side

    def detect_transitions(self, state: FocusInput) -> List[FocusTransition]:
        """对比上一状态，产出 present/absent/focused/distracted/phone_detected 事件。"""
        events: List[FocusTransition] = []

        if self._prev_present is not None and state.present != self._prev_present:
            if state.present:
                events.append(FocusTransition("present", notes="motion"))
            else:
                events.append(FocusTransition("absent", notes="motion"))

        if state.present:
            ori = state.face_orientation
            prev_ori = self._prev_orientation
            if ori == "front" and prev_ori != "front":
                events.append(FocusTransition("focused", notes=ori or ""))
            elif ori in ("left", "right") and prev_ori == "front":
                events.append(
                    FocusTransition("distracted", notes=ori or "", confidence=0.0)
                )

        if state.phone_detected and self._prev_phone is not True:
            events.append(
                FocusTransition(
                    "phone_detected",
                    confidence=state.phone_confidence,
                    notes="yolo",
                )
            )

        self._prev_present = state.present
        self._prev_orientation = state.face_orientation
        self._prev_phone = state.phone_detected
        return events

    def apply_tick(self, state: FocusInput, elapsed_sec: float) -> FocusTickResult:
        """
        按经过的秒数结算分数（先检测状态迁移，再按 rate × elapsed 更新总分）。
        elapsed_sec 通常取 1.0（每秒调用一次）。
        """
        elapsed = max(0.0, float(elapsed_sec))
        rate = self.rate_per_sec(state)
        raw_delta = int(round(rate * elapsed))
        new_score = max(self._score_min, self._total_score + raw_delta)
        actual_delta = new_score - self._total_score
        self._total_score = new_score

        return FocusTickResult(
            elapsed_sec=elapsed,
            rate_per_sec=rate,
            score_delta=actual_delta,
            total_score=self._total_score,
            input_state=state,
        )
