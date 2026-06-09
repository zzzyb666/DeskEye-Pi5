"""
SQLite 数据持久化（阶段 4）：专注事件与每日统计。
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import config
from src.utils import ensure_dir


@dataclass(frozen=True)
class FocusEventRow:
    """focus_events 表一行。"""

    id: int
    timestamp: str
    event_type: str
    confidence: float
    score_delta: int
    notes: str


@dataclass(frozen=True)
class ScorePoint:
    """分数趋势图上的一个点。"""

    time: str
    score: int


@dataclass(frozen=True)
class TodayDashboard:
    """Web 仪表盘今日数据汇总。"""

    date: str
    total_focus_seconds: int
    distraction_count: int
    phone_detect_count: int
    avg_score: float
    score_series: List[ScorePoint]


@dataclass(frozen=True)
class DailyStatsRow:
    """daily_stats 表一行。"""

    date: str
    total_focus_seconds: int
    distraction_count: int
    avg_score: float
    phone_detect_count: int


class DeskEyeDatabase:
    """DeskEye SQLite 封装：建表、写事件、更新每日统计。"""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._path = db_path if db_path is not None else config.DATABASE_PATH
        ensure_dir(self._path.parent)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.init_schema()

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        self._conn.close()

    def init_schema(self) -> None:
        """创建规范要求的表结构。"""
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS focus_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT,
                confidence REAL,
                score_delta INTEGER,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS daily_stats (
                date TEXT PRIMARY KEY,
                total_focus_seconds INTEGER DEFAULT 0,
                distraction_count INTEGER DEFAULT 0,
                avg_score REAL DEFAULT 0,
                phone_detect_count INTEGER DEFAULT 0
            );
            """
        )
        self._conn.commit()

    def insert_focus_event(
        self,
        event_type: str,
        confidence: float = 0.0,
        score_delta: int = 0,
        notes: str = "",
        timestamp: Optional[datetime] = None,
    ) -> int:
        """写入一条专注事件，返回新行 id。"""
        ts = timestamp.isoformat(sep=" ", timespec="seconds") if timestamp else None
        if ts:
            cur = self._conn.execute(
                """
                INSERT INTO focus_events (timestamp, event_type, confidence, score_delta, notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ts, event_type, confidence, score_delta, notes),
            )
        else:
            cur = self._conn.execute(
                """
                INSERT INTO focus_events (event_type, confidence, score_delta, notes)
                VALUES (?, ?, ?, ?)
                """,
                (event_type, confidence, score_delta, notes),
            )
        self._conn.commit()
        return int(cur.lastrowid)

    def get_recent_events(self, limit: int = 20) -> List[FocusEventRow]:
        """按时间倒序读取最近事件。"""
        cur = self._conn.execute(
            """
            SELECT id, timestamp, event_type, confidence, score_delta, notes
            FROM focus_events
            ORDER BY id DESC
            LIMIT ?
            """,
            (max(1, limit),),
        )
        rows: List[FocusEventRow] = []
        for r in cur.fetchall():
            rows.append(
                FocusEventRow(
                    id=int(r["id"]),
                    timestamp=str(r["timestamp"]),
                    event_type=str(r["event_type"]),
                    confidence=float(r["confidence"] or 0.0),
                    score_delta=int(r["score_delta"] or 0),
                    notes=str(r["notes"] or ""),
                )
            )
        return rows

    def get_today_stats(self, day: Optional[date] = None) -> Optional[DailyStatsRow]:
        """读取指定日期（默认今天）的 daily_stats。"""
        d = (day or date.today()).isoformat()
        cur = self._conn.execute(
            """
            SELECT date, total_focus_seconds, distraction_count, avg_score, phone_detect_count
            FROM daily_stats
            WHERE date = ?
            """,
            (d,),
        )
        r = cur.fetchone()
        if r is None:
            return None
        return DailyStatsRow(
            date=str(r["date"]),
            total_focus_seconds=int(r["total_focus_seconds"] or 0),
            distraction_count=int(r["distraction_count"] or 0),
            avg_score=float(r["avg_score"] or 0.0),
            phone_detect_count=int(r["phone_detect_count"] or 0),
        )

    def upsert_daily_stats(
        self,
        day: Optional[date] = None,
        *,
        add_focus_seconds: int = 0,
        add_distraction: int = 0,
        add_phone_detect: int = 0,
        score_sample: Optional[float] = None,
    ) -> DailyStatsRow:
        """
        累加今日统计；若提供 score_sample 则与已有 avg_score 做简单均值更新。
        """
        d = (day or date.today()).isoformat()
        existing = self.get_today_stats(date.fromisoformat(d))
        focus_sec = (existing.total_focus_seconds if existing else 0) + max(
            0, add_focus_seconds
        )
        distract = (existing.distraction_count if existing else 0) + max(
            0, add_distraction
        )
        phone_cnt = (existing.phone_detect_count if existing else 0) + max(
            0, add_phone_detect
        )
        if score_sample is not None:
            if existing and existing.avg_score > 0:
                avg = (existing.avg_score + score_sample) * 0.5
            else:
                avg = float(score_sample)
        else:
            avg = existing.avg_score if existing else 0.0

        self._conn.execute(
            """
            INSERT INTO daily_stats (date, total_focus_seconds, distraction_count, avg_score, phone_detect_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                total_focus_seconds = excluded.total_focus_seconds,
                distraction_count = excluded.distraction_count,
                avg_score = excluded.avg_score,
                phone_detect_count = excluded.phone_detect_count
            """,
            (d, focus_sec, distract, avg, phone_cnt),
        )
        self._conn.commit()
        row = self.get_today_stats(date.fromisoformat(d))
        assert row is not None
        return row

    def _day_iso(self, day: Optional[date] = None) -> str:
        return (day or date.today()).isoformat()

    @staticmethod
    def _time_from_timestamp(ts: str) -> str:
        if " " in ts:
            return ts.split(" ", 1)[1][:8]
        if "T" in ts:
            return ts.split("T", 1)[1][:8]
        return ts[:8]

    def get_today_score_series(self, day: Optional[date] = None) -> List[ScorePoint]:
        """读取指定日期 score_tick 事件，按时间升序重建累计分数曲线。"""
        d = self._day_iso(day)
        cur = self._conn.execute(
            """
            SELECT timestamp, score_delta
            FROM focus_events
            WHERE event_type = 'score_tick' AND date(timestamp) = ?
            ORDER BY id ASC
            """,
            (d,),
        )
        total = 0
        points: List[ScorePoint] = []
        for r in cur.fetchall():
            total += int(r["score_delta"] or 0)
            points.append(
                ScorePoint(
                    time=self._time_from_timestamp(str(r["timestamp"])),
                    score=total,
                )
            )
        return points

    def get_today_event_counts(self, day: Optional[date] = None) -> Dict[str, int]:
        """统计指定日期各类型事件出现次数（Web 兜底用）。"""
        d = self._day_iso(day)
        cur = self._conn.execute(
            """
            SELECT event_type, COUNT(*) AS cnt
            FROM focus_events
            WHERE date(timestamp) = ?
              AND event_type IN ('distracted', 'phone_detected', 'focused')
            GROUP BY event_type
            """,
            (d,),
        )
        counts: Dict[str, int] = {}
        for r in cur.fetchall():
            counts[str(r["event_type"])] = int(r["cnt"])
        return counts

    def get_today_dashboard(self, day: Optional[date] = None) -> TodayDashboard:
        """
        组装 Web 仪表盘所需今日数据。
        daily_stats 缺失时，分心/手机次数从 focus_events 计数兜底。
        """
        d = day or date.today()
        d_iso = d.isoformat()
        stats = self.get_today_stats(d)
        event_counts = self.get_today_event_counts(d)
        series = self.get_today_score_series(d)

        if stats:
            focus_sec = stats.total_focus_seconds
            distract = stats.distraction_count
            phone_cnt = stats.phone_detect_count
            avg = stats.avg_score
        else:
            cur = self._conn.execute(
                """
                SELECT COUNT(*) AS cnt FROM focus_events
                WHERE event_type = 'score_tick' AND date(timestamp) = ? AND score_delta > 0
                """,
                (d_iso,),
            )
            focus_sec = int(cur.fetchone()["cnt"] or 0)
            distract = event_counts.get("distracted", 0)
            phone_cnt = event_counts.get("phone_detected", 0)
            avg = float(series[-1].score) if series else 0.0

        return TodayDashboard(
            date=d_iso,
            total_focus_seconds=focus_sec,
            distraction_count=distract,
            phone_detect_count=phone_cnt,
            avg_score=avg,
            score_series=series,
        )
