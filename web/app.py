"""
DeskEye Web 仪表盘（阶段 5）：Flask 本地服务，展示今日专注统计。

用法（项目根）：
  python3 web/app.py

浏览器访问：http://<pi-ip>:5000/
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, render_template  # noqa: E402

import config  # noqa: E402
from src.database import DeskEyeDatabase  # noqa: E402

app = Flask(__name__, template_folder=str(config.WEB_DIR / "templates"))


def _dashboard_to_json(data) -> dict:
    return {
        "date": data.date,
        "total_focus_seconds": data.total_focus_seconds,
        "distraction_count": data.distraction_count,
        "phone_detect_count": data.phone_detect_count,
        "avg_score": round(data.avg_score, 1),
        "score_series": [
            {"time": p.time, "score": p.score} for p in data.score_series
        ],
        "refresh_sec": int(config.WEB_REFRESH_SEC),
    }


@app.route("/")
def index():
    return render_template(
        "index.html",
        refresh_sec=int(config.WEB_REFRESH_SEC),
    )


@app.route("/api/today")
def api_today():
    db = DeskEyeDatabase()
    try:
        payload = _dashboard_to_json(db.get_today_dashboard())
        return jsonify(payload)
    finally:
        db.close()


def main() -> None:
    print(
        f"DeskEye Web 仪表盘启动：http://0.0.0.0:{config.WEB_PORT}/\n"
        f"本机可试 http://127.0.0.1:{config.WEB_PORT}/\n"
        f"数据文件：{config.DATABASE_PATH}\n"
        f"自动刷新间隔：{config.WEB_REFRESH_SEC}s"
    )
    app.run(
        host=config.WEB_HOST,
        port=int(config.WEB_PORT),
        debug=False,
        threaded=True,
    )


if __name__ == "__main__":
    main()
