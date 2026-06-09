#!/usr/bin/env python3
"""
下载阶段 3 物体检测用 YOLOv5n ONNX（Ultralytics 官方 release，当前为 FP32）。

用法（项目根）：python3 scripts/download_object_model.py

INT8：DeskEye 规范要求树莓派使用 INT8 量化模型；请自行导出/获取 yolov5n INT8 ONNX
并覆盖 models/yolov5n.onnx，或修改 config.OBJECT_ONNX_PATH 指向该文件。
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402

# Ultralytics YOLOv5 v7.0 附带 ONNX（当前 CDN 上多为 **float16** 输入；ObjectDetector 会按图元数据自动 cast）
YOLOV5N_ONNX_URL = (
    "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx"
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"下载: {url}\n  -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "DeskEye/object-models"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = resp.read()
    if len(data) < 2 * 1024 * 1024:
        raise ValueError(
            f"文件过小（{len(data)} 字节），可能为 404/限速页而非 yolov5n.onnx；请检查网络或使用浏览器下载后放到 {dest}"
        )
    if data[:9].lower().startswith(b"<!doctype") or data[:5].lower() == b"<html":
        raise ValueError("下载内容疑似 HTML 页面而非 ONNX，请检查 URL 或代理")
    dest.write_bytes(data)
    print(f"  完成，{len(data)} 字节")


def main() -> int:
    dest = config.OBJECT_ONNX_PATH
    try:
        _download(YOLOV5N_ONNX_URL, dest)
    except Exception as e:
        print(f"下载失败: {e}")
        return 1
    print("模型已就绪。运行: python3 tests/test_object.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
