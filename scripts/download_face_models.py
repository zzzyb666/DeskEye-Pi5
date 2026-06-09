#!/usr/bin/env python3
"""
下载阶段 2 人脸检测所需 Caffe 模型（OpenCV 官方样例用 Res10 SSD）。

用法（在项目根）：python3 scripts/download_face_models.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402

# OpenCV 3rdparty + samples（与 opencv/samples/dnn/face_detector 一致）
CAFFE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
    "dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
)
PROTO_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt"
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"下载: {url}\n  -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "DeskEye/face-models"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    dest.write_bytes(data)
    print(f"  完成，{len(data)} 字节")


def main() -> int:
    proto = config.FACE_PROTO_PATH
    caffemodel = config.FACE_CAFFEMODEL_PATH
    try:
        _download(PROTO_URL, proto)
        _download(CAFFE_URL, caffemodel)
    except Exception as e:
        print(f"下载失败: {e}")
        return 1
    print("模型已就绪。运行: python3 tests/test_face.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
