"""
浏览器内实时预览：HTTP multipart MJPEG（无需 X11 转发）。

在树莓派上从 SSH 运行时，OpenCV 窗口常因无 DISPLAY 无法显示；
同一局域网内用浏览器打开 http://<树莓派IP>:<端口>/ 即可查看标注后的画面。
"""
from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional


class _MJPEGHandler(BaseHTTPRequestHandler):
    server_version = "DeskEye-MJPEG/0.1"

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        httpd: ThreadingHTTPServer = self.server
        lock: threading.Lock = httpd.frame_lock  # type: ignore[attr-defined]
        stop_ev: threading.Event = httpd.stop_event  # type: ignore[attr-defined]
        fps: int = httpd.target_fps  # type: ignore[attr-defined]

        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            body = (
                "<!DOCTYPE html><html><head><meta charset='utf-8'><meta name='viewport' "
                "content='width=device-width'>"
                "<title>DeskEye 预览</title></head><body style='margin:0;background:#111;color:#ccc;"
                "font-family:sans-serif'>"
                "<p style='padding:8px'>DeskEye 人脸检测实时流。关闭本页不会停止脚本（需 SSH 里 Ctrl+C）。</p>"
                "<img src='/stream' alt='stream' style='max-width:100%;height:auto;display:block'/></body></html>"
            )
            self.wfile.write(body.encode("utf-8"))
            return
        if self.path != "/stream":
            self.send_error(404)
            return

        self.send_response(200)
        bnd = "mjpeg"
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={bnd}")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        boundary = f"--{bnd}\r\n".encode("ascii")
        try:
            while not stop_ev.is_set():
                jpg: Optional[bytes] = None
                with lock:
                    jpg = httpd.latest_jpeg  # type: ignore[attr-defined]
                if jpg:
                    try:
                        self.wfile.write(boundary)
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(jpg)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        break
                time.sleep(1.0 / max(1, fps))
        finally:
            try:
                self.wfile.close()
            except OSError:
                pass


class MjpegPreviewServer:
    """在独立线程中提供 MJPEG；主线程调用 set_frame_bgr 更新画面。"""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        target_fps: int = 20,
    ) -> None:
        self._host = host
        self._port = port
        self._target_fps = target_fps
        self._frame_lock = threading.Lock()
        self._stop = threading.Event()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def port(self) -> int:
        return self._port

    def start(self) -> None:
        if self._httpd is not None:
            return
        self._stop = threading.Event()
        httpd = ThreadingHTTPServer((self._host, self._port), _MJPEGHandler)
        httpd.allow_reuse_address = True
        httpd.frame_lock = self._frame_lock  # type: ignore[attr-defined]
        httpd.latest_jpeg = None  # type: ignore[attr-defined]
        httpd.stop_event = self._stop  # type: ignore[attr-defined]
        httpd.target_fps = self._target_fps  # type: ignore[attr-defined]
        self._httpd = httpd
        self._thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        self._thread.start()

    def set_frame_bgr(self, frame_bgr, jpeg_quality: int = 82) -> None:
        import cv2

        ok, buf = cv2.imencode(
            ".jpg",
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
        if not ok:
            return
        data = buf.tobytes()
        if self._httpd is None:
            return
        with self._frame_lock:
            self._httpd.latest_jpeg = data  # type: ignore[attr-defined]

    def stop(self) -> None:
        self._stop.set()
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
            except Exception:
                pass
            try:
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None
        self._thread = None
        self._stop = threading.Event()
