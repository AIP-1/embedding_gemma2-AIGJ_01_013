"""카메라 실시간 보기 (보정 없이 화면만): 두 카메라를 나란히 브라우저로 본다
사용법: python cam_view.py --cams 0 2   →  http://<Jetson IP>:8020
보정 스크립트와 동시에 실행할 수 없다 (같은 카메라를 잡기 때문). 다 보면 Ctrl+C
"""
import argparse, time, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cams", type=int, nargs="+", default=[0, 2])
ap.add_argument("--res", default="640x480")
ap.add_argument("--port", type=int, default=8020)
args = ap.parse_args()
W, H = map(int, args.res.split("x"))

caps = []
for i in args.cams:
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    assert cap.isOpened(), f"카메라 {i}을 열 수 없음"
    caps.append(cap)

frames = [np.zeros((H, W, 3), np.uint8) for _ in caps]
latest = {"jpg": None}
lock = threading.Lock()

def grab(n):
    while True:
        ok, f = caps[n].read()
        if ok:
            cv2.putText(f, f"cam {args.cams[n]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            frames[n] = f

for n in range(len(caps)):
    threading.Thread(target=grab, args=(n,), daemon=True).start()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def do_GET(self):
        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            try:
                while True:
                    j = cv2.imencode(".jpg", np.hstack(frames))[1].tobytes()
                    self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n")
                    time.sleep(0.07)
            except (BrokenPipeError, ConnectionResetError):
                return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"<html><body style='background:#111;color:#eee;font-family:sans-serif'>"
                         b"<h3>camera view</h3><img src='/stream' style='max-width:100%'></body></html>")

print(f"보기: http://<Jetson IP>:{args.port}  (Ctrl+C로 종료)")
ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
