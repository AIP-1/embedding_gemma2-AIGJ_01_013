# 샘플 영상들을 반복 재생해서 MJPEG으로 내보낸다 (웹캠 대신 pose_app.py 입력용). Jetson localhost:8096에만 연다.
# 사용: python video_stream.py media/pose/*.webm   → pose_app.py --cam http://127.0.0.1:8096/stream
import sys, time, threading
import cv2
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST, PORT, MAX_H = "127.0.0.1", 8096, 720
FILES = sys.argv[1:]
lock, latest = threading.Lock(), None

def play():
    global latest
    while True:
        for f in FILES:
            cap = cv2.VideoCapture(f)
            dt = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30)
            while True:
                t = time.time()
                ok, fr = cap.read()
                if not ok:
                    break
                if fr.shape[0] > MAX_H:
                    fr = cv2.resize(fr, (int(fr.shape[1] * MAX_H / fr.shape[0]), MAX_H))
                ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with lock:
                        latest = buf.tobytes()
                time.sleep(max(0.0, dt - (time.time() - t)))
            cap.release()

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/stream":
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with lock:
                    f = latest
                if f:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n")
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError):
            pass
    def log_message(self, *a): pass

if __name__ == "__main__":
    assert FILES, "영상 파일을 인자로 주세요"
    threading.Thread(target=play, daemon=True).start()
    print(f"http://{HOST}:{PORT}/stream  ({len(FILES)}개 반복 재생)", flush=True)
    ThreadingHTTPServer((HOST, PORT), H).serve_forever()
