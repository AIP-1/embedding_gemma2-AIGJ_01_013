# USB 웹캠(/dev/video0)을 브라우저로 실시간 보기 (MJPEG over HTTP). Jetson localhost에만 연다.
# Mac에서: ssh -L 8090:127.0.0.1:8090 jetson  → http://localhost:8090
import subprocess, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST, PORT = "127.0.0.1", 8090
latest, cond = None, threading.Condition()

def grab():
    global latest
    p = subprocess.Popen(["gst-launch-1.0", "-q", "v4l2src", "device=/dev/video0", "!",
                          "image/jpeg,width=640,height=480,framerate=15/1", "!", "fdsink", "fd=1"],
                         stdout=subprocess.PIPE)
    buf = b""
    while chunk := p.stdout.read(65536):
        buf += chunk
        while True:
            s = buf.find(b"\xff\xd8"); e = buf.find(b"\xff\xd9", s + 2)
            if s < 0 or e < 0: break
            with cond:
                latest = buf[s:e + 2]; cond.notify_all()
            buf = buf[e + 2:]

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/stream":
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers()
            self.wfile.write(b"<html><body style=\"margin:0;background:#000\"><img src=\"/stream\" style=\"width:100%\"></body></html>"); return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame"); self.end_headers()
        try:
            while True:
                with cond:
                    cond.wait(); f = latest
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(f)).encode() + b"\r\n\r\n" + f + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
    def log_message(self, *a): pass

threading.Thread(target=grab, daemon=True).start()
print(f"http://{HOST}:{PORT}", flush=True)
ThreadingHTTPServer((HOST, PORT), H).serve_forever()
