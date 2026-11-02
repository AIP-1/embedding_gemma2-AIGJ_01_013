"""10-1 렌즈 보정 (실시간 미리보기판): calib_intrinsics.py와 같은 보정 + 브라우저로 카메라 화면을 실시간 확인
브라우저: http://<Jetson IP>:8020  (초록 코너 = 인식됨, 빨간 글씨 = 인식 안 됨, 상단 = 모은 장 수)
사용법: python calib_intrinsics_live.py --cam 0
v3: 인식률 개선 - 자동 초점 끄고 고정(--focus), findChessboardCornersSB 우선 + 기존 방식 보조,
    흔들림 판정을 평균 이동량으로 완화(--still 3px), 모은 위치를 화면에 점으로 표시
v4: k3 고정(CALIB_FIX_K3), 사진별 오차 계산 버그 수정, 렌즈 값 이상 경고, 기울인 사진 수 표시,
    코너 좌표를 calib/camN_points.npz에 저장
"""
import argparse, os, time, threading, subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cam", type=int, required=True)
ap.add_argument("--cols", type=int, default=9, help="체커보드 내부 코너 가로 개수")
ap.add_argument("--rows", type=int, default=6, help="체커보드 내부 코너 세로 개수")
ap.add_argument("--square", type=float, default=0.025, help="칸 한 변 길이(m)")
ap.add_argument("--n", type=int, default=20, help="모을 이미지 수")
ap.add_argument("--res", default="640x480")
ap.add_argument("--port", type=int, default=8020)
ap.add_argument("--still", type=float, default=3.0, help="직전 프레임 대비 코너 평균 이동(px)이 이 값 이하일 때만 촬영 (흔들림 방지)")
ap.add_argument("--focus", type=int, default=0, help="고정 초점값 0~250 (0=먼 곳, 강의실용). -1이면 자동 초점 유지")
args = ap.parse_args()
W, H = map(int, args.res.split("x"))
os.makedirs("calib", exist_ok=True)

objp = np.zeros((args.rows * args.cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square

cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
assert cap.isOpened(), f"카메라 {args.cam}을 열 수 없음"

# 자동 초점이 켜져 있으면 판이 움직일 때마다 렌즈 값이 바뀌어 RMS가 커진다 → 고정
if args.focus >= 0:
    dev = f"/dev/video{args.cam}"
    for c in ("focus_automatic_continuous=0", f"focus_absolute={args.focus}", "exposure_dynamic_framerate=0"):
        subprocess.run(["v4l2-ctl", "-d", dev, "-c", c], check=False)
    print(f"초점 고정: {dev} focus_absolute={args.focus} (프로그램 실행 전에도 같은 값으로 고정해야 함)")

latest = {"jpg": None}
lock = threading.Lock()

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
                    with lock:
                        j = latest["jpg"]
                    if j is not None:
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n")
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(f"<html><body style='background:#111;color:#eee;font-family:sans-serif'>"
                         f"<h3>cam{args.cam} 렌즈 보정</h3><img src='/stream' style='max-width:100%'></body></html>".encode())

threading.Thread(target=lambda: ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever(),
                 daemon=True).start()
print(f"미리보기: http://<Jetson IP>:{args.port}")

def show(frame, found, corners, msg):
    v = frame.copy()
    if found:
        cv2.drawChessboardCorners(v, (args.cols, args.rows), corners, True)
    for p in centers:                               # 이미 모은 위치 (화면 고루 채우기용)
        cv2.circle(v, (int(p[0]), int(p[1])), 6, (255, 200, 0), -1)
    col = (0, 255, 0) if found else (0, 0, 255)
    tl = "" if not found else (" TILT" if tilt(corners) else " flat")
    cv2.putText(v, f"{len(img_pts)}/{args.n} tilted {n_tilt}  {msg}{tl}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    with lock:
        latest["jpg"] = cv2.imencode(".jpg", v)[1].tobytes()

obj_pts, img_pts, last_t, last_c, prev, centers = [], [], 0.0, None, None, []
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
SIZE = (args.cols, args.rows)

def detect(gray):
    # SB 방식: 조명·기울기·약간의 흐림에 강하고 코너 정밀도도 높다
    found, corners = cv2.findChessboardCornersSB(gray, SIZE, cv2.CALIB_CB_NORMALIZE_IMAGE)
    if found:
        return True, corners
    # 보조: 기존 방식 (적응형 임계값 + 밝기 정규화) 후 서브픽셀 보정
    found, corners = cv2.findChessboardCorners(gray, SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
    if found:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
    return found, corners

def tilt(corners):
    # 마주 보는 두 변 길이 비가 1.15를 넘으면 기울인 사진 (원근 때문에 한쪽 변이 짧아짐)
    g = corners.reshape(args.rows, args.cols, 2)
    top, bot = np.linalg.norm(g[0, -1] - g[0, 0]), np.linalg.norm(g[-1, -1] - g[-1, 0])
    lef, rig = np.linalg.norm(g[-1, 0] - g[0, 0]), np.linalg.norm(g[-1, -1] - g[0, -1])
    r = max(top / bot, bot / top, lef / rig, rig / lef)
    return r > 1.15

n_tilt = 0

def motion(a, b):
    # 프레임마다 코너 순서가 뒤집혀 나올 수 있어 양방향 중 작은 값
    return min(np.abs(a - b).mean(), np.abs(a[::-1] - b).mean())
print(f"체커보드({args.cols}x{args.rows})를 화면 곳곳·여러 각도로 천천히 움직이세요. 목표 {args.n}장")
while len(img_pts) < args.n:
    ok, frame = cap.read()
    if not ok:
        continue
    assert frame.shape[1] == W, f"해상도가 {W}x{H}가 아님: {frame.shape[1]}x{frame.shape[0]}"
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = detect(gray)
    if not found:
        prev = None
        show(frame, False, None, "not found"); continue
    moving = prev is None or motion(corners, prev) > args.still
    prev = corners
    if moving:
        show(frame, True, corners, "hold still"); continue
    if time.time() - last_t < 1.0:
        show(frame, True, corners, "wait"); continue
    c = corners.reshape(-1, 2).mean(0)
    if last_c is not None and np.linalg.norm(c - last_c) < 40:     # 비슷한 위치는 건너뜀
        show(frame, True, corners, "move to a new spot"); continue
    obj_pts.append(objp); img_pts.append(corners)
    last_t, last_c = time.time(), c
    centers.append(c)
    n_tilt += int(tilt(corners))
    show(frame, True, corners, "captured!")
    cv2.drawChessboardCorners(frame, (args.cols, args.rows), corners, True)
    cv2.imwrite(f"calib/cam{args.cam}_last.jpg", frame)
    print(f"  {len(img_pts)}/{args.n}")

def calib(op, ip):
    # 웹캠은 k3까지 풀면 과적합되기 쉬워 고정한다
    rms, K, dist, rv, tv = cv2.calibrateCamera(op, ip, (W, H), None, None, flags=cv2.CALIB_FIX_K3)
    errs = [float(np.sqrt(np.mean(np.sum((cv2.projectPoints(o, r, t, K, dist)[0].reshape(-1, 2)
                                          - np.asarray(i).reshape(-1, 2)) ** 2, axis=1))))
            for o, i, r, t in zip(op, ip, rv, tv)]
    return rms, K, dist, np.array(errs)

rms, K, dist, errs = calib(obj_pts, img_pts)
print(f"\n1차 RMS {rms:.3f}px, 사진별 오차: {errs.round(2).tolist()}")
keep = errs <= max(1.0, 2 * np.median(errs))      # 오차가 큰 사진(흔들림·휨)은 빼고 다시 계산
if keep.sum() >= 10 and not keep.all():
    obj_pts = [o for o, k in zip(obj_pts, keep) if k]
    img_pts = [i for i, k in zip(img_pts, keep) if k]
    rms, K, dist, errs = calib(obj_pts, img_pts)
    print(f"오차 큰 사진 {int((~keep).sum())}장 제외 → {len(img_pts)}장으로 다시 계산")
np.savez(f"calib/cam{args.cam}_intr.npz", K=K, dist=dist, res=np.array([W, H]))
np.savez(f"calib/cam{args.cam}_points.npz", img=np.array([np.asarray(i).reshape(-1, 2) for i in img_pts]),
         obj=np.array(obj_pts), res=np.array([W, H]))      # 나중에 다시 계산할 때 사용
print(f"\n재투영 RMS {rms:.3f}px  (0.5 이하 양호, 1.0 이상이면 다시 촬영)")
print(f"K=\n{K.round(1)}\ndist={dist.ravel().round(3)}\n저장: calib/cam{args.cam}_intr.npz")
print(f"기울인 사진 {n_tilt}/{len(img_pts)}장")
d = dist.ravel()
if not 500 <= K[0, 0] <= 800 or abs(d[0]) > 1 or abs(d[1]) > 1:
    print("경고: 렌즈 값 비정상 — 판을 30~45도 기울인 사진과 화면 모서리 사진을 더 찍으세요")
