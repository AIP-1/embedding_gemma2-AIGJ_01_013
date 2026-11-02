"""10단계 EPFL 결과 그림: 정답 프레임 몇 개에서 위에서 본 평면도(정답 vs ankle1·stereo·fused) + 두 카메라 화면(관절, 정답 발 위치)
prox_epfl.py의 함수를 그대로 가져다 쓴다(실행 부분 앞까지만 불러옴).
사용: python viz_epfl.py --frames 500 1250 2000 --out step10_epfl_viz.jpg
"""
import argparse, os, sys, types
import cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, nargs="+", default=[500, 1250, 2000])
ap.add_argument("--out", default="step10_epfl_viz.jpg")
ap.add_argument("--h-guess", type=float, default=1.2)
ap.add_argument("--target-xy", default="1.79,1.80")
my = ap.parse_args()

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prox_epfl.py"), encoding="utf-8").read()
head = src.split("# ---------------- EPFL 보정 ----------------")[1].split("# ---------------- 10-6 알람")[0]
g = {"np": np, "cv2": cv2, "args": types.SimpleNamespace(near=0.8, far=1.6)}
exec(src.split("# ---------------- EPFL 보정 ----------------")[0].split("ap = argparse")[0], g)   # import 줄
g.update({"KP_CONF": 0.5, "NOSE": 0, "LS": 5, "LS": 5, "RS": 6, "LA": 15, "RA": 16, "HEAD_Z": 1.75,
          "ROOM": (-0.3, -0.3, 3.88, 3.90), "RAY_MAX": 0.3})
exec(head, g)
D = os.path.expanduser("~/eg2-lab/media/prox_epfl")
cams = [g["load_calib"](os.path.join(D, "calibration-6p.txt"))[c] for c in (0, 1)]
gt = g["load_gt"](os.path.join(D, "gt_lab_6p.txt"))
from ultralytics import YOLO
det = YOLO("yolo11n-pose.pt")
caps = [cv2.VideoCapture(os.path.join(D, f"6p-c{c}.avi")) for c in (0, 1)]
tgt = np.array([float(v) for v in my.target_xy.split(",")])

def to_img(cam, xy):
    """바닥 위치(m) → 영상 픽셀 (바닥 변환의 역)"""
    x = np.linalg.inv(cam["Hg"]) @ np.array([xy[0] * 100, xy[1] * 100, 1.0])
    return x[:2] / x[2]

rows = []
for f in my.frames:
    fs = min(gt, key=lambda k: abs(k - f))
    frames = []
    for c in caps:
        c.set(cv2.CAP_PROP_POS_FRAMES, fs); ok, fr = c.read(); frames.append(fr)
    rs = det.predict(frames, conf=0.5, verbose=False)
    kps = [np.concatenate([r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()[..., None]], -1)
           if r.keypoints is not None and r.keypoints.conf is not None else np.zeros((0, 17, 3)) for r in rs]
    obs = {m: g["observations"](cams, kps, m, my.h_guess) for m in ("ankle1", "stereo", "fused")}
    truth = list(gt[fs].values())
    # 평면도
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=100)
    ax.add_patch(plt.Rectangle((0, 0), 3.58, 3.60, fill=False, ec="gray"))
    ax.add_patch(plt.Circle(tgt, 0.8, fill=False, ec="red", ls="--"))
    ax.plot(*tgt, "r*", ms=12, label="target")
    if truth:
        t = np.array(truth); ax.plot(t[:, 0], t[:, 1], "kx", ms=11, mew=2.5, label="ground truth")
    styles = {"ankle1": ("o", "tab:blue"), "stereo": ("s", "tab:green")}
    for m, (mk, col) in styles.items():
        p = np.array([o[0] for o in obs[m]]) if obs[m] else np.zeros((0, 2))
        if len(p):
            ax.plot(p[:, 0], p[:, 1], mk, mfc="none", mec=col, ms=9, mew=1.8, label=m)
    sg = np.array([o[0] for o in obs["fused"] if o[2] == "single"]) if any(o[2] == "single" for o in obs["fused"]) else np.zeros((0, 2))
    if len(sg):
        ax.plot(sg[:, 0], sg[:, 1], "^", color="orange", ms=9, label=f"fused single (h={my.h_guess})")
    ax.set_xlim(-0.3, 3.9); ax.set_ylim(3.9, -0.3); ax.set_aspect("equal")
    ax.set_title(f"frame {fs}  (top view, m)", fontsize=9); ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout(); fig.canvas.draw()
    plan = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR); plt.close(fig)
    # 카메라 화면
    views = []
    for n in range(2):
        im = rs[n].plot(labels=False, boxes=False)
        for xy in truth:
            u, v = to_img(cams[n], xy)
            cv2.drawMarker(im, (int(u), int(v)), (0, 0, 0), cv2.MARKER_TILTED_CROSS, 14, 3)
            cv2.drawMarker(im, (int(u), int(v)), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 12, 1)
        u, v = to_img(cams[n], tgt)
        cv2.drawMarker(im, (int(u), int(v)), (0, 0, 255), cv2.MARKER_STAR, 16, 2)
        cv2.putText(im, f"cam{n} frame {fs}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        views.append(cv2.resize(im, (int(im.shape[1] * plan.shape[0] / im.shape[0] / 1.0), plan.shape[0])))
    rows.append(np.hstack([plan] + views))
w = max(r.shape[1] for r in rows)
rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
title = np.full((40, w, 3), 255, np.uint8)
cv2.putText(title, "EPFL lab 6p, cams 0+1: x = ground truth, o = ankle1(cam0), square = stereo, triangle = fused single, red * = virtual target (0.8 m circle)",
            (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
cv2.imwrite(my.out, np.vstack([title] + rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
print("saved", my.out)
