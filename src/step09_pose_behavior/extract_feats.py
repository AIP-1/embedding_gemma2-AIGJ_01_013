"""9-6 데이터 늘리기: 샘플 영상에서 pose_app.py --record와 같은 형식의 관절 특징 CSV를 만든다.
웹캠(15fps)과 맞추려고 25fps 이상 영상은 프레임을 건너뛴다. 시각은 영상 시간 + 영상별 오프셋이라
train_pose.py가 영상끼리 이어 붙이지 않는다(0.5초 넘게 끊기면 다른 시퀀스).
사용: python extract_feats.py --label wave --out data/action_v2/wave.csv media/pose/a.webm media/pose/b.mp4
"""
import argparse, csv, os, time
import cv2, numpy as np
from ultralytics import YOLO
from pose_model import body, facing_score, frame_features

ap = argparse.ArgumentParser()
ap.add_argument("videos", nargs="+")
ap.add_argument("--label", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--conf", type=float, default=0.5)
ap.add_argument("--fps", type=float, default=15.0)
args = ap.parse_args()
MAX_H = 720

det = YOLO("yolo11n-pose.pt")
f = open(args.out, "a", newline="")
w = csv.writer(f)
base = time.time()
for vi, path in enumerate(args.videos):
    det.predictor = None
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(src_fps / args.fps))
    off = base + vi * 10000.0
    prev, last_t, i, rows = {}, {}, 0, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        if (i - 1) % step:
            continue
        if fr.shape[0] > MAX_H:
            fr = cv2.resize(fr, (int(fr.shape[1] * MAX_H / fr.shape[0]), MAX_H))
        now = (i - 1) / src_fps
        r = det.track(fr, persist=True, conf=args.conf, verbose=False)[0]
        if r.boxes.id is None or r.keypoints is None:
            continue
        kps = np.concatenate([r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()[..., None]], axis=-1)
        for tid, k in zip(r.boxes.id.int().tolist(), kps):
            b = body(k)
            if b is None:
                continue
            dt = now - last_t.get(tid, now)
            spd = 0.0 if tid not in prev else float(np.linalg.norm(b[0] - prev[tid]) / b[1] / max(dt, 1e-3))
            prev[tid], last_t[tid] = b[0], now
            feat = frame_features(k, b, facing_score(k), spd)
            w.writerow([round(off + now, 3), f"v{vi}_{tid}", args.label, round(facing_score(k), 3), round(spd, 3)]
                       + feat[:51].round(3).tolist())
            rows += 1
    print(f"{os.path.basename(path)} → {args.label}: {rows}행", flush=True)
f.close()
