"""9-5 행동 패턴 비교: pose_app.py --behavior와 같은 규칙(정면 응시 시간·이동 속도·체류 시간)으로 영상 속 사람마다 행동을 판정한다.
실제 앱처럼 초당 약 15장이 되도록 프레임을 건너뛰고, GPU 메모리를 아끼려고 CPU로 돌릴 수 있다(CUDA_VISIBLE_DEVICES= 로 실행).
--model 을 주면 9-6 행동 모델(train_pose.py --task behavior)의 판정도 같이 센다(5프레임마다, 확률 0.7 이상).
출력: 사람(추적 ID)별 최종 행동(앱의 '요약: ... → 행동'과 같은 기준)과 행동이 바뀐 횟수
사용: CUDA_VISIBLE_DEVICES= python pose_eval_beh.py <영상> --json out.jsonl [--model models/behavior_v1.pt]
"""
import argparse, collections, json, os, time
import cv2, numpy as np, torch
from ultralytics import YOLO
from pose_model import body, facing_score, frame_features, load_model, predict

ap = argparse.ArgumentParser()
ap.add_argument("videos", nargs="+")
ap.add_argument("--conf", type=float, default=0.5)
ap.add_argument("--fps", type=float, default=15.0)
ap.add_argument("--model", default=None)
ap.add_argument("--json", default=None)
args = ap.parse_args()

# ---- pose_app.py와 같은 기준 ----
BEH_WIN = 15.0
FACE_TH = 0.6
LOOK_S, INTEREST_S, LINGER_S = 1.0, 3.0, 10.0
MOVE_TH = 0.5
ML_TH = 0.7
BEH_RANK = {"관찰 중": 0, "지나감": 1, "머무름": 2, "힐끗 봄": 3, "관심 있음": 4}
EN2KO = {"passing": "지나감", "glance": "힐끗 봄", "interested": "관심 있음", "linger": "머무름"}

def behavior(st, now):
    h = [x for x in st["beh"] if now - x[0] <= BEH_WIN]
    if len(h) < 5:
        return "관찰 중"
    run = 0.0
    for i in range(len(h) - 1, 0, -1):
        if h[i][1] < FACE_TH:
            break
        run = h[-1][0] - h[i - 1][0]
    look = sum(h[i][0] - h[i - 1][0] for i in range(1, len(h)) if h[i][1] >= FACE_TH)
    moving = np.median([s for t, _, s in h if now - t < 2.0]) > MOVE_TH
    if run >= INTEREST_S and not moving:
        return "관심 있음"
    if moving and LOOK_S <= look < INTEREST_S:
        return "힐끗 봄"
    if moving:
        return "지나감"
    if now - st["first"] >= LINGER_S:
        return "머무름"
    return "관찰 중"

dev = "cuda" if torch.cuda.is_available() else "cpu"
det = YOLO("yolo11n-pose.pt")
gru = load_model(args.model, device=dev) if args.model else None

def run(path):
    det.predictor = None
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(src_fps / args.fps))
    tracks, finals, trans, ml_final = {}, collections.Counter(), collections.Counter(), collections.Counter()
    i, t0 = 0, time.time()
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        if (i - 1) % step:
            continue
        if fr.shape[0] > 720:
            fr = cv2.resize(fr, (int(fr.shape[1] * 720 / fr.shape[0]), 720))
        now = (i - 1) / src_fps
        r = det.track(fr, persist=True, conf=args.conf, device=dev, verbose=False)[0]
        ids = r.boxes.id.int().tolist() if r.boxes.id is not None and r.keypoints is not None else []
        kps = np.concatenate([r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()[..., None]], axis=-1) if ids else []
        for tid, k in zip(ids, kps):
            st = tracks.setdefault(tid, {"beh": collections.deque(maxlen=600), "feat": collections.deque(maxlen=gru["window"] if gru else 1),
                                         "first": now, "seen": now, "prev": None, "bstate": "관찰 중", "best": "관찰 중",
                                         "ml": None, "ml_best": None, "n": 0})
            dt, st["seen"] = now - st["seen"], now
            b = body(k)
            if b is None:
                continue
            fa = facing_score(k)
            spd = 0.0 if st["prev"] is None else float(np.linalg.norm(b[0] - st["prev"]) / b[1] / max(dt, 1e-3))
            st["prev"] = b[0]
            st["beh"].append((now, fa, spd))
            st["feat"].append(frame_features(k, b, fa, spd))
            st["n"] += 1
            bs = behavior(st, now)
            if gru and st["n"] % 5 == 0 and len(st["feat"]) >= gru["window"]:
                lab, p = predict(gru, list(st["feat"])[-gru["window"]:], device=dev)
                if p >= ML_TH:
                    st["ml"] = EN2KO.get(lab, lab)
                    if st["ml_best"] is None or BEH_RANK.get(st["ml"], 0) > BEH_RANK.get(st["ml_best"], 0):
                        st["ml_best"] = st["ml"]
            if bs != st["bstate"]:
                if bs != "관찰 중":
                    trans[bs] += 1
                st["bstate"] = bs
            if BEH_RANK.get(bs, 1) > BEH_RANK.get(st["best"], 0):
                st["best"] = bs
        for tid in [t for t, s in tracks.items() if now - s["seen"] > 3]:
            s = tracks.pop(tid)
            if s["best"] != "관찰 중":
                finals[s["best"]] += 1
            if s["ml_best"]:
                ml_final[s["ml_best"]] += 1
    for s in tracks.values():                       # 영상 끝: 남은 사람도 요약
        if s["best"] != "관찰 중":
            finals[s["best"]] += 1
        if s["ml_best"]:
            ml_final[s["ml_best"]] += 1
    return finals, trans, ml_final, i / src_fps, (i / step) / max(time.time() - t0, 1e-3)

for vpath in args.videos:
    finals, trans, ml_final, sec, pf = run(vpath)
    res = {"video": os.path.basename(vpath), "sec": round(sec, 1), "proc_fps": round(pf, 1), "people_final": dict(finals),
           "transitions": dict(trans), "model_final": dict(ml_final)}
    print(json.dumps(res, ensure_ascii=False), flush=True)
    if args.json:
        with open(args.json, "a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
