"""9-4: 샘플 영상을 처음부터 끝까지 돌려 raw 판정 / smooth 판정(9-2) / GRU 모델(9-6)의 동작별 횟수를 센다.
판정 규칙은 pose_app.py와 같다(0.6초 다수결 + 1.5초 손 흔들기). 시각은 영상 시간(프레임 번호 / fps)을 쓴다.
사용: python pose_eval.py media/pose/*.webm --model models/action.pt --out step9_result.md
"""
import argparse, collections, os, time
import cv2, numpy as np
from ultralytics import YOLO
from pose_model import KP_CONF, LW, RW, body, facing_score, frame_features, load_model, predict

ap = argparse.ArgumentParser()
ap.add_argument("videos", nargs="+")
ap.add_argument("--model", default="models/action.pt")
ap.add_argument("--conf", type=float, default=0.5)
ap.add_argument("--out", default="step9_result.md")
args = ap.parse_args()

SMOOTH_S, WAVE_S, ML_TH, MAX_H = 0.6, 1.5, 0.7, 720
ACTS = ["한팔들기", "만세", "손흔들기"]

# ---- pose_app.py와 같은 규칙 ----
def pose_label(k, b):
    (cx, cy), sw = b
    raised = lambda w: k[w, 2] > KP_CONF and k[w, 1] < cy - 0.3 * sw
    lu, ru = raised(LW), raised(RW)
    if lu and ru:
        return "만세"
    if lu or ru:
        return "한팔들기"
    return "기본"

def is_wave(hist):
    xs = [x for _, x in hist]
    if len(xs) < 8:
        return False
    changes, last_dir, anchor = 0, 0, xs[0]
    for x in xs[1:]:
        d = x - anchor
        if abs(d) > 0.15:
            cur = 1 if d > 0 else -1
            if last_dir and cur != last_dir:
                changes += 1
            last_dir, anchor = cur, x
    return changes >= 2

def majority(dq, now, window):
    vals = [v for t, v in dq if now - t <= window]
    if not vals:
        return None, 0.0
    v, n = collections.Counter(vals).most_common(1)[0]
    return v, n / len(vals)

det = YOLO("yolo11n-pose.pt")
gru = load_model(args.model) if args.model and os.path.exists(args.model) else None

def run(path):
    det.predictor = None                       # 영상마다 추적 ID 초기화
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    tracks = {}
    cnt = {"raw": collections.Counter(), "smooth": collections.Counter(), "gru": collections.Counter()}
    i, t0 = 0, time.time()
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fr.shape[0] > MAX_H:
            fr = cv2.resize(fr, (int(fr.shape[1] * MAX_H / fr.shape[0]), MAX_H))
        now = i / fps
        i += 1
        r = det.track(fr, persist=True, conf=args.conf, half=True, verbose=False)[0]
        if r.boxes.id is None or r.keypoints is None:
            continue
        kps = np.concatenate([r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()[..., None]], axis=-1)
        for tid, k in zip(r.boxes.id.int().tolist(), kps):
            st = tracks.setdefault(tid, {"raw": collections.deque(maxlen=90), "wave": collections.deque(maxlen=90),
                                         "feat": collections.deque(maxlen=gru["window"] if gru else 1),
                                         "lab": {"raw": "기본", "smooth": "기본", "gru": "기본"},
                                         "prev": None, "seen": now, "n": 0})
            dt, st["seen"] = now - st["seen"], now
            b = body(k)
            if b is None:
                continue
            raw = pose_label(k, b)
            fa = facing_score(k)
            spd = 0.0 if st["prev"] is None else float(np.linalg.norm(b[0] - st["prev"]) / b[1] / max(dt, 1e-3))
            st["prev"] = b[0]
            st["raw"].append((now, raw))
            if raw in ("한팔들기", "만세"):
                w = LW if k[LW, 1] < k[RW, 1] else RW
                st["wave"].append((now, (k[w, 0] - b[0][0]) / b[1]))
            st["feat"].append(frame_features(k, b, fa, spd))
            st["n"] += 1

            v, frac = majority(st["raw"], now, SMOOTH_S)
            sm = v if frac >= 0.6 else st["lab"]["smooth"]
            if sm == "한팔들기" and is_wave([(t, x) for t, x in st["wave"] if now - t <= WAVE_S]):
                sm = "손흔들기"
            new = {"raw": raw, "smooth": sm}
            if gru and st["n"] % 5 == 0 and len(st["feat"]) >= gru["window"]:
                lab, p = predict(gru, list(st["feat"])[-gru["window"]:])
                if p >= ML_TH:
                    new["gru"] = "기본" if lab == "idle" else lab
            for m, lab in new.items():
                if lab != st["lab"][m]:
                    if lab != "기본":
                        cnt[m][lab] += 1             # pose_app.py처럼 '기본'이 아닌 상태로 바뀔 때 1회
                    st["lab"][m] = lab
    sec = i / fps
    return cnt, sec, i / max(time.time() - t0, 1e-3), len(tracks)

rows = []
for v in args.videos:
    cnt, sec, proc_fps, people = run(v)
    rows.append((os.path.basename(v), sec, proc_fps, people, cnt))
    print(os.path.basename(v), f"{sec:.1f}s", f"{proc_fps:.1f} fps", {m: dict(c) for m, c in cnt.items()}, flush=True)

labels = sorted({l for *_, c in rows for m in c.values() for l in m} | set(ACTS), key=lambda x: (x not in ACTS, x))
cols = ["raw (9-1)", "smooth (9-2)"] + (["GRU action.pt (9-6)"] if gru else [])
keys = ["raw", "smooth"] + (["gru"] if gru else [])
out = [f"# 9단계 결과: 샘플 영상 동작 판정 비교 ({time.strftime('%Y-%m-%d')})", "",
       "입력: 웹캠 대신 인터넷 샘플 영상(출처는 media/pose/SOURCES.txt). yolo11n-pose.pt, conf "
       f"{args.conf}. 횟수 = '기본'이 아닌 동작으로 바뀐 횟수(pose_app.py의 동작 기록과 같은 기준).", ""]
for name, sec, proc_fps, people, cnt in rows:
    out += [f"## {name} ({sec:.1f}초, 처리 {proc_fps:.1f} fps, 추적 ID {people}개)", "",
            "| 동작 | " + " | ".join(cols) + " |", "| --- |" + " --- |" * len(cols)]
    out += [f"| {l} | " + " | ".join(str(cnt[k][l]) for k in keys) + " |" for l in labels]
    out += [f"| 합계 | " + " | ".join(str(sum(cnt[k].values())) for k in keys) + " |", ""]
open(args.out, "w", encoding="utf-8").write("\n".join(out) + "\n")
print("saved", args.out)
