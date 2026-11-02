"""9단계 개선 비교: 같은 검출 결과로 v1 규칙(pose_app.py 그대로)과 v2 규칙, GRU 모델 여러 개를 한 번에 센다.
v2 규칙
  - 동작 횟수는 '동작 구간' 단위: 기본 → (한팔들기/만세/손흔들기) → 기본 한 번을 1회로, 구간에서 가장 높은 동작으로 센다
    (v1은 한팔들기↔손흔들기가 오갈 때마다 따로 셌다)
  - 손흔들기: 손목 좌우 방향 전환 3번 이상 + 창(1.5초) 안의 80% 이상 프레임에서 손이 올라가 있어야 함(팔 내리는 동작 제외)
  - --low-wave: 어깨 높이(어깨선 + 어깨 너비 0.2배 위쪽)에서 흔드는 손도 손흔들기로 봄
  - 악수: 두 손목이 각자 상대 쪽으로 뻗어 있고(자기 어깨 중심에서 상대 방향으로 어깨 너비 0.3배 이상), 두 사람 중심이
    어깨 너비 0.8배 이상 떨어져 있어야 후보. 0.5초 이상 이어져야 1회, 같은 자리(어깨 너비 2배 안)는 2초 안에 다시 안 셈
사용: python pose_eval_v2.py media/pose/a.webm --models models/action.pt models/action_v2.pt [--low-wave]
"""
import argparse, collections, json, os, time
import cv2, numpy as np
from ultralytics import YOLO
from pose_model import KP_CONF, LW, RW, body, facing_score, frame_features, load_model, predict

ap = argparse.ArgumentParser()
ap.add_argument("videos", nargs="+")
ap.add_argument("--models", nargs="*", default=["models/action.pt"])
ap.add_argument("--conf", type=float, default=0.5)
ap.add_argument("--low-wave", action="store_true")
ap.add_argument("--json", default=None, help="결과를 JSON 줄로 덧붙일 파일")
args = ap.parse_args()

SMOOTH_S, WAVE_S, MAX_H = 0.6, 1.5, 720
HS_MIN_S, HS_COOL_S = 0.5, 2.0
RANK = {"기본": 0, "한팔들기": 1, "만세": 2, "손흔들기": 3}

def pose_label(k, b):
    (cx, cy), sw = b
    raised = lambda w: k[w, 2] > KP_CONF and k[w, 1] < cy - 0.3 * sw
    lu, ru = raised(LW), raised(RW)
    return "만세" if lu and ru else "한팔들기" if lu or ru else "기본"

def direction_changes(xs, amp=0.15):
    changes, last_dir, anchor = 0, 0, xs[0]
    for x in xs[1:]:
        d = x - anchor
        if abs(d) > amp:
            cur = 1 if d > 0 else -1
            if last_dir and cur != last_dir:
                changes += 1
            last_dir, anchor = cur, x
    return changes

def is_wave_v1(hist):
    xs = [x for _, x in hist]
    return len(xs) >= 8 and direction_changes(xs) >= 2

def is_wave_v2(hist, n_frames):
    xs = [x for _, x in hist]
    return len(xs) >= 8 and len(xs) >= 0.8 * n_frames and direction_changes(xs) >= 3

def majority(dq, now, window):
    vals = [v for t, v in dq if now - t <= window]
    if not vals:
        return None, 0.0
    v, n = collections.Counter(vals).most_common(1)[0]
    return v, n / len(vals)

def wrist_pair(a, b, ba, bb):
    """두 사람 손목 중 가장 가까운 쌍: (거리, 중간점, a손목, b손목). 높이는 a의 가슴~허리"""
    best = (1e9, None, None, None)
    for wa in (LW, RW):
        for wb in (LW, RW):
            if a[wa, 2] > KP_CONF and b[wb, 2] > KP_CONF:
                mid = (a[wa, :2] + b[wb, :2]) / 2
                if ba[0][1] < mid[1] < ba[0][1] + 2.5 * ba[1]:
                    d = np.linalg.norm(a[wa, :2] - b[wb, :2])
                    if d < best[0]:
                        best = (d, mid, wa, wb)
    return best

def handshake_candidates(kps, ids):
    v1, v2 = {}, {}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = kps[i], kps[j]
            ba, bb = body(a), body(b)
            if ba is None or bb is None:
                continue
            key = tuple(sorted((ids[i], ids[j])))
            sw = (ba[1] + bb[1]) / 2
            d, mid, wa, wb = wrist_pair(a, b, ba, bb)
            v1[key] = (d < 0.5 * sw, mid)
            ok = False
            if mid is not None and d < 0.5 * sw and abs(ba[0][0] - bb[0][0]) >= 0.8 * sw:
                s = 1 if bb[0][0] > ba[0][0] else -1          # a에서 b 쪽 방향
                reach_a = (a[wa, 0] - ba[0][0]) * s
                reach_b = (b[wb, 0] - bb[0][0]) * -s
                ok = reach_a >= 0.3 * ba[1] and reach_b >= 0.3 * bb[1]
            v2[key] = (ok, mid, sw)
    return v1, v2

det = YOLO("yolo11n-pose.pt")
grus = [(os.path.basename(p), load_model(p)) for p in args.models if os.path.exists(p)]
GW = max([g["window"] for _, g in grus], default=1)
COLS = ["v1"] + ["v2"] + [f"{n}@0.7" for n, _ in grus] + [f"{n}@0.5" for n, _ in grus]

def run(path):
    det.predictor = None
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cnt = {c: collections.Counter() for c in COLS}
    tracks = {}
    pairs1, on1 = {}, set()
    pairs2, on2, hs_log = {}, {}, []                 # on2: key -> 시작 시각, hs_log: (시각, 중간점, sw)
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
        ids = r.boxes.id.int().tolist()

        # ---- 악수 ----
        c1, c2 = handshake_candidates(kps, ids) if len(ids) >= 2 else ({}, {})
        for key in list(pairs1):
            if key not in c1:
                pairs1.pop(key); on1.discard(key)
        for key, (hit, _) in c1.items():
            pairs1.setdefault(key, collections.deque(maxlen=60)).append((now, hit))
            v, frac = majority(pairs1[key], now, SMOOTH_S)
            on = bool(v) and frac >= 0.6
            if on and key not in on1:
                cnt["v1"]["악수"] += 1
            (on1.add if on else on1.discard)(key)
        for key in list(pairs2):
            if key not in c2:
                pairs2.pop(key); on2.pop(key, None)
        for key, (hit, mid, sw) in c2.items():
            pairs2.setdefault(key, collections.deque(maxlen=60)).append((now, hit))
            v, frac = majority(pairs2[key], now, SMOOTH_S)
            if bool(v) and frac >= 0.6:
                start = on2.setdefault(key, now)
                if start is not None and now - start >= HS_MIN_S:
                    recent = [h for h in hs_log if now - h[0] < HS_COOL_S and np.linalg.norm(h[1] - mid) < 2 * sw]
                    if not recent:
                        cnt["v2"]["악수"] += 1
                    hs_log.append((now, mid, sw))
                    on2[key] = None                      # 이 쌍은 끊길 때까지 다시 안 셈
            else:
                on2.pop(key, None)

        # ---- 한 사람 동작 ----
        for tid, k in zip(ids, kps):
            st = tracks.setdefault(tid, {"raw": collections.deque(maxlen=90), "w1": collections.deque(maxlen=90),
                                         "w2": collections.deque(maxlen=90), "fr": collections.deque(maxlen=90),
                                         "feat": collections.deque(maxlen=GW), "s1": "기본", "s2": "기본",
                                         "ep": None, "hold": -1.0, "g": {}, "prev": None, "seen": now, "n": 0})
            dt, st["seen"] = now - st["seen"], now
            b = body(k)
            if b is None:
                continue
            (cx, cy), sw = b
            raw = pose_label(k, b)
            fa = facing_score(k)
            spd = 0.0 if st["prev"] is None else float(np.linalg.norm(b[0] - st["prev"]) / sw / max(dt, 1e-3))
            st["prev"] = b[0]
            st["raw"].append((now, raw))
            st["fr"].append(now)
            w = LW if k[LW, 1] < k[RW, 1] else RW
            if raw in ("한팔들기", "만세"):
                st["w1"].append((now, (k[w, 0] - cx) / sw))
            low_up = k[w, 2] > KP_CONF and k[w, 1] < cy + 0.2 * sw
            if raw in ("한팔들기", "만세") or (args.low_wave and low_up):
                st["w2"].append((now, (k[w, 0] - cx) / sw))
            st["feat"].append(frame_features(k, b, fa, spd))
            st["n"] += 1
            v, frac = majority(st["raw"], now, SMOOTH_S)

            # v1 (pose_app.py 그대로)
            s1 = v if frac >= 0.6 else st["s1"]
            if s1 == "한팔들기" and is_wave_v1([(t, x) for t, x in st["w1"] if now - t <= WAVE_S]):
                s1 = "손흔들기"
            if s1 != st["s1"]:
                if s1 != "기본":
                    cnt["v1"][s1] += 1
                st["s1"] = s1

            # v2
            s2 = v if frac >= 0.6 else st["s2"]
            n_win = sum(1 for t in st["fr"] if now - t <= WAVE_S)
            if s2 in ("한팔들기", "기본") and is_wave_v2([(t, x) for t, x in st["w2"] if now - t <= WAVE_S], n_win):
                if s2 == "한팔들기" or args.low_wave:
                    s2, st["hold"] = "손흔들기", now
            if st["s2"] == "손흔들기" and s2 != "만세" and now - st["hold"] < 1.0:
                s2 = "손흔들기"                          # 1초 유지(깜빡임 방지)
            st["s2"] = s2
            if s2 != "기본":
                st["ep"] = s2 if st["ep"] is None or RANK[s2] > RANK[st["ep"]] else st["ep"]
            elif st["ep"] is not None:
                cnt["v2"][st["ep"]] += 1; st["ep"] = None

            # GRU
            if grus and st["n"] % 5 == 0:
                seq = list(st["feat"])
                for name, g in grus:
                    if len(seq) >= g["window"]:
                        lab, p = predict(g, seq[-g["window"]:])
                        for th in (0.7, 0.5):
                            key = f"{name}@{th}"
                            new = ("기본" if lab == "idle" else lab) if p >= th else st["g"].get(key, "기본")
                            if new != st["g"].get(key, "기본") and new != "기본":
                                cnt[key][new] += 1
                            st["g"][key] = new
    for st in tracks.values():                           # 영상이 끝날 때 열린 구간 마감
        if st["ep"] is not None:
            cnt["v2"][st["ep"]] += 1
    return cnt, i / fps, i / max(time.time() - t0, 1e-3)

for vpath in args.videos:
    cnt, sec, pf = run(vpath)
    res = {"video": os.path.basename(vpath), "sec": round(sec, 1), "fps": round(pf, 1), "low_wave": args.low_wave,
           "counts": {c: dict(cnt[c]) for c in COLS}}
    print(json.dumps(res, ensure_ascii=False), flush=True)
    if args.json:
        with open(args.json, "a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
