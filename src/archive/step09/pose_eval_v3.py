"""9단계 손흔들기 개선 비교: 같은 검출 결과로 손흔들기 판정 방식 여러 개를 한 번에 센다(손흔들기 구간 수).
방식(열)
  v2      : pose_eval_v2.py 그대로 (손목이 어깨보다 어깨 너비 0.3배 위, 손목 x 방향 전환 3번, 손 올림 80%)
  v2low   : v2 + 어깨 높이(어깨선 + 어깨 너비 0.2배)까지 허용
  traj    : 팔마다 '흔드는 자세'(손목이 팔꿈치보다 위 + 어깨선보다 몸통 0.25배 아래까지) 70% 이상,
            팔꿈치 기준 손목 궤적(아래팔 길이로 나눔)의 주 방향(PCA) 투영이 0.2 이상씩 2번 이상 방향 전환
  traj3   : traj와 같고 방향 전환 3번
  angle   : 같은 자세 조건, 아래팔 각도(팔꿈치→손목)가 15도 이상씩 2번 이상 방향 전환
  angle3w : angle과 같고 방향 전환 3번, 창 2.0초
  trajB   : traj + 두 손목이 붙은 프레임(박수 자세, 아래팔 길이 0.6배 안) 제외, 진폭 0.3, 흔드는 자세 80%
  angleB  : angle + 같은 박수 자세 제외, 각도 진폭 20도, 흔드는 자세 80%
  trajC   : traj + 박수 자세 제외만 (진폭 0.2, 자세 70%는 traj와 같음)
  angleC  : angle + 박수 자세 제외만 (15도, 70%)
  xnorm   : v2와 같은 손목 x 방식이지만 거리 단위를 max(어깨 너비, 0.4×몸통)으로 하고 손 높이는 '손목이 팔꿈치보다 위'
모든 방식: 방향 전환은 1.5초(angle3w는 2.0초) 창, 전환이 창 안에서 15번 넘으면(약 5Hz 초과, 떨림) 제외,
손흔들기면 1초 유지, 1초 넘게 끊겨야 새 구간.
진단(diag): v2가 놓친 이유를 보려고 사람·프레임 단위 통계를 함께 기록.
사용: python pose_eval_v3.py <영상> --json out.jsonl
"""
import argparse, collections, json, os, time
import cv2, numpy as np
from ultralytics import YOLO
from pose_model import KP_CONF, LW, RW, LH, RH, body

ap = argparse.ArgumentParser()
ap.add_argument("videos", nargs="+")
ap.add_argument("--conf", type=float, default=0.5)
ap.add_argument("--json", default=None)
args = ap.parse_args()

LE, RE = 7, 8
MAX_H, HOLD = 720, 1.0
METHODS = ["v2", "v2low", "traj", "traj3", "angle", "angle3w", "xnorm", "trajB", "angleB", "trajC", "angleC"]
WIN = {m: (2.0 if m == "angle3w" else 1.5) for m in METHODS}

def direction_changes(xs, amp):
    changes, last_dir, anchor = 0, 0, xs[0]
    for x in xs[1:]:
        d = x - anchor
        if abs(d) > amp:
            cur = 1 if d > 0 else -1
            if last_dir and cur != last_dir:
                changes += 1
            last_dir, anchor = cur, x
    return changes

def osc(vals, amp, need):
    if len(vals) < 6:
        return False
    c = direction_changes(vals, amp)
    return need <= c <= 15

def det_wave(m, st, now, n_win):
    """방식 m으로 지금 손흔들기인지"""
    W = WIN[m]
    if m in ("v2", "v2low"):
        h = [x for t, x in st[m] if now - t <= W]
        return len(h) >= 8 and len(h) >= 0.8 * n_win and osc(h, 0.15, 3)
    if m == "xnorm":
        for arm in (LW, RW):
            h = [x for t, x in st["xn"][arm] if now - t <= W]
            if len(h) >= 0.7 * n_win and osc(h, 0.15, 2):
                return True
        return False
    B = m in ("trajB", "angleB", "trajC", "angleC")
    C = m in ("trajC", "angleC")
    for arm in (LW, RW):
        h = [(d, a) for t, d, a, tog in st["arm"][arm] if now - t <= W and not (B and tog)]
        if len(h) < 6 or len(h) < (0.8 if B and not C else 0.7) * n_win:
            continue
        if m == "trajC":
            P = np.array([d for d, _ in h], dtype=np.float32)
            P -= P.mean(0)
            _, _, vt = np.linalg.svd(P, full_matrices=False)
            if osc(list(P @ vt[0]), 0.2, 2):
                return True
        elif m == "angleC":
            ang = np.degrees(np.unwrap(np.radians([a for _, a in h])))
            if osc(list(ang), 15.0, 2):
                return True
        elif m in ("trajB",):
            P = np.array([d for d, _ in h], dtype=np.float32)
            P -= P.mean(0)
            _, _, vt = np.linalg.svd(P, full_matrices=False)
            if osc(list(P @ vt[0]), 0.3, 2):
                return True
        elif m == "angleB":
            ang = np.degrees(np.unwrap(np.radians([a for _, a in h])))
            if osc(list(ang), 20.0, 2):
                return True
        elif m in ("traj", "traj3"):
            P = np.array([d for d, _ in h], dtype=np.float32)
            P -= P.mean(0)
            _, _, vt = np.linalg.svd(P, full_matrices=False)
            if osc(list(P @ vt[0]), 0.2, 2 if m == "traj" else 3):
                return True
        else:
            ang = np.degrees(np.unwrap(np.radians([a for _, a in h])))
            if osc(list(ang), 15.0, 2 if m == "angle" else 3):
                return True
    return False

det = YOLO("yolo11n-pose.pt")

def run(path):
    det.predictor = None
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cnt = collections.Counter()
    diag = collections.Counter()
    sw_ratio = []
    tracks = {}
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
            st = tracks.setdefault(tid, {"fr": collections.deque(maxlen=120), "v2": collections.deque(maxlen=120),
                                         "v2low": collections.deque(maxlen=120),
                                         "xn": {LW: collections.deque(maxlen=120), RW: collections.deque(maxlen=120)},
                                         "arm": {LW: collections.deque(maxlen=120), RW: collections.deque(maxlen=120)},
                                         "on": {m: False for m in METHODS}, "hold": {m: -9.0 for m in METHODS},
                                         "last": {m: -9.0 for m in METHODS}})
            b = body(k)
            diag["frames"] += 1
            if b is None:
                diag["no_shoulders"] += 1
                continue
            (cx, cy), sw = b
            torso = abs((k[LH, 1] + k[RH, 1]) / 2 - cy) if k[LH, 2] > KP_CONF and k[RH, 2] > KP_CONF else 2.5 * sw
            sw_ratio.append(abs(k[5, 0] - k[6, 0]) / max(torso, 1.0))
            st["fr"].append(now)
            w = LW if k[LW, 1] < k[RW, 1] else RW
            high = k[w, 2] > KP_CONF and k[w, 1] < cy - 0.3 * sw
            low = k[w, 2] > KP_CONF and k[w, 1] < cy + 0.2 * sw
            if high:
                st["v2"].append((now, (k[w, 0] - cx) / sw)); diag["v2_high"] += 1
            if high or low:
                st["v2low"].append((now, (k[w, 0] - cx) / sw))
            unit = max(abs(k[5, 0] - k[6, 0]), 0.4 * torso, 5.0)
            fa_mean = np.mean([max(np.linalg.norm(k[a, :2] - k[e, :2]), 5.0) for a, e in ((LW, LE), (RW, RE))])
            together = bool(k[LW, 2] > KP_CONF and k[RW, 2] > KP_CONF and np.linalg.norm(k[LW, :2] - k[RW, :2]) < 0.6 * fa_mean)
            for wj, ej in ((LW, LE), (RW, RE)):
                if k[wj, 2] > KP_CONF and k[ej, 2] > KP_CONF:
                    if k[wj, 1] < k[ej, 1]:
                        st["xn"][wj].append((now, (k[wj, 0] - cx) / unit))
                        if k[wj, 1] < cy + 0.25 * torso:
                            fa = max(np.linalg.norm(k[wj, :2] - k[ej, :2]), 5.0)
                            d = (k[wj, :2] - k[ej, :2]) / fa
                            st["arm"][wj].append((now, tuple(d), float(np.degrees(np.arctan2(d[1], d[0]))), together))
                            diag["elbow_up"] += 1
                else:
                    diag["low_conf_arm"] += 1
            n_win = {m: sum(1 for t in st["fr"] if now - t <= WIN[m]) for m in METHODS}
            # v2 진단: 손 올림 창 안 최대 방향 전환
            h = [x for t, x in st["v2"] if now - t <= 1.5]
            if len(h) >= 2:
                diag["v2_max_changes"] = max(diag["v2_max_changes"], direction_changes(h, 0.15))
            for m in METHODS:
                if det_wave(m, st, now, n_win[m]):
                    st["hold"][m] = now
                on = now - st["hold"][m] < HOLD
                if on and not st["on"][m] and now - st["last"][m] > HOLD:
                    cnt[m] += 1
                if on:
                    st["last"][m] = now
                st["on"][m] = on
    diag["sw_torso_median"] = round(float(np.median(sw_ratio)), 2) if sw_ratio else None
    return cnt, dict(diag), i / fps, i / max(time.time() - t0, 1e-3)

for vpath in args.videos:
    cnt, diag, sec, pf = run(vpath)
    res = {"video": os.path.basename(vpath), "sec": round(sec, 1), "fps": round(pf, 1),
           "wave": {m: cnt[m] for m in METHODS}, "diag": diag}
    print(json.dumps(res, ensure_ascii=False), flush=True)
    if args.json:
        with open(args.json, "a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
