"""9단계 악수 오판정 비교: 같은 검출 결과로 악수 판정 조건 여러 개를 한 번에 센다(악수 횟수).
방식(열)
  hs2 : pose_app_v2/v3의 악수 (두 사람 중심이 어깨 너비 0.8배 이상 떨어짐 + 가까운 두 손목(어깨 너비 0.5배 안)이
        가슴~허리 높이 + 각자 상대 쪽으로 어깨 너비 0.3배 이상 뻗음, 0.5초 이상, 같은 자리 2초 쿨다운)
  hsA : hs2 + 맞잡은 두 손목이 둘 다 팔꿈치보다 '많이 위'가 아님(손목 y >= 팔꿈치 y - 아래팔 0.3배): 손 흔들기·손 들기 제외
  hsB : hsA + 맞잡은 손목 중간점이 두 사람 몸 중심 사이(가운데 70% 안)
  hsC : hsB + 두 사람 사이 거리 조건을 어깨 너비 0.8배 → 0.5배로 완화(겹쳐 선 영접 장면 놓침 줄이기)
  hsD : hsC + 그 순간 두 사람 중 한쪽이라도 손흔들기(angleC) 중이면 제외
  hsE : hsD의 조건을 측면에 맞게: 거리 단위를 사람마다 max(어깨 너비, 몸통 길이 0.4배)로, 손 높이는 어깨~엉덩이(+몸통 0.3배)
        (악수는 대개 옆으로 마주 서서 하므로 카메라에는 측면 → 어깨 너비가 좁아 어깨 너비 기준이 어긋남)
  hsF : hsE에서 두 손목 거리 기준을 단위 0.5배 → 0.8배로 완화
  hsG : hsD + 두 사람 중 한쪽이라도 박수 자세(맞잡은 쪽 손목이 자기 다른 손목과 아래팔 0.6배 안)면 제외
        (나란히 서서 박수 치는 옆 사람끼리 악수로 세는 문제: 실시간 재현에서 사무실 박수 영상 2회 반복에 악수 5회)
  hsH : hsD + 두 사람 중 한쪽이라도 '박수 치는 사람'(최근 1.5초 프레임의 25% 이상에서 두 손목이 아래팔 0.6배 안으로 붙음)이면 제외
        (박수는 손을 붙였다 떼므로 순간 자세(hsG)로는 못 거름)
--stride N: N프레임마다 1장만 처리(실시간 화면처럼 프레임을 건너뛸 때 재현)
사용: python pose_eval_v4.py <영상> --json out.jsonl
"""
import argparse, collections, json, os, time
import cv2, numpy as np
from ultralytics import YOLO
from pose_model import KP_CONF, LW, RW, LH, RH, body

ap = argparse.ArgumentParser()
ap.add_argument("videos", nargs="+")
ap.add_argument("--conf", type=float, default=0.5)
ap.add_argument("--json", default=None)
ap.add_argument("--stride", type=int, default=1)
args = ap.parse_args()

LE, RE = 7, 8
MAX_H, SMOOTH_S, WAVE_S = 720, 0.6, 1.5
HS_MIN_S, HS_COOL_S = 0.5, 2.0
METHODS = ["hs2", "hsA", "hsB", "hsC", "hsD", "hsE", "hsF", "hsG", "hsH"]

def majority(dq, now, window):
    vals = [v for t, v in dq if now - t <= window]
    if not vals:
        return None, 0.0
    v, n = collections.Counter(vals).most_common(1)[0]
    return v, n / len(vals)

def direction_changes(xs, amp):
    changes, last_dir, anchor = 0, 0, xs[0]
    for x in xs[1:]:
        if abs(x - anchor) > amp:
            cur = 1 if x > anchor else -1
            if last_dir and cur != last_dir:
                changes += 1
            last_dir, anchor = cur, x
    return changes

def update_wave(st, k, b, now):
    """pose_app_v3.py의 angleC와 같은 손흔들기 판정. 손흔들기 중이면 st['wave_t'] 갱신"""
    (cx, cy), sw = b
    st["fr"].append(now)
    torso = abs((k[LH, 1] + k[RH, 1]) / 2 - cy) if k[LH, 2] > KP_CONF and k[RH, 2] > KP_CONF else 2.5 * sw
    fal = [max(np.linalg.norm(k[a, :2] - k[e, :2]), 5.0) for a, e in ((LW, LE), (RW, RE))]
    together = k[LW, 2] > KP_CONF and k[RW, 2] > KP_CONF and np.linalg.norm(k[LW, :2] - k[RW, :2]) < 0.6 * np.mean(fal)
    st["tog"].append((now, bool(together)))
    h = [x for t, x in st["tog"] if now - t <= WAVE_S]
    st["clapping"] = len(h) >= 5 and sum(h) >= 0.25 * len(h)
    for (wj, ej), fl in zip(((LW, LE), (RW, RE)), fal):
        if (not together and k[wj, 2] > KP_CONF and k[ej, 2] > KP_CONF
                and k[wj, 1] < k[ej, 1] and k[wj, 1] < cy + 0.25 * torso):
            d = (k[wj, :2] - k[ej, :2]) / fl
            st["arm"][wj].append((now, float(np.degrees(np.arctan2(d[1], d[0])))))
    n_win = sum(1 for t in st["fr"] if now - t <= WAVE_S)
    for wj in (LW, RW):
        h = [a for t, a in st["arm"][wj] if now - t <= WAVE_S]
        if len(h) >= 6 and len(h) >= 0.7 * n_win:
            c = direction_changes(list(np.degrees(np.unwrap(np.radians(h)))), 15.0)
            if 2 <= c <= 15:
                st["wave_t"] = now

def pair_flags(a, b, ba, bb, wave_a, wave_b):
    """한 쌍에 대해 방식별 후보 여부와 중간점"""
    sw = (ba[1] + bb[1]) / 2
    best, bw = 1e9, None
    for wa in (LW, RW):
        for wb in (LW, RW):
            if a[wa, 2] > KP_CONF and b[wb, 2] > KP_CONF:
                mid_y = (a[wa, 1] + b[wb, 1]) / 2
                if ba[0][1] < mid_y < ba[0][1] + 2.5 * ba[1]:
                    d = np.linalg.norm(a[wa, :2] - b[wb, :2])
                    if d < best:
                        best, bw = d, (wa, wb)
    out = {m: False for m in METHODS}
    s = 1 if bb[0][0] > ba[0][0] else -1
    gap = abs(ba[0][0] - bb[0][0])
    lo, hi = sorted((ba[0][0], bb[0][0]))
    between = lambda m: lo + 0.15 * (hi - lo) <= m[0] <= hi - 0.15 * (hi - lo)
    def not_raised(k, w):
        e = LE if w == LW else RE
        if k[e, 2] <= KP_CONF:
            return True
        fl = max(np.linalg.norm(k[w, :2] - k[e, :2]), 5.0)
        return k[w, 1] >= k[e, 1] - 0.3 * fl
    mid = None
    if bw is not None and best < 0.5 * sw:
        wa, wb = bw
        mid = (a[wa, :2] + b[wb, :2]) / 2
        reach = (a[wa, 0] - ba[0][0]) * s >= 0.3 * ba[1] and (b[wb, 0] - bb[0][0]) * -s >= 0.3 * bb[1]
        low = not_raised(a, wa) and not_raised(b, wb)
        out["hs2"] = reach and gap >= 0.8 * sw
        out["hsA"] = out["hs2"] and low
        out["hsB"] = out["hsA"] and between(mid)
        out["hsC"] = reach and gap >= 0.5 * sw and low and between(mid)
        out["hsD"] = out["hsC"] and not (wave_a or wave_b)
        def clapping(k, w):
            o = RW if w == LW else LW
            e = LE if w == LW else RE
            if k[o, 2] <= KP_CONF:
                return False
            fl = max(np.linalg.norm(k[w, :2] - k[e, :2]), 5.0) if k[e, 2] > KP_CONF else 0.5 * sw
            return np.linalg.norm(k[w, :2] - k[o, :2]) < 0.6 * fl
        out["hsG"] = out["hsD"] and not (clapping(a, wa) or clapping(b, wb))
    # hsE/hsF: 측면용 단위(max(어깨 너비, 몸통 0.4배))와 몸통 기준 높이
    def unit_torso(k, bx):
        (cx, cy), w = bx
        torso = abs((k[LH, 1] + k[RH, 1]) / 2 - cy) if k[LH, 2] > KP_CONF and k[RH, 2] > KP_CONF else 2.5 * w
        return max(w, 0.4 * torso), cy, cy + 1.3 * torso
    ua, ya0, ya1 = unit_torso(a, ba)
    ub, yb0, yb1 = unit_torso(b, bb)
    u = (ua + ub) / 2
    best2, bw2 = 1e9, None
    for wa2 in (LW, RW):
        for wb2 in (LW, RW):
            if a[wa2, 2] > KP_CONF and b[wb2, 2] > KP_CONF:
                my = (a[wa2, 1] + b[wb2, 1]) / 2
                if max(ya0, yb0) - 0.2 * u < my < min(ya1, yb1):
                    d2 = np.linalg.norm(a[wa2, :2] - b[wb2, :2])
                    if d2 < best2:
                        best2, bw2 = d2, (wa2, wb2)
    if bw2 is not None:
        wa2, wb2 = bw2
        m2 = (a[wa2, :2] + b[wb2, :2]) / 2
        ok = ((a[wa2, 0] - ba[0][0]) * s >= 0.3 * ua and (b[wb2, 0] - bb[0][0]) * -s >= 0.3 * ub
              and gap >= 0.5 * u and not_raised(a, wa2) and not_raised(b, wb2) and between(m2)
              and not (wave_a or wave_b))
        out["hsE"] = ok and best2 < 0.5 * u
        out["hsF"] = ok and best2 < 0.8 * u
        if mid is None and out["hsF"]:
            mid = m2
    return out, mid, max(sw, u)

det = YOLO("yolo11n-pose.pt")

def run(path):
    det.predictor = None
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cnt = collections.Counter()
    people = {}
    pairs = {m: {} for m in METHODS}
    start = {m: {} for m in METHODS}
    log = {m: [] for m in METHODS}
    i, t0 = 0, time.time()
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fr.shape[0] > MAX_H:
            fr = cv2.resize(fr, (int(fr.shape[1] * MAX_H / fr.shape[0]), MAX_H))
        now = i / fps
        i += 1
        if (i - 1) % args.stride:
            continue
        r = det.track(fr, persist=True, conf=args.conf, half=True, verbose=False)[0]
        if r.boxes.id is None or r.keypoints is None:
            continue
        kps = np.concatenate([r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()[..., None]], axis=-1)
        ids = r.boxes.id.int().tolist()
        bodies = [body(k) for k in kps]
        waving, clapping = [], []
        for tid, k, b in zip(ids, kps, bodies):
            st = people.setdefault(tid, {"fr": collections.deque(maxlen=90), "wave_t": -9.0, "tog": collections.deque(maxlen=90),
                                         "clapping": False,
                                         "arm": {LW: collections.deque(maxlen=90), RW: collections.deque(maxlen=90)}})
            if b is not None:
                update_wave(st, k, b, now)
            waving.append(now - st["wave_t"] < 1.0)
            clapping.append(st["clapping"])
        seen = set()
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                if bodies[x] is None or bodies[y] is None:
                    continue
                key = tuple(sorted((ids[x], ids[y])))
                seen.add(key)
                flags, mid, sw = pair_flags(kps[x], kps[y], bodies[x], bodies[y], waving[x], waving[y])
                flags["hsH"] = flags["hsD"] and not (clapping[x] or clapping[y])
                for m in METHODS:
                    dq = pairs[m].setdefault(key, collections.deque(maxlen=60))
                    dq.append((now, flags[m]))
                    v, frac = majority(dq, now, SMOOTH_S)
                    if bool(v) and frac >= 0.6 and mid is not None:
                        st0 = start[m].setdefault(key, now)
                        if st0 is not None and now - st0 >= HS_MIN_S:
                            if not any(now - t < HS_COOL_S and np.linalg.norm(p - mid) < 2 * sw for t, p in log[m]):
                                cnt[m] += 1
                            log[m].append((now, mid))
                            start[m][key] = None
                    elif not (bool(v) and frac >= 0.6):
                        start[m].pop(key, None)
        for m in METHODS:
            for key in list(pairs[m]):
                if key not in seen:
                    pairs[m].pop(key); start[m].pop(key, None)
    return cnt, i / fps, i / max(time.time() - t0, 1e-3)

for vpath in args.videos:
    cnt, sec, pf = run(vpath)
    res = {"video": os.path.basename(vpath), "sec": round(sec, 1), "fps": round(pf, 1), "handshake": {m: cnt[m] for m in METHODS}}
    print(json.dumps(res, ensure_ascii=False), flush=True)
    if args.json:
        with open(args.json, "a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
