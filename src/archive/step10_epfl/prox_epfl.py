"""10단계를 EPFL 실험실 6인 영상(카메라 0·1)으로 재현: 10-3 ankle1 / 10-4 stereo / 10-5 fused 위치 정확도와 10-6 접근 알람
README의 proximity_app.py와 같은 순서(관절 → 바닥 위치 → 추적 → ProximityAlarm)를 따르되, 카메라 모델만 EPFL 보정으로 바꿈:
  EPFL은 카메라마다 '영상 → 바닥(높이 0) 위 좌표'와 '영상 → 머리 높이(1.75m) 평면 위 좌표' 변환(호모그래피, cm)을 준다.
  한 픽셀을 두 평면으로 옮긴 두 점은 같은 시선 위에 있으므로, 그 시선(3D 직선)을 정확히 얻는다 → README의 ray_plane·삼각측량을 그대로 대신함.
정답: gt_lab_6p.txt (25프레임마다 6명 위치, 56x56 격자, 바닥 3.58 x 3.60m)
사용: python prox_epfl.py --mode eval            # 정답 프레임에서 세 방식의 위치 오차
      python prox_epfl.py --mode alarm --method fused   # 영상 전체로 추적·접근 알람, 정답으로 만든 알람과 비교
"""
import argparse, collections, json, os, time
import cv2, numpy as np
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["eval", "alarm"], default="eval")
ap.add_argument("--method", choices=["ankle1", "stereo", "fused"], default="fused")
ap.add_argument("--dir", default=os.path.expanduser("~/eg2-lab/media/prox_epfl"))
ap.add_argument("--cams", type=int, nargs=2, default=[0, 1])
ap.add_argument("--h-guess", type=float, default=1.2, help="fused: 한 카메라에만 보일 때 가정할 어깨 높이(m), README 1.2")
ap.add_argument("--target-xy", default="1.79,1.80", help="가상 타겟 바닥 위치(m), 기본 방 가운데")
ap.add_argument("--near", type=float, default=0.8)
ap.add_argument("--far", type=float, default=1.6)
ap.add_argument("--stride", type=int, default=2, help="alarm: N프레임마다 1장 (25fps/2 = 12.5fps, 실제 앱 수준)")
ap.add_argument("--json", default=None)
args = ap.parse_args()

KP_CONF = 0.5
NOSE, LS, RS, LA, RA = 0, 5, 6, 15, 16
HEAD_Z = 1.75
ROOM = (-0.3, -0.3, 3.88, 3.90)              # 바닥 3.58 x 3.60m + 여유
GATE, SMOOTH_S = 0.8, 0.5                     # README와 같음
RAY_MAX = 0.3                                 # 두 시선 사이 최소 거리 기준(m), README의 재투영 오차 15px 대신
FPS = 25.0

# ---------------- EPFL 보정 ----------------
def load_calib(path):
    """'# Camera N' 제목으로 나눠 읽음. 숫자 18개면 바닥·머리 평면 변환, 10개면 바닥 변환 + 머리 평면이 지나는 영상 높이(%)"""
    secs, cur = [], None
    for line in open(path):
        if line.startswith("#") and "Camera" in line:
            cur = []; secs.append(cur); continue
        line = line.split("#")[0].strip()
        if line and cur is not None:
            cur += [float(x) for x in line.split()]
    cams = []
    for nums in secs:
        g = np.array(nums[:9]).reshape(3, 3)
        h = np.array(nums[9:18]).reshape(3, 3) if len(nums) >= 18 else None
        cams.append({"Hg": g, "Hh": h})
    return cams

def to_plane(H, p):
    x = H @ np.array([p[0], p[1], 1.0])
    return x[:2] / x[2] / 100.0                # cm → m

def ray(cam, p):
    """픽셀 p를 지나는 시선: (바닥 위 점 G(z=0), 머리 평면 위 점 Hd(z=1.75))"""
    G = np.append(to_plane(cam["Hg"], p), 0.0)
    Hd = np.append(to_plane(cam["Hh"], p), HEAD_Z)
    return G, Hd

def ray_plane(cam, p, h):
    G, Hd = ray(cam, p)
    return G + (Hd - G) * (h / HEAD_Z)

def triangulate(c0, c1, p0, p1):
    """두 시선의 가장 가까운 두 점의 중간 → 3D 점, 두 시선 사이 거리(m)"""
    a0, b0 = ray(c0, p0); a1, b1 = ray(c1, p1)
    d0, d1 = b0 - a0, b1 - a1
    w = a0 - a1
    A = np.array([[d0 @ d0, -d0 @ d1], [d0 @ d1, -d1 @ d1]])
    if abs(np.linalg.det(A)) < 1e-9:
        return None, 1e9
    s, t = np.linalg.solve(A, [-w @ d0, -w @ d1])
    q0, q1 = a0 + s * d0, a1 + t * d1
    return (q0 + q1) / 2, float(np.linalg.norm(q0 - q1))

def upper_body(k):
    if k[LS, 2] > KP_CONF and k[RS, 2] > KP_CONF:
        return (k[LS, :2] + k[RS, :2]) / 2
    if k[NOSE, 2] > KP_CONF:
        return k[NOSE, :2].copy()
    return None

def ankle(k):
    vis = [k[i, :2] for i in (LA, RA) if k[i, 2] > KP_CONF]
    return np.mean(vis, axis=0) if vis else None

def match_people(c0, c1, pts0, pts1, h_range=(0.5, 2.1)):
    cands = []
    for i, a in enumerate(pts0):
        for j, b in enumerate(pts1):
            X, err = triangulate(c0, c1, a, b)
            if X is not None and err <= RAY_MAX and h_range[0] <= X[2] <= h_range[1]:
                cands.append((err, i, j, X))
    cands.sort(key=lambda c: c[0])
    used0, used1, out = set(), set(), []
    for err, i, j, X in cands:
        if i in used0 or j in used1:
            continue
        out.append((i, j, X, err)); used0.add(i); used1.add(j)
    return out, set(range(len(pts0))) - used0, set(range(len(pts1))) - used1

def observations(cams, kps, method, h_guess):
    """README process_loop와 같은 순서로 [(xy, h, kind)]"""
    obs = []
    if method == "ankle1":
        for k in kps[0]:
            a = ankle(k)
            if a is not None:
                obs.append((to_plane(cams[0]["Hg"], a), 0.0, "ankle"))
    else:
        pts = [[upper_body(k) for k in kk] for kk in kps]
        pts = [[p for p in pp if p is not None] for pp in pts]
        matched, rest0, rest1 = match_people(cams[0], cams[1], pts[0], pts[1])
        for i, j, X, err in matched:
            obs.append((X[:2], float(X[2]), "stereo"))
        if method == "fused":
            for n, rest in ((0, rest0), (1, rest1)):
                for i in rest:
                    xy = ray_plane(cams[n], pts[n][i], h_guess)[:2]
                    if all(np.linalg.norm(xy - o[0]) > 0.6 for o in obs):
                        obs.append((xy, h_guess, "single"))
    x0, y0, x1, y1 = ROOM
    return [o for o in obs if x0 <= o[0][0] <= x1 and y0 <= o[0][1] <= y1]

# ---------------- 정답 ----------------
def load_gt(path):
    rows = [l.split() for l in open(path) if l.strip()]
    nf, npl, gw, gh, step, first, last = map(int, rows[1])
    out = {}
    for f, line in enumerate(rows[2:]):          # 프레임마다 한 줄, 정답은 step(25)프레임마다
        if f < first or f > last or (f - first) % step:
            continue
        ppl = {}
        for pid, v in enumerate(map(int, line[:npl])):
            if v >= 0:
                ppl[pid] = np.array([((v % gw) + 0.5) * (358 / gw), ((v // gw) + 0.5) * (360 / gh)]) / 100.0
        out[f] = ppl
    return out

# ---------------- 10-6 알람 (README 그대로) ----------------
class ProximityAlarm:
    def __init__(self, near, far, margin=0.2, v_th=0.3):
        self.near, self.far, self.m, self.v_th = near, far, margin, v_th
        self.zone, self.hist, self.trend = "mid", [], "정지"
    def update(self, t, d):
        self.hist = [(tt, dd) for tt, dd in self.hist if t - tt < 1.0] + [(t, d)]
        span = self.hist[-1][0] - self.hist[0][0]
        v = (self.hist[-1][1] - self.hist[0][1]) / span if span > 0.3 else 0.0
        trend = "접근 중" if v < -self.v_th else "멀어지는 중" if v > self.v_th else "정지"
        ev = []
        if trend == "접근 중" and self.trend != "접근 중" and d < self.far:
            ev.append("접근 시작")
        self.trend = trend
        if self.zone != "near" and d < self.near:
            self.zone = "near"; ev.append("근접")
        elif self.zone == "near" and d > self.near + self.m:
            self.zone = "mid"
        if self.zone != "far" and d > self.far:
            if self.zone == "mid" and self.hist[0][1] <= self.far:
                ev.append("이탈")
            self.zone = "far"
        elif self.zone == "far" and d < self.far - self.m:
            self.zone = "mid"
        return ev

class Track:
    _next = 1
    def __init__(self, xy, h, kind, now):
        self.id, Track._next = Track._next, Track._next + 1
        self.hist = collections.deque([(now, xy, h)], maxlen=60)
        self.kind, self.last, self.first = kind, now, now
        self.alarm = ProximityAlarm(args.near, args.far)
    def observe(self, xy, h, kind, now):
        self.hist.append((now, xy, h)); self.kind, self.last = kind, now
    @property
    def xy(self):
        return np.median([p for t, p, _ in self.hist if self.last - t <= SMOOTH_S], axis=0)

def update_tracks(tracks, obs, now):
    free = list(range(len(tracks)))
    for xy, h, kind in sorted(obs, key=lambda o: o[2] != "stereo"):
        best, bi = GATE, None
        for i in free:
            d = np.linalg.norm(tracks[i].xy - xy)
            if d < best:
                best, bi = d, i
        if bi is None:
            tracks.append(Track(xy, h, kind, now))
        else:
            tracks[bi].observe(xy, h, kind, now); free.remove(bi)
    for tr in [t for t in tracks if now - t.last > 2.0]:
        tracks.remove(tr)

# ---------------- 실행 ----------------
cams_all = load_calib(os.path.join(args.dir, "calibration-6p.txt"))
cams = [cams_all[c] for c in args.cams]
assert all(c["Hh"] is not None for c in cams), "머리 평면 변환이 없는 카메라"
gt = load_gt(os.path.join(args.dir, "gt_lab_6p.txt"))
det = YOLO("yolo11n-pose.pt")
caps = [cv2.VideoCapture(os.path.join(args.dir, f"6p-c{c}.avi")) for c in args.cams]
n_total = int(min(c.get(cv2.CAP_PROP_FRAME_COUNT) for c in caps))

def detect(frames):
    rs = det.predict(frames, conf=0.5, half=True, verbose=False)
    return [np.concatenate([r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()[..., None]], -1)
            if r.keypoints is not None and r.keypoints.conf is not None else np.zeros((0, 17, 3)) for r in rs]

def match_gt(est, truth, max_d=0.75):
    pairs = sorted((np.linalg.norm(e - g), i, j) for i, e in enumerate(est) for j, g in enumerate(truth))
    ue, ug, errs = set(), set(), []
    for d, i, j in pairs:
        if d > max_d or i in ue or j in ug:
            continue
        ue.add(i); ug.add(j); errs.append(d)
    return errs, len(est) - len(ue), len(truth) - len(ug)

t0 = time.time()
if args.mode == "eval":
    variants = {"ankle1(cam0)": ("ankle1", None), "stereo": ("stereo", None),
                f"fused(h={args.h_guess})": ("fused", args.h_guess), "fused(h=1.45)": ("fused", 1.45)}
    acc = {k: {"errs": [], "fp": 0, "miss": 0, "gt": 0, "single_errs": []} for k in variants}
    for f in sorted(gt):
        if f >= n_total or not gt[f]:
            continue
        frames = []
        for c in caps:
            c.set(cv2.CAP_PROP_POS_FRAMES, f); ok, fr = c.read(); frames.append(fr if ok else None)
        if any(x is None for x in frames):
            continue
        kps = detect(frames)
        truth = list(gt[f].values())
        for name, (m, h) in variants.items():
            obs = observations(cams, kps, m, h if h else args.h_guess)
            errs, fp, miss = match_gt([o[0] for o in obs], truth)
            a = acc[name]; a["errs"] += errs; a["fp"] += fp; a["miss"] += miss; a["gt"] += len(truth)
            if m == "fused":
                singles = [o[0] for o in obs if o[2] == "single"]
                a["single_errs"] += match_gt(singles, truth)[0]
    res = {}
    for name, a in acc.items():
        e = np.array(a["errs"]) if a["errs"] else np.array([np.nan])
        res[name] = {"gt_people": a["gt"], "found": len(a["errs"]), "recall": round(len(a["errs"]) / max(a["gt"], 1), 3),
                     "mean_err_m": round(float(np.nanmean(e)), 3), "median_err_m": round(float(np.nanmedian(e)), 3),
                     "p90_err_m": round(float(np.nanpercentile(e, 90)), 3), "false_pos": a["fp"],
                     "single_mean_err_m": round(float(np.mean(a["single_errs"])), 3) if a["single_errs"] else None,
                     "single_n": len(a["single_errs"])}
        print(name, res[name], flush=True)
    out = {"mode": "eval", "cams": args.cams, "frames": len(gt), "sec": round(time.time() - t0), "results": res}
else:
    tgt = np.array([float(v) for v in args.target_xy.split(",")])
    tracks, events, n_obs, kinds = [], collections.Counter(), 0, collections.Counter()
    sys_ev, sys_near = [], {}          # (시각, 알람), 프레임마다 근접 구역에 트랙이 있는지
    for f in range(0, n_total, args.stride):
        frames = []
        for c in caps:
            c.set(cv2.CAP_PROP_POS_FRAMES, f); ok, fr = c.read(); frames.append(fr if ok else None)
        if any(x is None for x in frames):
            break
        now = f / FPS
        obs = observations(cams, detect(frames), args.method, args.h_guess)
        n_obs += len(obs); kinds.update(o[2] for o in obs)
        update_tracks(tracks, obs, now)
        for tr in tracks:
            for ev in tr.alarm.update(now, float(np.linalg.norm(tr.xy - tgt))):
                events[ev] += 1; sys_ev.append((now, ev))
        sys_near[f] = any(tr.alarm.zone == "near" for tr in tracks)
    # 정답으로 같은 알람 (1초마다 6명 위치, 사람마다 따로)
    gt_events, gt_alarm, gt_ev = collections.Counter(), {}, []
    for f in sorted(gt):
        for pid, xy in gt[f].items():
            al = gt_alarm.setdefault(pid, ProximityAlarm(args.near, args.far))
            for ev in al.update(f / FPS, float(np.linalg.norm(xy - tgt))):
                gt_events[ev] += 1; gt_ev.append((f / FPS, ev))
    # 시각 비교 (정답은 1초 간격이라 ±2초 허용)
    near_t = [s_ for s_, e in sys_ev if e == "근접"]; gnear_t = [s_ for s_, e in gt_ev if e == "근접"]
    gt_min = {f: min([float(np.linalg.norm(xy - tgt)) for xy in gt[f].values()] or [9.9]) for f in gt if f < n_total}
    def gt_close(t, tol=2.0, dmax=args.near + 0.2):
        return any(abs(f / FPS - t) <= tol and d <= dmax for f, d in gt_min.items())
    tp_sys = sum(gt_close(t) for t in near_t)
    tp_gt = sum(any(abs(t - g) <= 2.0 for t in near_t) for g in gnear_t)
    st = collections.Counter()
    for f, d in gt_min.items():
        fs_ = f if f in sys_near else f - 1
        if fs_ in sys_near:
            st[(d <= args.near, sys_near[fs_])] += 1
    timing = {"near_alarm_precision": round(tp_sys / max(len(near_t), 1), 3), "near_alarm_recall": round(tp_gt / max(len(gnear_t), 1), 3),
              "sec_both_near": st[(True, True)], "sec_gt_only": st[(True, False)], "sec_sys_only": st[(False, True)], "sec_both_far": st[(False, False)]}
    out = {"mode": "alarm", "method": args.method, "h_guess": args.h_guess, "target": tgt.tolist(), "near": args.near,
           "far": args.far, "fps_input": FPS / args.stride, "frames": n_total, "proc_sec": round(time.time() - t0),
           "tracks_created": Track._next - 1, "obs_per_frame": round(n_obs / max(n_total // args.stride, 1), 2),
           "obs_kinds": dict(kinds), "events": dict(events), "gt_events_1hz": dict(gt_events), "timing": timing}
    print(out, flush=True)
if args.json:
    with open(args.json, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(out, ensure_ascii=False) + "\n")
