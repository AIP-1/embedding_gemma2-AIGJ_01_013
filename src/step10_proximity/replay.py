"""지원님 기록(fused_log.csv, 0.3초마다 사람별 위치·타겟 거리 d·kind)을 트랙별로 다시 돌려 이탈 알람 수를 센다.
알람 로직은 앱의 ProximityAlarm 그대로 (근접 1.0m, 여유 0.2m, 속도 기준 0.3m/s)."""
import csv, sys, collections

class ProximityAlarm:   # proximity_app_cpu.py에서 TTC 부분만 뺌
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

rows = [r for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")) if r["type"] == "person" and r["d"]]
WIN = [("포인트 측정 등 (~15:04:50)", "00:00:00", "15:04:50"), ("10-6 1차 (15:04:51–15:06:35)", "15:04:51", "15:06:35"),
       ("10-6 사이 (15:06:36–15:10:32)", "15:06:36", "15:10:32"), ("10-6 2차 (15:10:33–15:12:43)", "15:10:33", "15:12:43"),
       ("그 뒤 (15:12:44~)", "15:12:44", "23:59:59")]
def win(tm):
    return next(n for n, a, b in WIN if a <= tm <= b)

VARIANTS = {"원래 (stereo만, far 3.0)": (False, 3.0), "A: single도 이탈 판단, far 3.0": (True, 3.0),
            "B: stereo만, far 2.0": (False, 2.0), "A+B: single도, far 2.0": (True, 2.0)}
res = {}
for name, (use_single, far) in VARIANTS.items():
    main, leave = {}, {}
    near_ev, leave_ev = collections.Counter(), []
    for r in rows:
        t, d, k, tid, tm = float(r["t"]), float(r["d"]), r["kind"], r["id"], r["time"]
        if k != "single":
            a = main.setdefault(tid, ProximityAlarm(1.0, far))
            for e in a.update(t, d):
                if e == "근접": near_ev[win(tm)] += 1
                if e == "이탈": leave_ev.append((tm, tid, k, round(d, 2)))
        if use_single:
            b = leave.setdefault(tid, ProximityAlarm(1.0, far))
            for e in b.update(t, d):
                if e == "이탈" and k == "single" and main.get(tid) and main[tid].zone != "far":
                    main[tid].zone = "far"; leave_ev.append((tm, tid, k, round(d, 2)))
                elif e == "이탈" and k == "single" and tid not in main:
                    leave_ev.append((tm, tid, k, round(d, 2)))
    res[name] = (near_ev, leave_ev)
    print(f"\n== {name}")
    print("  근접 알람(재계산, 중복 억제 전):", dict(near_ev))
    by = collections.Counter(win(e[0]) for e in leave_ev)
    print("  이탈 알람:", dict(by))
    for e in leave_ev:
        print("   ", win(e[0]), e)

# 트랙별 최대 거리 (2차 구간)
print("\n2차 구간 트랙별 d 범위 / kind 수")
tr = collections.defaultdict(list)
for r in rows:
    if "15:10:33" <= r["time"] <= "15:12:43":
        tr[r["id"]].append((float(r["d"]), r["kind"]))
for k, v in sorted(tr.items(), key=lambda x: int(x[0])):
    print(f"  #{k}: d {min(x[0] for x in v):.2f}–{max(x[0] for x in v):.2f}, stereo {sum(x[1]!='single' for x in v)}, single {sum(x[1]=='single' for x in v)}")
print("\n전체 기록 d 최대: stereo %.2f, single %.2f" % (max(float(r['d']) for r in rows if r['kind'] != 'single'),
                                                  max(float(r['d']) for r in rows if r['kind'] == 'single')))
