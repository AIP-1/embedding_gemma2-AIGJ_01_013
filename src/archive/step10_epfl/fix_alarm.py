# 알람 결과에 시각 비교 추가: 근접 알람이 정답과 ±2초 안에 맞는지, 초마다 "가까이 있는 사람 있음" 상태 일치
p = "prox_epfl.py"; s = open(p, encoding="utf-8").read()
r = [
("""    tracks, events, n_obs, kinds = [], collections.Counter(), 0, collections.Counter()
""",
"""    tracks, events, n_obs, kinds = [], collections.Counter(), 0, collections.Counter()
    sys_ev, sys_near = [], {}          # (시각, 알람), 프레임마다 근접 구역에 트랙이 있는지
"""),
("""            for ev in tr.alarm.update(now, float(np.linalg.norm(tr.xy - tgt))):
                events[ev] += 1
""",
"""            for ev in tr.alarm.update(now, float(np.linalg.norm(tr.xy - tgt))):
                events[ev] += 1; sys_ev.append((now, ev))
        sys_near[f] = any(tr.alarm.zone == "near" for tr in tracks)
"""),
("""            for ev in al.update(f / FPS, float(np.linalg.norm(xy - tgt))):
                gt_events[ev] += 1
""",
"""            for ev in al.update(f / FPS, float(np.linalg.norm(xy - tgt))):
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
        if f in sys_near:
            st[(d <= args.near, sys_near[f])] += 1
    timing = {"near_alarm_precision": round(tp_sys / max(len(near_t), 1), 3), "near_alarm_recall": round(tp_gt / max(len(gnear_t), 1), 3),
              "sec_both_near": st[(True, True)], "sec_gt_only": st[(True, False)], "sec_sys_only": st[(False, True)], "sec_both_far": st[(False, False)]}
"""),
("""    gt_events, gt_alarm = collections.Counter(), {}
""",
"""    gt_events, gt_alarm, gt_ev = collections.Counter(), {}, []
"""),
("""           "obs_kinds": dict(kinds), "events": dict(events), "gt_events_1hz": dict(gt_events)}
""",
"""           "obs_kinds": dict(kinds), "events": dict(events), "gt_events_1hz": dict(gt_events), "timing": timing}
"""),
]
for a, b in r:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s); print("patched")
