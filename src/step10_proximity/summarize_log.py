# runs/<mode>_log.csv 를 트랙별로 요약: 최소 거리, 알람/접근 거리, 점프(>0.4m/샘플), 끊김(>0.8s)
import csv, math, sys
path, since = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0
rows = [r for r in csv.DictReader(open(path)) if float(r["t"]) >= since]
ev, tr = {}, {}
for r in rows:
    if r["type"] == "event":
        ev.setdefault(r["id"].lstrip("#"), []).append(f'{r["time"]} {r["label"]}')
    elif r["type"] == "person":
        tr.setdefault(r["id"], []).append(r)
for k, v in tr.items():
    d = [float(r["d"]) for r in v]
    t = [float(r["t"]) for r in v]
    xy = [(float(r["x"]), float(r["y"])) for r in v]
    chair = all(1.0 < x < 1.45 and y < -0.85 for x, y in xy)
    jumps = [f'{math.dist(xy[i], xy[i-1]):.2f}m@{v[i]["time"]}' for i in range(1, len(v)) if math.dist(xy[i], xy[i-1]) > 0.4]
    gaps = [f'{t[i]-t[i-1]:.1f}s@{v[i]["time"]}' for i in range(1, len(v)) if t[i] - t[i-1] > 0.8]
    print(f'#{k}{" (chair spot)" if chair else ""}: {v[0]["time"]}-{v[-1]["time"]} samples={len(v)} '
          f'd min {min(d):.2f} max {max(d):.2f} | x {min(p[0] for p in xy):.2f}..{max(p[0] for p in xy):.2f} '
          f'y {min(p[1] for p in xy):.2f}..{max(p[1] for p in xy):.2f} | zones {sorted(set(r["zone"] for r in v))}')
    print(f'   events: {ev.get(k, [])}')
    print(f'   jumps>0.4m: {jumps or "none"} | gaps>0.8s: {gaps or "none"}')
