# /state 폴링 로거: 앱은 건드리지 않고 0.3초마다 상태를 runs/<mode>_log.csv 에 남긴다
import csv, json, os, re, time, urllib.request
URL = "http://127.0.0.1:8020/state"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
COLS = ["t", "time", "mode", "type", "id", "x", "y", "h", "d", "zone", "trend", "closing", "ttc", "danger", "kind", "label", "fps"]
seen, files = set(), {}

def writer(mode):
    name = re.sub(r"[^a-z0-9]+", "_", mode.lower()).strip("_") or "unknown"
    if name not in files:
        path = os.path.join(OUT, f"{name}_log.csv")
        new = not os.path.exists(path)
        f = open(path, "a", newline="", buffering=1)
        w = csv.DictWriter(f, COLS)
        if new: w.writeheader()
        files[name] = (f, w)
    return files[name][1]

while True:
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(URL, timeout=2))
        mode = r.get("stats", {}).get("mode") or "unknown"
        fps = r.get("stats", {}).get("fps")
        w = writer(mode)
        for e in reversed(r.get("events", [])):
            k = (e["ts"], e["who"], e["label"])
            if k in seen: continue
            seen.add(k)
            w.writerow({"t": round(e["ts"], 3), "time": time.strftime("%H:%M:%S", time.localtime(e["ts"])),
                        "mode": mode, "type": "event", "id": e["who"], "label": e["label"]})
        for p in r.get("people", []):
            w.writerow({"t": round(t0, 3), "time": time.strftime("%H:%M:%S", time.localtime(t0)), "mode": mode,
                        "type": "person", "id": p["id"], "x": p["x"], "y": p["y"], "h": p.get("h"), "d": p.get("d"),
                        "zone": p.get("zone"), "trend": p.get("trend"), "closing": p.get("closing"), "ttc": p.get("ttc"),
                        "danger": p.get("danger"), "kind": p.get("kind"), "fps": fps})
    except Exception as ex:
        w = writer("unknown")
        w.writerow({"t": round(t0, 3), "time": time.strftime("%H:%M:%S", time.localtime(t0)), "type": "error", "label": repr(ex)[:200]})
    time.sleep(max(0, 0.3 - (time.time() - t0)))
