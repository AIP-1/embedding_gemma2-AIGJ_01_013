# Jetson proximity_app /state 를 0.3초마다 보고, 새 '알람' 이벤트마다 맥에서 경보음 3번 (0.4초 간격) — 브라우저 소리 대신
import json, subprocess, time, urllib.request
URL = "http://100.105.144.42:8020/state"
RING = "for i in 1 2 3; do afplay -v 3 /System/Library/Sounds/Sosumi.aiff; sleep 0.4; done"
last, player = None, None
while True:
    try:
        ev = json.load(urllib.request.urlopen(URL, timeout=2)).get("events", [])
        if last is None:                       # 시작 이전 알람은 울리지 않음
            last = max([e["ts"] for e in ev], default=0)
        new = [e for e in ev if e["ts"] > last and e["label"].startswith("알람")]
        if ev:
            last = max(last, max(e["ts"] for e in ev))
        for e in reversed(new):
            print(time.strftime("%H:%M:%S", time.localtime(e["ts"])), e["who"], e["label"], flush=True)
        if new and (player is None or player.poll() is not None):   # 울리는 중이면 겹치지 않게 건너뜀
            player = subprocess.Popen(["sh", "-c", RING])
    except Exception as ex:
        print(time.strftime("%H:%M:%S"), "error", repr(ex)[:120], flush=True)
        time.sleep(2)
    time.sleep(0.3)
