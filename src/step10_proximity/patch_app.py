# 지원님 proximity_app_cpu.py 사본에 이탈 알람 수정 넣기 → proximity_app_leave.py
#  A) --leave-single: 한 카메라(single) 위치도 이탈 판단에만 씀 (근접 알람은 그대로 stereo만)
#  B) --far 기본값 3.0 → 2.0
s = open("proximity_app_cpu.py", encoding="utf-8").read()
rep = [
('ap.add_argument("--far", type=float, default=3.0, help="이탈 알람 거리(m)")',
 'ap.add_argument("--far", type=float, default=2.0, help="이탈 알람 거리(m), 3.0이면 교실에서 cam2 시야 밖이라 2.0으로")\n'
 'ap.add_argument("--leave-single", action="store_true", help="이탈 판단에는 한 카메라(single) 위치도 씀")'),
('        self.alarm = ProximityAlarm(args.near, args.far)\n',
 '        self.alarm = ProximityAlarm(args.near, args.far)\n'
 '        self.leave_alarm = ProximityAlarm(args.near, args.far)   # 이탈 전용 (single 포함)\n'),
('            if d is not None and tr.kind != "single":   # 한 카메라 추정은',
 '            if d is not None and args.leave_single:\n'
 '                for msg in tr.leave_alarm.update(now, d):\n'
 '                    if tr.kind == "single" and msg.startswith("알람: 이탈") and tr.alarm.zone != "far":\n'
 '                        tr.alarm.zone = "far"                      # stereo 쪽이 같은 이탈을 다시 내지 않게\n'
 '                        log(now, f"#{tr.id}", msg + " (single)")\n'
 '            if d is not None and tr.kind != "single":   # 한 카메라 추정은'),
]
for a, b in rep:
    assert s.count(a) == 1, a[:50]
    s = s.replace(a, b)
open("proximity_app_leave.py", "w", encoding="utf-8").write(s)
print("ok")
