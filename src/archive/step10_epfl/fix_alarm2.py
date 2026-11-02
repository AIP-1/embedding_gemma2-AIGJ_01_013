# 정답 프레임이 홀수(25, 75…)면 stride 2로 건너뛴 앞 프레임 상태를 쓴다
p = "prox_epfl.py"; s = open(p, encoding="utf-8").read()
a = """        if f in sys_near:
            st[(d <= args.near, sys_near[f])] += 1
"""
b = """        fs_ = f if f in sys_near else f - 1
        if fs_ in sys_near:
            st[(d <= args.near, sys_near[fs_])] += 1
"""
assert s.count(a) == 1; open(p, "w", encoding="utf-8").write(s.replace(a, b)); print("patched2")
