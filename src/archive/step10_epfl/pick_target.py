# 정답 위치로 타깃 후보마다 "0.8m 안에 사람 있음" 초 비율과 근접 진입 횟수
import numpy as np
src = open("prox_epfl.py", encoding="utf-8").read()
h = src.split("def load_gt")[1].split("# ----------------")[0]
g = {"np": np}; exec("def load_gt" + h, g)
gt = {f: v for f, v in g["load_gt"]("../media/prox_epfl/gt_lab_6p.txt").items() if v}
print("labelled sec with people", len(gt))
for x in np.arange(0.4, 3.4, 0.4):
    row = []
    for y in np.arange(0.4, 3.4, 0.4):
        t = np.array([x, y]); near = [min(np.linalg.norm(p - t) for p in v.values()) <= 0.8 for v in gt.values()]
        ent = sum(1 for a, b in zip(near, near[1:]) if b and not a)
        row.append(f"{np.mean(near):.2f}/{ent:2d}")
    print(f"x={x:.1f}", " ".join(row))
