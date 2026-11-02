p="prox_epfl.py"; s=open(p,encoding="utf-8").read()
old="""    for r, line in enumerate(rows[2:]):
        f = first + r * step
        if f > last:
            break
"""
new="""    for f, line in enumerate(rows[2:]):          # 프레임마다 한 줄, 정답은 step(25)프레임마다
        if f < first or f > last or (f - first) % step:
            continue
"""
assert old in s; open(p,"w",encoding="utf-8").write(s.replace(old,new)); print("ok")
import numpy as np
h=s.replace(old,new).split("def load_gt")[1].split("# ----------------")[0]
g={"np":np}; exec("def load_gt"+h, g); gt=g["load_gt"]("../media/prox_epfl/gt_lab_6p.txt")
print(len(gt), sum(len(v) for v in gt.values()), gt[500])
