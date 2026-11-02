"""9-6 평가: 학습에 안 쓴 영상의 특징 CSV(data/action_test/<라벨>.csv)로 GRU 모델들을 같은 기준으로 비교한다.
윈도 나누기는 train_pose.py와 같다(추적 ID별, 0.5초 넘게 끊기면 다른 시퀀스, window 프레임, stride 5).
사용: python eval_gru.py --data data/action_test models/action.pt models/action_v2.pt
"""
import argparse, collections, csv, glob, os
import numpy as np
from pose_model import load_model, predict

ap = argparse.ArgumentParser()
ap.add_argument("models", nargs="+")
ap.add_argument("--data", default="data/action_test")
ap.add_argument("--stride", type=int, default=5)
args = ap.parse_args()

def load_sequences(path, window):
    by_tid = collections.defaultdict(list)
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) != 5 + 51:
                continue
            feat = np.array([float(x) for x in row[5:]] + [float(row[3]), float(row[4])], dtype=np.float32)
            by_tid[row[1]].append((float(row[0]), feat))
    seqs = []
    for rows in by_tid.values():
        rows.sort(key=lambda r: r[0])
        cur = [rows[0][1]]
        for (t0, _), (t1, f) in zip(rows, rows[1:]):
            if t1 - t0 > 0.5:
                seqs.append(cur); cur = []
            cur.append(f)
        seqs.append(cur)
    return [np.stack(s) for s in seqs if len(s) >= window]

files = sorted(glob.glob(os.path.join(args.data, "*.csv")))
for mp in args.models:
    ck = load_model(mp)
    W = ck["window"]
    print(f"\n### {mp} (라벨 {ck['labels']}, window {W})")
    print("| 정답 | 윈도 | 맞힘 | 정확도 | 평균 최고 확률 | 예측 분포 | p≥0.7 판정(맞음) | p≥0.5 판정(맞음) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    tot = ok = 0
    for p in files:
        lab = os.path.splitext(os.path.basename(p))[0]
        preds = []
        for seq in load_sequences(p, W):
            for i in range(0, len(seq) - W + 1, args.stride):
                x = seq[i:i + W].copy()
                if ck.get("upper"):                      # train_pose_v3.py --upper로 학습한 모델: 하체 특징을 지움
                    x[:, list(range(22, 34)) + list(range(45, 51))] = 0
                preds.append(predict(ck, x))
        n = len(preds)
        hit = sum(l == lab for l, _ in preds)
        dist = collections.Counter(l for l, _ in preds)
        c7 = [(l, q) for l, q in preds if q >= 0.7]
        c5 = [(l, q) for l, q in preds if q >= 0.5]
        print(f"| {lab} | {n} | {hit} | {hit / max(n, 1):.0%} | {np.mean([q for _, q in preds]) if preds else 0:.2f} | "
              f"{dict(dist.most_common())} | {len(c7)}({sum(l == lab for l, _ in c7)}) | {len(c5)}({sum(l == lab for l, _ in c5)}) |")
        tot += n; ok += hit
    print(f"| 전체 | {tot} | {ok} | {ok / max(tot, 1):.0%} | | | | |")
