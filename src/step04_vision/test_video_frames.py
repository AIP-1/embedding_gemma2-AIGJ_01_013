import time, cv2, numpy as np
from PIL import Image
from eg2 import load

cap = cv2.VideoCapture("media/clip.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 15
frames, i = [], 0
while True:
    ok, f = cap.read()
    if not ok: break
    if i % int(round(fps)) == 0:
        frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    i += 1
print(f"프레임 {len(frames)}장 추출 (1초당 1장)")

model = load(vision=True)
t = time.perf_counter()
emb = model.encode([{"image": im} for im in frames], normalize_embeddings=True)
vid = emb.mean(0); vid /= np.linalg.norm(vid)
print(f"동영상 임베딩: {(time.perf_counter()-t)*1000:.0f} ms")

queries = ["모니터 앞에 사람들이 앉아 있는 컴퓨터 실습실", "바다와 모래사장", "고양이 사진"]
q = model.encode(queries, prompt_name="SearchQuery", normalize_embeddings=True)
for qi, s in zip(queries, q @ vid):
    print(f"{qi}: {s:.3f}")
