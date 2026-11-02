import time
from eg2 import load

model = load(vision=True)
t = time.perf_counter()
vid_emb = model.encode({"video": "media/clip.mp4"}, normalize_embeddings=True)
print(f"동영상 1개: {(time.perf_counter()-t)*1000:.0f} ms, shape {vid_emb.shape}")

queries = ["모니터 앞에 사람들이 앉아 있는 컴퓨터 실습실", "바다와 모래사장", "고양이 사진"]
q = model.encode(queries, prompt_name="SearchQuery", normalize_embeddings=True, batch_size=2)
for qi, s in zip(queries, q @ vid_emb):
    print(f"{qi}: {s:.3f}")
