import time
from eg2 import load
model = load()
from test_text_data import docs, queries
for label, dp, qp in [("prompt", "Document", "SearchQuery"), ("no-prompt", None, None)]:
    d = model.encode(docs, prompt_name=dp, normalize_embeddings=True)
    q = model.encode(queries, prompt_name=qp, normalize_embeddings=True)
    for _ in range(3):
        t = time.perf_counter(); model.encode(queries, prompt_name=qp, normalize_embeddings=True); ms = (time.perf_counter()-t)*1000
    sim = q @ d.T
    print(f"[{label}] dim=768, warm latency {ms:.0f} ms")
    for qi, row in zip(queries, sim):
        print("  ", qi[:14], " ".join(f"{x:.3f}" for x in row))
