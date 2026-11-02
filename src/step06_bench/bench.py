import sys, time, glob, re, subprocess, threading
import cv2, torch
from PIL import Image
from eg2 import load

cfg = sys.argv[1]
def used_mb():
    m = {l.split(':')[0]: int(l.split()[1]) for l in open('/proc/meminfo')}
    return (m['MemTotal'] - m['MemAvailable']) // 1024

gpu = []
def watch():
    p = subprocess.Popen(['tegrastats', '--interval', '200'], stdout=subprocess.PIPE, text=True)
    for line in p.stdout:
        g = re.search(r'GR3D_FREQ (\d+)%', line)
        if g: gpu.append(int(g.group(1)))
threading.Thread(target=watch, daemon=True).start()

def video_frames():
    cap = cv2.VideoCapture('media/clip.mp4'); step = int(round(cap.get(cv2.CAP_PROP_FPS) or 15))
    out, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if i % step == 0: out.append({'image': Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))})
        i += 1
    return out

texts = ["수업 중에 졸리고 머리가 맑지 않아요", "여름에 냉방 몇 도로 맞춰야 하나요?"]
imgs = [{'image': p} for p in sorted(glob.glob('media/shot_*.jpg'))]
wavs = [{'audio': p} for p in sorted(glob.glob('media/*.wav')) if '/raw_' not in p]
groups = {'text': [texts], 'vision': [imgs], 'video': [video_frames()],
          'audio': [wavs], 'full': [texts, imgs, wavs]}[cfg]
size = {'text': '270M', 'vision': '440M', 'video': '440M', 'audio': '570M', 'full': '740M'}[cfg]

before = used_mb()
model = load(vision=cfg in ('vision', 'video', 'full'), audio=cfg in ('audio', 'full'))
loaded = used_mb()

def run():
    for g in groups:
        kw = {'prompt_name': 'SearchQuery'} if g is texts else {}
        model.encode(g, normalize_embeddings=True, batch_size=4, **kw)
    torch.cuda.synchronize()

run()                                   # 첫 실행은 준비 시간이라 버림
g0 = len(gpu); t = time.perf_counter()
run()
ms = (time.perf_counter() - t) * 1000
peak = used_mb(); time.sleep(0.6)
n = sum(len(g) for g in groups)
row = f"| {cfg} | {size} | {loaded} (+{loaded - before}) | {peak} | {n}개 | {ms:.0f} | {max(gpu[g0:] or [0])} |"
print(row); open('bench_result.md', 'a').write(row + '\n')
