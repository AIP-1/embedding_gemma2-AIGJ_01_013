import cv2, time
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
assert cap.isOpened(), "카메라를 열 수 없음"
for _ in range(10): cap.read()          # 노출 안정화
for i in range(1, 6):
    input(f"장면 {i} 준비되면 엔터: ")
    for _ in range(5): cap.read()       # 이전 프레임 비우기
    ok, f = cap.read(); cv2.imwrite(f"media/shot_{i}.jpg", f); print("저장", i, ok)
input("10초 동영상 녹화, 준비되면 엔터: ")
w = cv2.VideoWriter("media/clip.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 15, (1280, 720))
t = time.time()
while time.time() - t < 10:
    ok, f = cap.read()
    if ok: w.write(cv2.resize(f, (1280, 720)))
w.release(); cap.release(); print("동영상 저장 완료")
