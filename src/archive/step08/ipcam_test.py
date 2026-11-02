import cv2, getpass, os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
ip = input("카메라 IP: "); user = input("카메라 계정: "); pw = getpass.getpass("비밀번호: ")
for path in ["stream1", "stream2"]:
    cap = cv2.VideoCapture(f"rtsp://{user}:{pw}@{ip}:554/{path}", cv2.CAP_FFMPEG)
    ok, f = cap.read()
    print(path, "성공" if ok else "실패", f.shape if ok else "")
    if ok: cv2.imwrite(f"media/ipcam_{path}.jpg", f)
    cap.release()
