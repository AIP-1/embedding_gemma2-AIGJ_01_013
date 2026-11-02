import os, cv2
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

def open_ipcam(hd=False):
    base = open(os.path.expanduser("~/.ipcam")).read().strip()
    cap = cv2.VideoCapture(f"{base}/{'stream1' if hd else 'stream2'}", cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap
