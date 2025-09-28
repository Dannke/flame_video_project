import cv2, os

def extract_frames(video_path, out_dir, fps_out=5):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / fps_out)))
    idx = 0; saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            cv2.imwrite(os.path.join(out_dir, f"frame_{saved:06d}.jpg"), frame)
            saved += 1
        idx += 1
    cap.release()
    print(f"Saved {saved} frames to {out_dir}")

if __name__ == "__main__":
    import sys
    extract_frames(sys.argv[1], sys.argv[2], fps_out=int(sys.argv[3]) if len(sys.argv)>3 else 5)
