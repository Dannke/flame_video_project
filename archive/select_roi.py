import cv2, numpy as np, sys
fn = sys.argv[1] if len(sys.argv)>1 else "data/videos/1video.avi"
cap = cv2.VideoCapture(fn)
ret, frame = cap.read()
cap.release()
if not ret:
    print("Не удалось прочитать первый кадр")
    exit(1)
pts = []
def click(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        pts.append((x,y))
        cv2.circle(frame,(x,y),4,(0,255,0),-1)
        cv2.imshow("frame", frame)
cv2.imshow("frame", frame)
cv2.setMouseCallback("frame", click)
print("Кликайте точки полигона (Esc - закончить).")
while True:
    k = cv2.waitKey(0)
    if k==27:
        break
cv2.destroyAllWindows()
if len(pts) < 3:
    print("Мало точек")
else:
    np.save("data/roi_1video.npy", np.array(pts))
    print("ROI сохранён в data/roi_1video.npy")
