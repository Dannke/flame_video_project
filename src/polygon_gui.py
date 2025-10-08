# polygon_gui.py
import cv2
import json
import os
from pathlib import Path

INSTRUCTIONS = (
    "ЛКМ: добавить точку (до 6). ПКМ / u: удалить последнюю. R: сброс. S: сохранить. K: оставить дефолт. ESC: отмена."
)


def _mouse_callback(event, x, y, flags, param):
    pts = param["points"]
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pts) < 6:
            pts.append((int(x), int(y)))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if pts:
            pts.pop()


def prompt_polygon_on_image(img, default_polygon=None, video_name="video", save_dir=None, window_name="Set polygon"):
    """
    Показывает окно с кадром img и позволяет выставить 6 точек.
    Возвращает polygon (list of 6 (x,y)) или default_polygon (если отмена/keep).
    Если save_dir указан — сохраняет JSON: <save_dir>/<video_name>_polygon.json
    """
    if img is None:
        raise ValueError("img is None")

    pts = []
    clone = img.copy()
    h, w = img.shape[:2]
    display = clone.copy()

    params = {"points": pts}

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback(window_name, _mouse_callback, params)

    def draw():
        nonlocal display
        display = clone.copy()
        cv2.rectangle(display, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.putText(display, INSTRUCTIONS, (8, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        for i, (x, y) in enumerate(pts):
            cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(display, str(i + 1), (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                cv2.line(display, pts[i], pts[i + 1], (0, 200, 0), 2)
        if len(pts) == 6:
            cv2.line(display, pts[-1], pts[0], (0, 200, 0), 2)

    draw()

    while True:
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            cv2.destroyWindow(window_name)
            return default_polygon
        elif key == ord('u'):
            if pts:
                pts.pop()
                draw()
        elif key == ord('r'):
            pts.clear()
            draw()
        elif key == ord('k'):
            cv2.destroyWindow(window_name)
            return default_polygon
        elif key == ord('s'):
            if len(pts) == 6:
                polygon = [(int(x), int(y)) for (x, y) in pts]
                if save_dir:
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    fname = os.path.join(
                        save_dir, f"{video_name}_polygon.json")
                    with open(fname, "w", encoding="utf-8") as f:
                        json.dump({"polygon": polygon}, f, indent=2)
                cv2.destroyWindow(window_name)
                return polygon
            else:
                print(f"Требуется 6 точек полигона. Текущих: {len(pts)}")
        else:
            draw()

    cv2.destroyWindow(window_name)
    return default_polygon


def load_saved_polygon(save_dir, video_name):
    fname = os.path.join(save_dir, f"{video_name}_polygon.json")
    if os.path.exists(fname):
        try:
            with open(fname, "r", encoding="utf-8") as f:
                data = json.load(f)
                poly = data.get("polygon")
                if isinstance(poly, list) and len(poly) == 6:
                    return [(int(x), int(y)) for (x, y) in poly]
        except Exception:
            return None
    return None
