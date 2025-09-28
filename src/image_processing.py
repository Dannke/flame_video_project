import cv2
import numpy as np
from collections import deque

# При необходимости подставьте свои точки из projection.py
DEFAULT_SRC_PTS = None  # если не используем гомографию, оставьте None


def apply_perspective_transform(frame, src_pts, dst_size=(928, 576)):
    """Применение перспективного преобразования"""
    if src_pts is None:
        return frame
    dst_pts = np.array([[0, 0], [dst_size[0]-1, 0], [dst_size[0]-1,
                       dst_size[1]-1], [0, dst_size[1]-1]], dtype=np.float32)
    H, _ = cv2.findHomography(np.array(src_pts, dtype=np.float32), dst_pts)
    warped = cv2.warpPerspective(frame, H, (dst_size[0], dst_size[1]))
    return warped


def apply_preprocessing(frame, gamma=1.0):
    """CLAHE + опциональная гамма-коррекция"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    frame_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if abs(gamma - 1.0) > 1e-6:
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        frame_clahe = cv2.LUT(frame_clahe, table)
    return frame_clahe


def detect_flame_color_improved(frame):
    """Улучшенная детекция цвета пламени — предложенные пороги tuned для #e9e665 / #dfca57 / #e6d85d"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Жёлто-оранжевое (основной цвет пламени)
    # (OpenCV H: 0..179)
    H_low, H_high = 18, 35
    flame_mask_yellow = cv2.inRange(h, H_low, H_high)

    # "Красная" составляющая (через границу 0/179) — на случай очень горячих/внутренних точек
    flame_mask_red1 = cv2.inRange(h, 0, 8)
    flame_mask_red2 = cv2.inRange(h, 165, 179)
    flame_mask_red = cv2.bitwise_or(flame_mask_red1, flame_mask_red2)

    # Объединяем оттенки
    hue_mask = cv2.bitwise_or(flame_mask_yellow, flame_mask_red)

    # Требования к насыщенности и яркости (строгое)
    S_low = 110    # можно уменьшить до 90-100 если слишком много отбрасывается
    V_low = 200    # можно поднять до 210 для ещё большей строгости
    saturation_mask = cv2.inRange(s, S_low, 255)
    brightness_mask = cv2.inRange(v, V_low, 255)

    # Комбинируем все условия
    flame_mask = cv2.bitwise_and(hue_mask, saturation_mask)
    flame_mask = cv2.bitwise_and(flame_mask, brightness_mask)

    # Морфология для удаления мелкого шума
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    flame_mask = cv2.morphologyEx(
        flame_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    flame_mask = cv2.morphologyEx(
        flame_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return flame_mask


def detect_temporal_flickering(frame_buffer, min_frames=3,
                               flicker_threshold=25,
                               consistency_threshold=0.6):
    """
    Улучшенное обнаружение мерцания на основе нескольких кадров

    Args:
        frame_buffer: буфер кадров (deque или список)
        min_frames: минимальное количество кадров для анализа
        flicker_threshold: порог изменения между кадрами
        consistency_threshold: минимальная доля кадров с изменениями

    Returns:
        Маска мерцающих областей
    """
    if len(frame_buffer) < min_frames:
        return None

    # Конвертируем все кадры в grayscale
    gray_frames = []
    for frame in frame_buffer:
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        gray_frames.append(gray)

    h, w = gray_frames[0].shape
    flicker_accumulator = np.zeros((h, w), dtype=np.float32)

    # Анализируем изменения между последовательными кадрами
    for i in range(1, len(gray_frames)):
        # Вычисляем разницу
        diff = cv2.absdiff(gray_frames[i], gray_frames[i-1])

        # Применяем порог
        _, diff_thresh = cv2.threshold(
            diff, flicker_threshold, 1, cv2.THRESH_BINARY)

        # Накапливаем изменения
        flicker_accumulator += diff_thresh

    # Нормализуем по количеству сравнений
    flicker_accumulator /= (len(gray_frames) - 1)

    # Области, которые изменялись достаточно часто
    _, flicker_mask = cv2.threshold(flicker_accumulator, consistency_threshold,
                                    255, cv2.THRESH_BINARY)

    # Морфологическая обработка
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    flicker_mask = cv2.morphologyEx(flicker_mask.astype(np.uint8),
                                    cv2.MORPH_CLOSE, kernel)
    flicker_mask = cv2.morphologyEx(flicker_mask, cv2.MORPH_OPEN, kernel)

    # Фильтруем по размеру областей
    contours, _ = cv2.findContours(flicker_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(flicker_mask)
    min_area = 150  # Минимальная площадь мерцающей области

    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(filtered_mask, [contour], -1, 255, -1)

    return filtered_mask


def generate_flame_mask_improved(frame, frame_buffer=None,
                                 polygon_vertices=None,
                                 resize_to=(928, 576),
                                 use_roi=False,
                                 # Настраиваемые параметры
                                 brightness_thresh=220,  # Повышен
                                 saturation_thresh=120,  # Повышен
                                 color_weight=0.5,       # Вес цветовой маски
                                 flicker_weight=0.3,     # Снижен вес мерцания
                                 min_flicker_frames=5,   # Минимум кадров для мерцания
                                 use_color_filter=True):  # Использовать цветовой фильтр
    """
    Улучшенная генерация маски пламени с многокадровым анализом

    Args:
        frame: текущий кадр
        frame_buffer: буфер предыдущих кадров (deque или list)
        polygon_vertices: вершины ROI
        resize_to: размер для обработки
        use_roi: использовать ли ROI
        brightness_thresh: порог яркости (210-240 рекомендуется)
        saturation_thresh: порог насыщенности (100-140 рекомендуется)
        color_weight: вес цветовой детекции (0-1)
        flicker_weight: вес детекции мерцания (0-1)
        min_flicker_frames: минимальное количество кадров для анализа мерцания
        use_color_filter: использовать ли строгую цветовую фильтрацию

    Returns:
        (mask, flame_percent, confidence)
    """
    frame_resized = cv2.resize(frame, resize_to)
    img_hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    # 1. Базовая детекция по яркости и насыщенности (более строгая)
    _, mask_bright = cv2.threshold(
        v, brightness_thresh, 255, cv2.THRESH_BINARY)
    _, mask_saturation = cv2.threshold(
        s, saturation_thresh, 255, cv2.THRESH_BINARY)
    basic_mask = cv2.bitwise_and(mask_bright, mask_saturation)

    # 2. Цветовая детекция пламени (если включена)
    if use_color_filter:
        color_mask = detect_flame_color_improved(frame_resized)
        # Применяем цветовой фильтр как дополнительное условие
        basic_mask = cv2.bitwise_and(basic_mask, color_mask)

    # 3. Детекция мерцания (если есть буфер кадров)
    flicker_mask = np.zeros_like(basic_mask)
    if frame_buffer is not None and len(frame_buffer) >= min_flicker_frames:
        flicker_mask = detect_temporal_flickering(
            frame_buffer,
            min_frames=min_flicker_frames,
            flicker_threshold=25,
            consistency_threshold=0.5
        )
        if flicker_mask is not None:
            flicker_mask = cv2.resize(flicker_mask, resize_to)

    # 4. Комбинирование масок с весами
    combined = cv2.addWeighted(basic_mask, 1.0 - flicker_weight,
                               flicker_mask, flicker_weight, 0)

    # 5. Пороговая обработка и морфология
    _, combined = cv2.threshold(combined, 127, 255, cv2.THRESH_BINARY)

    # Более агрессивная морфологическая фильтрация
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Удаляем мелкий шум
    combined = cv2.morphologyEx(
        combined, cv2.MORPH_OPEN, kernel_small, iterations=2)
    # Заполняем дыры
    combined = cv2.morphologyEx(
        combined, cv2.MORPH_CLOSE, kernel_large, iterations=1)

    # 6. Фильтрация по размеру областей
    contours, _ = cv2.findContours(combined.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(combined)
    min_contour_area = 200  # Минимальная площадь области пламени

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_contour_area:
            # Дополнительная проверка формы (опционально)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                # Пламя обычно имеет неправильную форму (circularity < 0.8)
                if circularity < 0.9:
                    cv2.drawContours(filtered_mask, [contour], -1, 255, -1)

    # 7. Применение ROI (если задан)
    if use_roi and polygon_vertices is not None and len(polygon_vertices) >= 3:
        roi_mask = np.zeros_like(filtered_mask)
        cv2.fillPoly(
            roi_mask, [np.array(polygon_vertices, dtype=np.int32)], 255)
        filtered_mask = cv2.bitwise_and(filtered_mask, roi_mask)

    # 8. Расчет статистик
    flame_pixels = cv2.countNonZero(filtered_mask)
    total_pixels = filtered_mask.shape[0] * filtered_mask.shape[1]
    flame_percent = (flame_pixels / total_pixels) * \
        100 if total_pixels > 0 else 0

    # Улучшенный расчет уверенности
    if flame_pixels > 0:
        # Анализируем характеристики обнаруженных областей
        avg_brightness = np.mean(v[filtered_mask > 0])
        avg_saturation = np.mean(s[filtered_mask > 0])
        avg_hue = np.mean(h[filtered_mask > 0])

        # Проверяем, попадает ли средний оттенок в диапазон пламени
        hue_score = 1.0 if (5 <= avg_hue <= 35) or (avg_hue >= 165) else 0.5

        # Комбинированная уверенность
        brightness_score = min(1.0, (avg_brightness - 200) / 55)  # 200-255
        saturation_score = min(1.0, (avg_saturation - 100) / 155)  # 100-255

        confidence = (brightness_score * 30 +
                      saturation_score * 30 +
                      hue_score * 40)  # Больший вес на цвет
    else:
        confidence = 0

    return filtered_mask.astype(np.uint8), flame_percent, confidence


# Оставляем старую функцию для совместимости, но перенаправляем на новую
def generate_flame_mask(frame, polygon_vertices=None, resize_to=(928, 576),
                        prev_frame=None, use_roi=False,
                        brightness_thresh=220, saturation_thresh=120,
                        flicker_weight=0.3):
    """
    Обертка для совместимости с существующим кодом
    """
    # Создаем простой буфер из двух кадров если есть prev_frame
    frame_buffer = None
    if prev_frame is not None:
        frame_buffer = [prev_frame, frame]

    return generate_flame_mask_improved(
        frame=frame,
        frame_buffer=frame_buffer,
        polygon_vertices=polygon_vertices,
        resize_to=resize_to,
        use_roi=use_roi,
        brightness_thresh=brightness_thresh,
        saturation_thresh=saturation_thresh,
        flicker_weight=flicker_weight,
        min_flicker_frames=2,  # Минимум для работы со старым API
        use_color_filter=True
    )
