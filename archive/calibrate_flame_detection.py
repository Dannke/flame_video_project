#!/usr/bin/env python3
"""
Интерактивная калибровка параметров детекции пламени
"""

import cv2
import numpy as np
import os
import sys
import glob

# Добавляем src в путь
sys.path.append('src')

# Параметры по умолчанию
DEFAULT_PARAMS = {
    'hue_low1': 0,      # Нижний предел красно-оранжевого
    'hue_high1': 25,    # Верхний предел красно-оранжевого
    'hue_low2': 160,    # Нижний предел красного (второй диапазон)
    'hue_high2': 180,   # Верхний предел красного
    'sat_min': 120,     # Минимальная насыщенность
    'val_min': 180,     # Минимальная яркость
    'brightness_percentile': 85,  # Процентиль для порога яркости
    'brightness_min': 200,        # Минимальное значение порога яркости
    'temp_threshold': 80,         # Порог для температурного индекса
    'min_area': 500,             # Минимальная площадь контура
    'max_area_percent': 30,      # Максимальная площадь в % от общей
    'max_circularity': 70,       # Максимальная округлость (в процентах)
    'min_aspect': 30,            # Минимальное соотношение сторон (в процентах)
    'max_aspect': 300,           # Максимальное соотношение сторон
    'min_fill': 30               # Минимальная заполненность прямоугольника
}


class FlameCalibrator:
    def __init__(self, frame_path, roi_path=None):
        self.frame_path = frame_path
        self.roi_path = roi_path
        self.params = DEFAULT_PARAMS.copy()

        # Загружаем кадр
        self.frame = cv2.imread(frame_path)
        if self.frame is None:
            raise ValueError(f"Не удалось загрузить кадр: {frame_path}")

        # Изменяем размер для обработки
        self.target_size = (928, 576)
        self.frame_resized = cv2.resize(self.frame, self.target_size)

        # Загружаем ROI если есть
        self.polygon = None
        self.roi_mask = None
        if roi_path and os.path.exists(roi_path):
            roi_points = np.load(roi_path)
            self.polygon = roi_points.tolist()

            # Масштабируем ROI под новый размер
            scale_x = self.target_size[0] / self.frame.shape[1]
            scale_y = self.target_size[1] / self.frame.shape[0]

            scaled_vertices = []
            for x, y in self.polygon:
                scaled_vertices.append((int(x * scale_x), int(y * scale_y)))

            # Создаем маску ROI
            self.roi_mask = np.zeros(self.target_size[::-1], dtype=np.uint8)
            pts = np.array(scaled_vertices, dtype=np.int32)
            cv2.fillPoly(self.roi_mask, [pts], 255)

        # Создаем окна
        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        cv2.namedWindow('HSV Mask', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Brightness Mask', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Temperature Mask', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Final Result', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)

        # Создаем пустое изображение для контролов
        self.controls_img = np.ones((800, 500, 3), dtype=np.uint8) * 50

        # Создаем трекбары
        self.create_trackbars()

    def create_trackbars(self):
        """Создание трекбаров для управления параметрами"""
        cv2.createTrackbar('Hue Low 1', 'Controls',
                           self.params['hue_low1'], 180, self.on_trackbar)
        cv2.createTrackbar('Hue High 1', 'Controls',
                           self.params['hue_high1'], 180, self.on_trackbar)
        cv2.createTrackbar('Hue Low 2', 'Controls',
                           self.params['hue_low2'], 180, self.on_trackbar)
        cv2.createTrackbar('Hue High 2', 'Controls',
                           self.params['hue_high2'], 180, self.on_trackbar)
        cv2.createTrackbar('Sat Min', 'Controls',
                           self.params['sat_min'], 255, self.on_trackbar)
        cv2.createTrackbar('Val Min', 'Controls',
                           self.params['val_min'], 255, self.on_trackbar)
        cv2.createTrackbar('Brightness %ile', 'Controls',
                           self.params['brightness_percentile'], 100, self.on_trackbar)
        cv2.createTrackbar('Brightness Min', 'Controls',
                           self.params['brightness_min'], 255, self.on_trackbar)
        cv2.createTrackbar('Temp Threshold', 'Controls',
                           self.params['temp_threshold'], 255, self.on_trackbar)
        cv2.createTrackbar('Min Area', 'Controls',
                           self.params['min_area'] // 10, 500, self.on_trackbar)
        cv2.createTrackbar('Max Area %', 'Controls',
                           self.params['max_area_percent'], 100, self.on_trackbar)
        cv2.createTrackbar('Max Circular %', 'Controls',
                           self.params['max_circularity'], 100, self.on_trackbar)

    def on_trackbar(self, val):
        """Обработчик изменения трекбаров"""
        # Обновляем параметры
        self.params['hue_low1'] = cv2.getTrackbarPos('Hue Low 1', 'Controls')
        self.params['hue_high1'] = cv2.getTrackbarPos('Hue High 1', 'Controls')
        self.params['hue_low2'] = cv2.getTrackbarPos('Hue Low 2', 'Controls')
        self.params['hue_high2'] = cv2.getTrackbarPos('Hue High 2', 'Controls')
        self.params['sat_min'] = cv2.getTrackbarPos('Sat Min', 'Controls')
        self.params['val_min'] = cv2.getTrackbarPos('Val Min', 'Controls')
        self.params['brightness_percentile'] = cv2.getTrackbarPos(
            'Brightness %ile', 'Controls')
        self.params['brightness_min'] = cv2.getTrackbarPos(
            'Brightness Min', 'Controls')
        self.params['temp_threshold'] = cv2.getTrackbarPos(
            'Temp Threshold', 'Controls')
        self.params['min_area'] = cv2.getTrackbarPos(
            'Min Area', 'Controls') * 10
        self.params['max_area_percent'] = cv2.getTrackbarPos(
            'Max Area %', 'Controls')
        self.params['max_circularity'] = cv2.getTrackbarPos(
            'Max Circular %', 'Controls')

        # Применяем детекцию
        self.update_detection()

    def enhance_flame_regions_custom(self, img):
        """Улучшенная детекция областей пламени с настраиваемыми параметрами"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Применяем настраиваемые пороги
        flame_hue_mask1 = cv2.inRange(
            h, self.params['hue_low1'], self.params['hue_high1'])
        flame_hue_mask2 = cv2.inRange(
            h, self.params['hue_low2'], self.params['hue_high2'])
        flame_hue_mask = cv2.bitwise_or(flame_hue_mask1, flame_hue_mask2)

        saturation_mask = cv2.inRange(s, self.params['sat_min'], 255)
        brightness_mask = cv2.inRange(v, self.params['val_min'], 255)

        # Все условия должны выполняться одновременно
        color_mask = cv2.bitwise_and(flame_hue_mask, saturation_mask)
        color_mask = cv2.bitwise_and(color_mask, brightness_mask)

        return color_mask

    def detect_bright_regions_custom(self, img_gray):
        """Детекция ярких областей с настраиваемыми параметрами"""
        blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

        # Используем настраиваемый процентиль
        threshold_value = np.percentile(
            blurred, self.params['brightness_percentile'])
        threshold_value = max(threshold_value, self.params['brightness_min'])

        _, bright_thresh = cv2.threshold(
            blurred, threshold_value, 255, cv2.THRESH_BINARY)

        return bright_thresh

    def create_temperature_mask_custom(self, img):
        """Создание температурной маски с настраиваемыми параметрами"""
        b, g, r = cv2.split(img)

        # Нормализуем каналы
        r_norm = r.astype(np.float32) / 255.0
        g_norm = g.astype(np.float32) / 255.0
        b_norm = b.astype(np.float32) / 255.0

        # Температурный индекс
        temp_index = (r_norm - b_norm - 0.3 * g_norm) / \
            (r_norm + b_norm + g_norm + 0.001)
        temp_index = np.clip(temp_index * 255, 0, 255).astype(np.uint8)

        # Применяем настраиваемый порог
        _, temp_mask = cv2.threshold(
            temp_index, self.params['temp_threshold'], 255, cv2.THRESH_BINARY)

        # Морфологическая обработка
        kernel = np.ones((3, 3), np.uint8)
        temp_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_OPEN, kernel)

        return temp_mask

    def filter_contours_custom(self, mask):
        """Фильтрация контуров с настраиваемыми параметрами"""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(mask)

        total_area = mask.shape[0] * mask.shape[1]
        max_allowed_area = total_area * self.params['max_area_percent'] / 100.0

        for contour in contours:
            area = cv2.contourArea(contour)

            # Проверяем площадь
            if area < self.params['min_area'] or area > max_allowed_area:
                continue

            # Анализ формы
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            # Округлость
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            max_circularity = self.params['max_circularity'] / 100.0

            if circularity > max_circularity:
                continue

            # Соотношение сторон
            x, y, w, h = cv2.boundingRect(contour)
            if h == 0:
                continue

            aspect_ratio = float(w) / h
            min_aspect = self.params['min_aspect'] / 100.0
            max_aspect = self.params['max_aspect'] / 100.0

            if aspect_ratio < min_aspect or aspect_ratio > max_aspect:
                continue

            # Заполненность прямоугольника
            rect_area = w * h
            if rect_area > 0:
                fill_ratio = area / rect_area
                min_fill = self.params['min_fill'] / 100.0
                if fill_ratio < min_fill:
                    continue

            # Рисуем контур
            cv2.drawContours(filtered_mask, [contour], -1, 255, -1)

        return filtered_mask

    def update_detection(self):
        """Обновление результата детекции"""
        img_gray = cv2.cvtColor(self.frame_resized, cv2.COLOR_BGR2GRAY)

        # Получаем маски
        hsv_mask = self.enhance_flame_regions_custom(self.frame_resized)
        brightness_mask = self.detect_bright_regions_custom(img_gray)
        temp_mask = self.create_temperature_mask_custom(self.frame_resized)

        # Комбинируем маски
        combined_mask = cv2.bitwise_and(hsv_mask, brightness_mask)

        # Добавляем температурную маску только если есть пересечения
        temp_and_combined = cv2.bitwise_and(temp_mask, combined_mask)
        if cv2.countNonZero(temp_and_combined) > cv2.countNonZero(combined_mask) * 0.1:
            combined_mask = cv2.bitwise_or(combined_mask, temp_and_combined)

        # Применяем морфологию
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined_mask = cv2.morphologyEx(
            combined_mask, cv2.MORPH_CLOSE, kernel_close)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(
            combined_mask, cv2.MORPH_OPEN, kernel_open)

        # Фильтруем контуры
        final_mask = self.filter_contours_custom(combined_mask)

        # Применяем ROI если есть
        if self.roi_mask is not None:
            final_mask = cv2.bitwise_and(final_mask, self.roi_mask)

        # Создаем финальную визуализацию
        mask_colored = np.zeros_like(self.frame_resized)
        mask_colored[:, :, 2] = final_mask
        mask_colored[:, :, 1] = final_mask // 2

        overlay = cv2.addWeighted(
            self.frame_resized, 0.7, mask_colored, 0.3, 0)

        # Рисуем ROI если есть
        if self.polygon:
            pts = np.array([(int(x * self.target_size[0] / self.frame.shape[1]),
                           int(y * self.target_size[1] / self.frame.shape[0]))
                            for x, y in self.polygon], dtype=np.int32)
            cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)

        # Добавляем статистику
        total_pixels = final_mask.shape[0] * final_mask.shape[1]
        flame_pixels = cv2.countNonZero(final_mask)
        flame_percent = (flame_pixels / total_pixels) * 100

        # Информация о контурах
        contours, _ = cv2.findContours(
            final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = len(contours)

        # Добавляем текст с информацией
        info_text = [
            f"Flame: {flame_percent:.1f}%",
            f"Pixels: {flame_pixels}",
            f"Regions: {num_contours}",
            f"Frame: {os.path.basename(self.frame_path)}"
        ]

        for i, text in enumerate(info_text):
            y_pos = 25 + i * 25
            cv2.rectangle(overlay, (5, y_pos - 18),
                          (300, y_pos + 7), (0, 0, 0), -1)
            cv2.putText(overlay, text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Показываем результаты
        cv2.imshow('Original', self.frame_resized)
        cv2.imshow('HSV Mask', hsv_mask)
        cv2.imshow('Brightness Mask', brightness_mask)
        cv2.imshow('Temperature Mask', temp_mask)
        cv2.imshow('Final Result', overlay)

        # Обновляем панель управления
        self.update_controls_display()

        return final_mask, flame_percent

    def update_controls_display(self):
        """Обновление отображения панели управления"""
        img = np.ones((800, 500, 3), dtype=np.uint8) * 50

        # Заголовок
        cv2.putText(img, "FLAME DETECTION CALIBRATION", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Текущие параметры
        params_text = [
            "COLOR PARAMETERS:",
            f"  Hue Range 1: {self.params['hue_low1']}-{self.params['hue_high1']}",
            f"  Hue Range 2: {self.params['hue_low2']}-{self.params['hue_high2']}",
            f"  Saturation Min: {self.params['sat_min']}",
            f"  Value Min: {self.params['val_min']}",
            "",
            "BRIGHTNESS PARAMETERS:",
            f"  Percentile: {self.params['brightness_percentile']}%",
            f"  Min Threshold: {self.params['brightness_min']}",
            "",
            "TEMPERATURE PARAMETERS:",
            f"  Threshold: {self.params['temp_threshold']}",
            "",
            "SHAPE FILTERING:",
            f"  Min Area: {self.params['min_area']}",
            f"  Max Area: {self.params['max_area_percent']}%",
            f"  Max Circularity: {self.params['max_circularity']}%",
            "",
            "CONTROLS:",
            "  S - Save parameters",
            "  L - Load parameters",
            "  R - Reset to defaults",
            "  ESC - Exit"
        ]

        y_offset = 60
        for text in params_text:
            if text.startswith("  "):
                color = (200, 200, 200)
                font_scale = 0.5
            elif text == "":
                y_offset += 10
                continue
            elif text.endswith(":"):
                color = (100, 255, 100)
                font_scale = 0.6
            else:
                color = (255, 255, 255)
                font_scale = 0.5

            cv2.putText(img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, 1)
            y_offset += 25

        cv2.imshow('Controls', img)

    def save_parameters(self, filename="flame_params.json"):
        """Сохранение текущих параметров"""
        import json
        with open(filename, 'w') as f:
            json.dump(self.params, f, indent=2)
        print(f"Параметры сохранены в {filename}")

    def load_parameters(self, filename="flame_params.json"):
        """Загрузка параметров"""
        import json
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.params = json.load(f)
            # Обновляем трекбары
            cv2.setTrackbarPos('Hue Low 1', 'Controls',
                               self.params['hue_low1'])
            cv2.setTrackbarPos('Hue High 1', 'Controls',
                               self.params['hue_high1'])
            cv2.setTrackbarPos('Hue Low 2', 'Controls',
                               self.params['hue_low2'])
            cv2.setTrackbarPos('Hue High 2', 'Controls',
                               self.params['hue_high2'])
            cv2.setTrackbarPos('Sat Min', 'Controls', self.params['sat_min'])
            cv2.setTrackbarPos('Val Min', 'Controls', self.params['val_min'])
            cv2.setTrackbarPos('Brightness %ile', 'Controls',
                               self.params['brightness_percentile'])
            cv2.setTrackbarPos('Brightness Min', 'Controls',
                               self.params['brightness_min'])
            cv2.setTrackbarPos('Temp Threshold', 'Controls',
                               self.params['temp_threshold'])
            cv2.setTrackbarPos('Min Area', 'Controls',
                               self.params['min_area'] // 10)
            cv2.setTrackbarPos('Max Area %', 'Controls',
                               self.params['max_area_percent'])
            cv2.setTrackbarPos('Max Circular %', 'Controls',
                               self.params['max_circularity'])
            self.update_detection()
            print(f"Параметры загружены из {filename}")
        else:
            print(f"Файл {filename} не найден")

    def reset_to_defaults(self):
        """Сброс к параметрам по умолчанию"""
        self.params = DEFAULT_PARAMS.copy()
        # Обновляем трекбары
        self.load_parameters()  # Используем load для обновления трекбаров
        self.update_detection()
        print("Параметры сброшены к значениям по умолчанию")

    def run(self):
        """Запуск калибровщика"""
        print("🔥 Калибровщик детекции пламени")
        print("Используйте трекбары для настройки параметров")
        print(
            "Нажмите 'S' для сохранения, 'L' для загрузки, 'R' для сброса, ESC для выхода")

        # Первоначальное обновление
        self.update_detection()

        while True:
            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # ESC
                break
            elif key == ord('s') or key == ord('S'):
                self.save_parameters()
            elif key == ord('l') or key == ord('L'):
                self.load_parameters()
            elif key == ord('r') or key == ord('R'):
                self.reset_to_defaults()

        cv2.destroyAllWindows()
        return self.params


def calibrate_on_multiple_frames(frames_dir, roi_path=None, num_frames=5):
    """Калибровка на нескольких кадрах"""
    frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if len(frames) == 0:
        print("Кадры не найдены")
        return

    # Выбираем кадры равномерно
    selected_frames = [frames[i * len(frames) // num_frames]
                       for i in range(num_frames)]

    print(f"Калибровка на {len(selected_frames)} кадрах:")
    for frame in selected_frames:
        print(f"  - {os.path.basename(frame)}")

    # Калибруем на каждом кадре
    all_params = []
    for i, frame_path in enumerate(selected_frames):
        print(f"\n--- Калибровка кадра {i+1}/{len(selected_frames)} ---")
        calibrator = FlameCalibrator(frame_path, roi_path)
        params = calibrator.run()
        all_params.append(params)

    # Усредняем параметры
    if all_params:
        avg_params = {}
        for key in DEFAULT_PARAMS.keys():
            avg_params[key] = int(np.mean([p[key] for p in all_params]))

        # Сохраняем усредненные параметры
        import json
        with open("flame_params_averaged.json", 'w') as f:
            json.dump(avg_params, f, indent=2)

        print("\n✅ Усредненные параметры сохранены в flame_params_averaged.json")
        print("Параметры:")
        for key, value in avg_params.items():
            print(f"  {key}: {value}")

    return avg_params if all_params else None


def main():
    """Основная функция"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Калибровка параметров детекции пламени')
    parser.add_argument('--frame', help='Путь к кадру для калибровки')
    parser.add_argument(
        '--frames_dir', help='Папка с кадрами для множественной калибровки')
    parser.add_argument('--roi', help='Файл с ROI координатами')
    parser.add_argument('--num_frames', type=int, default=5,
                        help='Количество кадров для множественной калибровки')

    args = parser.parse_args()

    if args.frames_dir:
        # Множественная калибровка
        calibrate_on_multiple_frames(
            args.frames_dir, args.roi, args.num_frames)
    elif args.frame:
        # Одиночная калибровка
        calibrator = FlameCalibrator(args.frame, args.roi)
        params = calibrator.run()
        print("Финальные параметры:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    else:
        # Автоматический поиск данных
        if os.path.exists("data/frames/1video"):
            frames = glob.glob("data/frames/1video/*.jpg")
            if frames:
                roi_path = "data/roi_1video.npy" if os.path.exists(
                    "data/roi_1video.npy") else None
                calibrator = FlameCalibrator(frames[0], roi_path)
                calibrator.run()
            else:
                print("Кадры не найдены в data/frames/1video/")
        else:
            print(
                "Использование: python calibrate_flame_detection.py --frame путь/к/кадру.jpg")
            print(
                "         или: python calibrate_flame_detection.py --frames_dir путь/к/папке/")


if __name__ == "__main__":
    main()
