"""
Улучшенный анализатор шихты с адаптивной детекцией пламени и зональным анализом.

Основные возможности:
- Адаптивная сегментация с учётом наличия пламени
- Зональный анализ (ближняя/дальняя зоны для алгоритма)
- Статистика по 3 зонам печи (0-30%, 30-70%, 70-100%)
- Поддержка перспективной коррекции изображения
- Параллельная обработка кадров
- Автоматическая визуализация и построение графиков

Автор: [Ваше имя]
Версия: 2.0
Дата: 2025-01-20
"""

import cv2
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage import img_as_float64, restoration, img_as_ubyte
import scipy.signal

# ============================================================================
# ИМПОРТ МОДУЛЯ ДЕТЕКЦИИ ПЛАМЕНИ
# ============================================================================

try:
    from image_processing import detect_flame_color_improved
    FLAME_DETECTOR_AVAILABLE = True
except ImportError:
    FLAME_DETECTOR_AVAILABLE = False
    print("⚠ Модуль image_processing не найден, адаптивная детекция пламени недоступна")


# ============================================================================
# КОНСТАНТЫ
# ============================================================================

# Полигон по умолчанию (координаты ROI области для анализа)
DEFAULT_POLYGON = np.array([
    [215, 113], [625, 118], [733, 270],
    [577, 529], [144, 530], [54, 277]
], dtype=np.int32)

# Размер изображения по умолчанию
DEFAULT_TARGET_SIZE = (928, 576)

# Границы зон для статистики (в долях от высоты полигона)
ZONE1_BOUNDARY = 0.3  # Дальняя зона: 0-30%
ZONE2_BOUNDARY = 0.7  # Средняя зона: 30-70%
                      # Ближняя зона: 70-100%


# ============================================================================
# ГЛАВНЫЙ КЛАСС АНАЛИЗАТОРА
# ============================================================================

class ImprovedShikhtaAnalyzer:
    """
    Улучшенный анализатор шихты с зональной обработкой.
    
    Класс предоставляет функционал для:
    - Обработки видеокадров с детекцией шихты
    - Адаптивной сегментации в зависимости от наличия пламени
    - Разделения области на зоны и подсчёта статистики
    - Визуализации результатов и построения графиков
    
    Attributes:
        polygon (np.ndarray): Полигон области интереса (ROI)
        target_size (tuple): Размер целевого изображения (width, height)
        perspective_transformer: Объект для перспективной коррекции
        min_contour_area (int): Минимальная площадь контура для фильтрации
        near_zone_ratio (float): Доля ближней зоны для алгоритма (0.0-1.0)
        near_zone_c_offset (int): Параметр C для adaptiveThreshold в ближней зоне
        far_zone_c_offset (int): Параметр C для adaptiveThreshold в дальней зоне
        near_zone_area_multiplier (float): Множитель площади для ближней зоны
        use_adaptive_flame_detection (bool): Использовать адаптивную детекцию пламени
        far_c_boost_no_flame (int): Увеличение чувствительности без пламени
        flame_detection_threshold (float): Порог определения наличия пламени (%)
    """
    
    def __init__(
        self,
        polygon=None,
        target_size=DEFAULT_TARGET_SIZE,
        perspective_transformer=None,
        min_contour_area=100,
        near_zone_ratio=0.5,
        near_zone_c_offset=-5,
        far_zone_c_offset=5,
        near_zone_area_multiplier=2,
        use_adaptive_flame_detection=True,
        far_c_boost_no_flame=5,
        flame_detection_threshold=15.0
    ):
        """
        Инициализация анализатора.
        
        Args:
            polygon: Полигон области интереса или None (будет использован дефолтный)
            target_size: Размер целевого изображения (width, height)
            perspective_transformer: Объект для перспективной коррекции
            min_contour_area: Минимальная площадь контура (пикс²)
            near_zone_ratio: Доля ближней зоны (0.0-1.0)
            near_zone_c_offset: Параметр C для ближней зоны (-15 до -3)
            far_zone_c_offset: Параметр C для дальней зоны (3-12)
            near_zone_area_multiplier: Множитель площади для ближней зоны (2.0-5.0)
            use_adaptive_flame_detection: Включить адаптивную детекцию пламени
            far_c_boost_no_flame: Увеличение far_c без пламени (3-8)
            flame_detection_threshold: Порог наличия пламени в процентах (10-25)
        """
        # Инициализация полигона
        self.polygon = self._initialize_polygon(polygon)
        
        # Параметры обработки
        self.target_size = target_size
        self.perspective_transformer = perspective_transformer
        self.min_contour_area = min_contour_area
        self.near_zone_ratio = near_zone_ratio
        self.near_zone_c_offset = near_zone_c_offset
        self.far_zone_c_offset = far_zone_c_offset
        self.near_zone_area_multiplier = near_zone_area_multiplier
        
        # Параметры адаптивной детекции пламени
        self.use_adaptive_flame_detection = (
            use_adaptive_flame_detection and FLAME_DETECTOR_AVAILABLE
        )
        self.far_c_boost_no_flame = far_c_boost_no_flame
        self.flame_detection_threshold = flame_detection_threshold
        
        # Хранилище метрик
        self.frame_metrics = []
        
        # Состояние последнего кадра (для визуализации и логирования)
        self._last_frame_has_flame = False
        self._last_frame_brightness = 0.0
        self._last_flame_percent = 0.0
        self._last_zone1_percent = 0.0
        self._last_zone2_percent = 0.0
        self._last_zone3_percent = 0.0
        self._last_detection_mode = 'UNKNOWN'
        self._last_far_c_used = far_zone_c_offset
        
        # Предвычисленные маски (создаются в _setup_masks)
        self._polygon_mask = None
        self._left_mask = None
        self._right_mask = None
        self._near_mask = None
        self._far_mask = None
        self._zone1_mask = None
        self._zone2_mask = None
        self._zone3_mask = None
        self._mid_x = None
        
        # Инициализация масок
        self._setup_masks()
    
    # ========================================================================
    # ИНИЦИАЛИЗАЦИЯ И НАСТРОЙКА
    # ========================================================================
    
    def _initialize_polygon(self, polygon):
        """
        Инициализация и валидация полигона.
        
        Args:
            polygon: Полигон или None
            
        Returns:
            np.ndarray: Валидный полигон в формате int32
        """
        if polygon is None:
            return DEFAULT_POLYGON.copy()
        
        if not isinstance(polygon, np.ndarray):
            try:
                polygon = np.array(polygon, dtype=np.int32)
            except Exception:
                print("⚠ Ошибка преобразования полигона, используется дефолтный")
                return DEFAULT_POLYGON.copy()
        
        return polygon.astype(np.int32)
    
    def _setup_masks(self):
        """
        Создание всех предвычисленных масок для ускорения обработки.
        
        Создаёт следующие маски:
        - Маска полигона (общая ROI область)
        - Маски левой и правой частей
        - Маски ближней и дальней зон (для алгоритма)
        - Маски 3 зон для статистики (0-30%, 30-70%, 70-100%)
        """
        # Создаём пустое изображение размера target_size
        dummy = np.zeros(self.target_size[::-1], dtype=np.uint8)
        
        # ===== Основная маска полигона =====
        self._polygon_mask = np.zeros_like(dummy, dtype=np.uint8)
        cv2.fillPoly(self._polygon_mask, [self.polygon], 255)
        
        # ===== Разделение на левую/правую части =====
        self._mid_x = (self.polygon[:, 0].min() + self.polygon[:, 0].max()) // 2
        
        # Разделяем точки полигона на левые и правые
        left_polygon = [pt for pt in self.polygon if pt[0] < self._mid_x]
        right_polygon = [pt for pt in self.polygon if pt[0] >= self._mid_x]
        
        left_polygon = np.array(left_polygon, np.int32)
        right_polygon = np.array(right_polygon, np.int32)
        
        self._left_mask = np.zeros_like(dummy, dtype=np.uint8)
        self._right_mask = np.zeros_like(dummy, dtype=np.uint8)
        cv2.fillPoly(self._left_mask, [left_polygon], 255)
        cv2.fillPoly(self._right_mask, [right_polygon], 255)
        
        # ===== Зональное разделение по Y для алгоритма =====
        y_coords = self.polygon[:, 1]
        y_min, y_max = y_coords.min(), y_coords.max()
        y_threshold = y_min + (y_max - y_min) * self.near_zone_ratio
        
        # Ближняя зона (нижняя часть изображения, больше Y)
        self._near_mask = self._polygon_mask.copy()
        self._near_mask[:int(y_threshold), :] = 0
        
        # Дальняя зона (верхняя часть изображения, меньше Y)
        self._far_mask = self._polygon_mask.copy()
        self._far_mask[int(y_threshold):, :] = 0
        
        # ===== 3 зоны для статистики =====
        y_zone1 = y_min + (y_max - y_min) * ZONE1_BOUNDARY
        y_zone2 = y_min + (y_max - y_min) * ZONE2_BOUNDARY
        
        # Зона 1: 0-30% от верха (дальняя)
        self._zone1_mask = self._polygon_mask.copy()
        self._zone1_mask[int(y_zone1):, :] = 0
        
        # Зона 2: 30-70% (средняя)
        self._zone2_mask = self._polygon_mask.copy()
        self._zone2_mask[:int(y_zone1), :] = 0
        self._zone2_mask[int(y_zone2):, :] = 0
        
        # Зона 3: 70-100% (ближняя)
        self._zone3_mask = self._polygon_mask.copy()
        self._zone3_mask[:int(y_zone2), :] = 0
    
    # ========================================================================
    # ПРЕДОБРАБОТКА ИЗОБРАЖЕНИЯ
    # ========================================================================
    
    def preprocess_frame(self, frame):
        """
        Предобработка кадра: перспективная коррекция + фильтрация.
        
        Применяет следующую обработку:
        1. Перспективная коррекция (если настроена)
        2. Конвертация в grayscale
        3. Изменение размера до target_size
        4. Эквализация гистограммы
        5. Фильтр Винера для шумоподавления
        6. CLAHE для улучшения контраста
        
        Args:
            frame: Входной кадр (BGR изображение)
            
        Returns:
            tuple: (original_gray, preprocessed)
                - original_gray: Исходное изображение в grayscale
                - preprocessed: Предобработанное изображение
        """
        # Применяем перспективную коррекцию если настроена
        if self.perspective_transformer is not None:
            frame = self.perspective_transformer.transform(frame)
        
        # Конвертация в grayscale
        if len(frame.shape) == 3:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            img = frame
        
        # Изменение размера если нужно
        if img.shape[:2] != self.target_size[::-1]:
            img = cv2.resize(img, self.target_size)
        
        # Эквализация гистограммы
        img_hist_eq = cv2.equalizeHist(img)
        img_float = img_as_float64(img_hist_eq)
        
        # Фильтр Винера для шумоподавления
        kernel = np.ones((15, 15), np.float64)
        image_filtered = scipy.signal.convolve2d(img_float, kernel, 'same')
        img_wiener = restoration.wiener(image_filtered, kernel, 5.1e4)
        img_wiener = img_as_ubyte(img_wiener)
        
        # CLAHE для улучшения локального контраста
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        image_filtered = clahe.apply(img_wiener)
        
        return img, image_filtered
    
    # ========================================================================
    # ДЕТЕКЦИЯ ПЛАМЕНИ
    # ========================================================================
    
    def detect_flame_presence(self, frame, preprocessed_img):
        """
        Определение наличия яркого пламени в кадре.
        
        Использует два метода:
        1. Цветовой анализ (HSV) - если доступен модуль image_processing
        2. Анализ яркости - в качестве fallback
        
        Args:
            frame: Оригинальный цветной кадр
            preprocessed_img: Предобработанное grayscale изображение
            
        Returns:
            tuple: (has_flame, brightness_level, flame_percent)
                - has_flame: True если обнаружено пламя
                - brightness_level: Средняя яркость в ROI
                - flame_percent: Процент площади пламени
        """
        # Анализ яркости в ROI
        masked_img = cv2.bitwise_and(preprocessed_img, self._polygon_mask)
        pixels = masked_img[self._polygon_mask > 0]
        
        if len(pixels) == 0:
            return False, 0.0, 0.0
        
        brightness_level = float(np.mean(pixels))
        
        # Метод 1: Цветовая детекция пламени
        flame_percent = 0.0
        has_flame_color = False
        
        if FLAME_DETECTOR_AVAILABLE and self.use_adaptive_flame_detection:
            try:
                # Используем функцию из image_processing
                flame_mask = detect_flame_color_improved(frame)
                flame_mask_roi = cv2.bitwise_and(flame_mask, self._polygon_mask)
                
                # Подсчёт процента пламени
                flame_pixels = cv2.countNonZero(flame_mask_roi)
                total_pixels = cv2.countNonZero(self._polygon_mask)
                flame_percent = (
                    (flame_pixels / total_pixels * 100)
                    if total_pixels > 0
                    else 0.0
                )
                
                # Пламя обнаружено если процент > порога И яркость высокая
                has_flame_color = (
                    flame_percent > self.flame_detection_threshold and
                    brightness_level > 110
                )
                
            except Exception as e:
                print(f"⚠ Ошибка детекции пламени: {e}")
                has_flame_color = False
        
        # Финальное решение: цвет (если доступен) ИЛИ яркость
        if FLAME_DETECTOR_AVAILABLE and self.use_adaptive_flame_detection:
            has_flame = has_flame_color
        else:
            # Fallback: простая проверка по яркости
            bright_pixels = np.sum(pixels > 200)
            bright_ratio = bright_pixels / len(pixels)
            has_flame = (bright_ratio > 0.05) and (brightness_level > 120)
        
        return has_flame, brightness_level, flame_percent
    
    def detect_flame_regions_for_exclusion(self, preprocessed_img):
        """
        Детекция областей с ярким пламенем для исключения из анализа шихты.
        
        Args:
            preprocessed_img: Предобработанное изображение
            
        Returns:
            np.ndarray: Бинарная маска областей пламени
        """
        _, flame_mask = cv2.threshold(preprocessed_img, 220, 255, cv2.THRESH_BINARY)
        kernel = np.ones((15, 15), np.uint8)
        flame_mask = cv2.dilate(flame_mask, kernel, iterations=2)
        return flame_mask
    
    # ========================================================================
    # СЕГМЕНТАЦИЯ ШИХТЫ
    # ========================================================================
    
    def segment_shikhta(self, preprocessed_img, polygon_mask, frame=None):
        """
        Обёртка для обратной совместимости.
        
        Args:
            preprocessed_img: Предобработанное изображение
            polygon_mask: Маска полигона
            frame: Оригинальный кадр (опционально)
            
        Returns:
            np.ndarray: Бинарная маска детектированной шихты
        """
        return self.segment_shikhta_adaptive(preprocessed_img, polygon_mask, frame)
    
    def segment_shikhta_adaptive(self, preprocessed_img, polygon_mask, frame=None):
        """
        Адаптивная сегментация шихты с подстройкой под наличие пламени.
        
        Алгоритм:
        1. Определение наличия пламени в кадре
        2. Адаптация параметров в зависимости от наличия пламени:
           - С пламенем: базовые параметры
           - Без пламени: повышенная чувствительность в дальней зоне
        3. Сегментация adaptiveThreshold для ближней и дальней зон
        4. Исключение областей яркого пламени
        5. Морфологическая фильтрация
        6. Удаление мелких контуров (с разными порогами для зон)
        
        Args:
            preprocessed_img: Предобработанное изображение
            polygon_mask: Маска полигона
            frame: Оригинальный кадр для детекции пламени
            
        Returns:
            np.ndarray: Бинарная маска детектированной шихты
        """
        # === ШАГ 1: Определение наличия пламени ===
        if frame is not None and self.use_adaptive_flame_detection:
            has_flame, brightness, flame_percent = self.detect_flame_presence(
                frame, preprocessed_img
            )
            # Сохраняем для метрик и визуализации
            self._last_frame_has_flame = has_flame
            self._last_frame_brightness = brightness
            self._last_flame_percent = flame_percent
        else:
            # Простая проверка без цветового детектора
            masked_img = cv2.bitwise_and(preprocessed_img, polygon_mask)
            pixels = masked_img[polygon_mask > 0]
            brightness = float(np.mean(pixels)) if len(pixels) > 0 else 0.0
            bright_pixels = np.sum(pixels > 200) if len(pixels) > 0 else 0
            bright_ratio = bright_pixels / len(pixels) if len(pixels) > 0 else 0
            has_flame = (bright_ratio > 0.05) and (brightness > 120)
            flame_percent = bright_ratio * 100
            
            self._last_frame_has_flame = has_flame
            self._last_frame_brightness = brightness
            self._last_flame_percent = flame_percent
        
        # === ШАГ 2: Адаптация параметров ===
        if has_flame:
            # С пламенем: используем базовые параметры
            far_c = self.far_zone_c_offset
            far_min_area = self.min_contour_area
            detection_mode = "FLAME"
        else:
            # Без пламени: повышаем чувствительность в дальней зоне
            far_c = self.far_zone_c_offset + self.far_c_boost_no_flame
            far_min_area = max(self.min_contour_area // 2, 30)
            detection_mode = "NO_FLAME"
        
        # === ШАГ 3: Детекция областей яркого пламени для исключения ===
        flame_mask = self.detect_flame_regions_for_exclusion(preprocessed_img)
        
        # === ШАГ 4: Сегментация для ближней зоны ===
        near_thresh = cv2.adaptiveThreshold(
            preprocessed_img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            191,  # blockSize
            self.near_zone_c_offset  # C (строгий порог)
        )
        
        # === ШАГ 5: Сегментация для дальней зоны (адаптивный порог!) ===
        far_thresh = cv2.adaptiveThreshold(
            preprocessed_img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            191,  # blockSize
            far_c  # C (адаптируется к наличию пламени)
        )
        
        # === ШАГ 6: Комбинирование результатов по зонам ===
        combined_thresh = np.zeros_like(near_thresh)
        combined_thresh = cv2.bitwise_or(
            combined_thresh,
            cv2.bitwise_and(near_thresh, self._near_mask)
        )
        combined_thresh = cv2.bitwise_or(
            combined_thresh,
            cv2.bitwise_and(far_thresh, self._far_mask)
        )
        
        # === ШАГ 7: Исключение областей с пламенем ===
        combined_thresh = cv2.bitwise_and(
            combined_thresh,
            cv2.bitwise_not(flame_mask)
        )
        
        # === ШАГ 8: Морфологическая обработка ===
        kernel_close = np.ones((7, 7), np.uint8)
        combined_thresh = cv2.morphologyEx(
            combined_thresh,
            cv2.MORPH_CLOSE,
            kernel_close
        )
        
        # === ШАГ 9: Удаление мелких объектов в ближней зоне ===
        near_region = cv2.bitwise_and(combined_thresh, self._near_mask)
        near_region = self._remove_small_contours(
            near_region,
            min_area=int(self.min_contour_area * self.near_zone_area_multiplier)
        )
        
        # === ШАГ 10: Удаление мелких объектов в дальней зоне ===
        far_region = cv2.bitwise_and(combined_thresh, self._far_mask)
        far_region = self._remove_small_contours(
            far_region,
            min_area=far_min_area  # Адаптивный порог
        )
        
        # === ШАГ 11: Объединение обработанных зон ===
        combined_thresh = cv2.bitwise_or(near_region, far_region)
        
        # === ШАГ 12: Применение маски полигона ===
        combined_thresh = cv2.bitwise_and(combined_thresh, polygon_mask)
        
        # Сохраняем параметры для логирования
        self._last_detection_mode = detection_mode
        self._last_far_c_used = far_c
        
        return combined_thresh
    
    def _remove_small_contours(self, binary_img, min_area=50):
        """
        Удаление контуров меньше заданной площади.
        
        Args:
            binary_img: Бинарное изображение
            min_area: Минимальная площадь контура (пикс²)
            
        Returns:
            np.ndarray: Отфильтрованное бинарное изображение
        """
        contours, _ = cv2.findContours(
            binary_img,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        filtered = np.zeros_like(binary_img)
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area:
                cv2.drawContours(filtered, [cnt], -1, 255, -1)
        
        return filtered
    
    # ========================================================================
    # АНАЛИЗ КАДРА
    # ========================================================================
    
    def analyze_frame(self, frame, frame_idx=0, save_visualization=False,
                      output_path=None):
        """
        Анализ одного кадра с адаптивной детекцией пламени.
        
        Args:
            frame: Входной кадр (BGR изображение)
            frame_idx: Индекс кадра
            save_visualization: Сохранить визуализацию
            output_path: Путь для сохранения визуализации
            
        Returns:
            dict: Словарь с метриками кадра
        """
        original_frame = frame.copy()
        
        # Предобработка
        img_gray, preprocessed = self.preprocess_frame(frame)
        
        # Сегментация с адаптацией под пламя
        thresh = self.segment_shikhta_adaptive(
            preprocessed, self._polygon_mask, frame=original_frame
        )
        
        # Поиск контуров
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Разделение контуров (лево/право)
        left_thresh = cv2.bitwise_and(thresh, self._left_mask)
        right_thresh = cv2.bitwise_and(thresh, self._right_mask)
        
        left_contours, _ = cv2.findContours(
            left_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        right_contours, _ = cv2.findContours(
            right_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Расчёт площадей (лево/право)
        left_area = sum(cv2.contourArea(cnt) for cnt in left_contours)
        right_area = sum(cv2.contourArea(cnt) for cnt in right_contours)
        total_area = left_area + right_area
        
        # Расчёт площадей по 3 зонам
        zone1_thresh = cv2.bitwise_and(thresh, self._zone1_mask)
        zone2_thresh = cv2.bitwise_and(thresh, self._zone2_mask)
        zone3_thresh = cv2.bitwise_and(thresh, self._zone3_mask)
        
        zone1_contours, _ = cv2.findContours(
            zone1_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        zone2_contours, _ = cv2.findContours(
            zone2_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        zone3_contours, _ = cv2.findContours(
            zone3_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        zone1_area = sum(cv2.contourArea(cnt) for cnt in zone1_contours)
        zone2_area = sum(cv2.contourArea(cnt) for cnt in zone2_contours)
        zone3_area = sum(cv2.contourArea(cnt) for cnt in zone3_contours)
        zones_total_area = zone1_area + zone2_area + zone3_area
        
        # Процентное соотношение (лево/право)
        if total_area > 0:
            left_percent = (left_area / total_area) * 100
            right_percent = (right_area / total_area) * 100
        else:
            left_percent = right_percent = 0
        
        # Процент по зонам (от общей шихты)
        if zones_total_area > 0:
            zone1_percent = (zone1_area / zones_total_area) * 100
            zone2_percent = (zone2_area / zones_total_area) * 100
            zone3_percent = (zone3_area / zones_total_area) * 100
        else:
            zone1_percent = zone2_percent = zone3_percent = 0
        
        # Сохраняем для визуализации
        self._last_zone1_percent = zone1_percent
        self._last_zone2_percent = zone2_percent
        self._last_zone3_percent = zone3_percent
        
        # Формирование метрик
        metrics = {
            'frame_idx': frame_idx,
            'left_area': float(left_area),
            'right_area': float(right_area),
            'total_area': float(total_area),
            'left_percent': float(left_percent),
            'right_percent': float(right_percent),
            'left_contours_count': len(left_contours),
            'right_contours_count': len(right_contours),
            # Метрики по 3 зонам
            'zone1_area': float(zone1_area),
            'zone2_area': float(zone2_area),
            'zone3_area': float(zone3_area),
            'zone1_percent': float(zone1_percent),
            'zone2_percent': float(zone2_percent),
            'zone3_percent': float(zone3_percent),
            'zone1_contours_count': len(zone1_contours),
            'zone2_contours_count': len(zone2_contours),
            'zone3_contours_count': len(zone3_contours),
            'perspective_corrected': self.perspective_transformer is not None,
            # Метрики о пламени
            'has_flame': self._last_frame_has_flame,
            'brightness': self._last_frame_brightness,
            'flame_percent': self._last_flame_percent,
            'detection_mode': self._last_detection_mode,
            'far_c_used': self._last_far_c_used
        }
        
        # Визуализация
        if save_visualization and output_path:
            self._save_visualization(
                original_frame, contours,
                left_percent, right_percent, output_path,
                has_flame=metrics['has_flame'],
                flame_percent=metrics['flame_percent'],
                detection_mode=metrics['detection_mode']
            )
        
        return metrics
    
    # ========================================================================
    # ВИЗУАЛИЗАЦИЯ
    # ========================================================================
    
    def _save_visualization(self, original_frame, contours,
                            left_percent, right_percent, output_path,
                            has_flame=False, flame_percent=0.0, 
                            detection_mode='UNKNOWN'):
        """
        Сохранение визуализации с информацией о пламени и зонах.
        
        Args:
            original_frame: Оригинальный кадр
            contours: Список контуров шихты
            left_percent: Процент левой части
            right_percent: Процент правой части
            output_path: Путь для сохранения
            has_flame: Наличие пламени
            flame_percent: Процент пламени
            detection_mode: Режим детекции
        """
        # Применяем перспективу если нужно
        if self.perspective_transformer:
            img_color = self.perspective_transformer.transform(original_frame)
        else:
            img_color = cv2.resize(original_frame.copy(), self.target_size)
        
        min_y = self.polygon[:, 1].min()
        max_y = self.polygon[:, 1].max()
        
        # Рисуем контуры и границы
        cv2.drawContours(img_color, contours, -1, (0, 255, 0), 1)
        cv2.polylines(img_color, [self.polygon], True, (255, 0, 0), 2)
        cv2.line(img_color, (self._mid_x, min_y),
                 (self._mid_x, max_y), (0, 0, 255), 2)
        
        # Показываем зональное разделение (для алгоритма)
        y_threshold = int(min_y + (max_y - min_y) * self.near_zone_ratio)
        cv2.line(img_color, (self.polygon[:, 0].min(), y_threshold),
                 (self.polygon[:, 0].max(), y_threshold), (255, 255, 0), 1)
        
        # Показываем границы 3 зон для статистики
        y_zone1 = int(min_y + (max_y - min_y) * ZONE1_BOUNDARY)
        y_zone2 = int(min_y + (max_y - min_y) * ZONE2_BOUNDARY)
        cv2.line(img_color, (self.polygon[:, 0].min(), y_zone1),
                 (self.polygon[:, 0].max(), y_zone1), (52, 152, 219), 1)
        cv2.line(img_color, (self.polygon[:, 0].min(), y_zone2),
                 (self.polygon[:, 0].max(), y_zone2), (231, 76, 60), 1)
        
        # Текст с метриками
        text = f"L: {left_percent:.1f}% | R: {right_percent:.1f}%"
        if self.perspective_transformer:
            text += " [PERSP]"
        cv2.putText(img_color, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Информация о пламени и режиме детекции
        y_offset = 60
        if self.use_adaptive_flame_detection:
            flame_text = f"Flame: {'YES' if has_flame else 'NO'} ({flame_percent:.1f}%)"
            flame_color = (0, 255, 255) if has_flame else (128, 128, 128)
            cv2.putText(img_color, flame_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, flame_color, 2)
            
            mode_text = f"Mode: {detection_mode}"
            mode_color = (0, 200, 0) if detection_mode == "FLAME" else (200, 200, 0)
            cv2.putText(img_color, mode_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
            y_offset = 120
        
        # Информация о зонах на текущем кадре
        if hasattr(self, '_last_zone1_percent'):
            zone_text = (f"Z1: {self._last_zone1_percent:.1f}% | "
                         f"Z2: {self._last_zone2_percent:.1f}% | "
                         f"Z3: {self._last_zone3_percent:.1f}%")
            cv2.putText(img_color, zone_text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
            y_offset += 30
        
        # Дополнительная информация
        info_text = f"Contours: {len(contours)} | MinArea: {self.min_contour_area}"
        cv2.putText(img_color, info_text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Композитное изображение если используется перспектива
        if self.perspective_transformer:
            h, w = img_color.shape[:2]
            composite = np.zeros((h, w * 2 + 10, 3), dtype=np.uint8)
            
            orig_resized = cv2.resize(original_frame, (w, h))
            composite[:, :w] = orig_resized
            composite[:, w:w+10] = 128
            composite[:, w+10:] = img_color
            
            cv2.imwrite(output_path, composite)
        else:
            cv2.imwrite(output_path, img_color)
    
    # ========================================================================
    # ОБРАБОТКА ВИДЕО
    # ========================================================================
    
    def _process_single_frame_wrapper(self, task):
        """
        Обёртка для параллельной обработки одного кадра.
        
        Args:
            task: Кортеж (idx, frame_path, output_dir, save_every_n)
            
        Returns:
            dict: Метрики кадра или None при ошибке
        """
        idx, frame_path, output_dir, save_every_n = task
        frame = cv2.imread(str(frame_path))
        if frame is None:
            return None
        
        save_vis = output_dir and (idx % save_every_n == 0)
        vis_path = None
        if save_vis:
            vis_path = os.path.join(output_dir, f"shikhta_{idx:06d}.jpg")
        
        return self.analyze_frame(frame, frame_idx=idx,
                                  save_visualization=save_vis,
                                  output_path=vis_path)
    
    def process_video_frames(self, frames_dir, output_dir=None, save_every_n=10,
                             use_parallel=True, max_workers=4, max_frames=None):
        """
        Обработка всех кадров видео с поддержкой параллелизма.
        
        Args:
            frames_dir: Директория с кадрами
            output_dir: Директория для сохранения визуализаций
            save_every_n: Сохранять каждый N-й кадр
            use_parallel: Использовать параллельную обработку
            max_workers: Максимальное количество потоков
            max_frames: Максимальное количество кадров для обработки
            
        Returns:
            list: Список метрик всех кадров
        """
        frame_files = sorted(Path(frames_dir).glob("*.jpg"))
        if max_frames:
            frame_files = frame_files[:max_frames]
        
        if not frame_files:
            print(f"Нет кадров в {frames_dir}")
            return []
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        self.frame_metrics = []
        
        print(f"Обработка {len(frame_files)} кадров из {Path(frames_dir).name}...")
        if self.perspective_transformer:
            print("  • Перспективная коррекция: включена")
        print(f"  • Зональная адаптация: включена (порог={self.near_zone_ratio:.0%})")
        print(f"  • Мин. площадь: {self.min_contour_area} пикс² (ближняя зона: x{self.near_zone_area_multiplier})")
        print(f"  • Пороги C: ближняя={self.near_zone_c_offset}, дальняя={self.far_zone_c_offset}")
        
        if self.use_adaptive_flame_detection:
            print(f"  • АДАПТИВНАЯ детекция пламени: включена (boost без пламени: +{self.far_c_boost_no_flame})")
            print(f"  • Порог определения пламени: {self.flame_detection_threshold}%")
        else:
            print("  • Адаптивная детекция пламени: выключена")
        
        if use_parallel and len(frame_files) > 50:
            # Параллельная обработка
            print(f"  • Режим: параллельный ({max_workers} потоков)")
            
            tasks = [(idx, frame_path, output_dir, save_every_n)
                     for idx, frame_path in enumerate(frame_files)]
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self._process_single_frame_wrapper, task)
                           for task in tasks]
                
                for future in as_completed(futures):
                    metrics = future.result()
                    if metrics:
                        self.frame_metrics.append(metrics)
                    
                    processed = len(self.frame_metrics)
                    if processed % 100 == 0:
                        print(f"    Обработано {processed}/{len(frame_files)} кадров")
            
            # Сортировка по frame_idx
            self.frame_metrics.sort(key=lambda x: x['frame_idx'])
        else:
            # Последовательная обработка
            print("  • Режим: последовательный")
            for idx, frame_path in enumerate(frame_files):
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    continue
                
                save_vis = output_dir and (idx % save_every_n == 0)
                vis_path = None
                if save_vis:
                    vis_path = os.path.join(output_dir, f"shikhta_{idx:06d}.jpg")
                
                metrics = self.analyze_frame(frame, frame_idx=idx,
                                             save_visualization=save_vis,
                                             output_path=vis_path)
                self.frame_metrics.append(metrics)
                
                if (idx + 1) % 100 == 0:
                    print(f"    Обработано {idx + 1}/{len(frame_files)} кадров")
        
        print(f"✓ Обработано {len(self.frame_metrics)} кадров")
        return self.frame_metrics
    
    # ========================================================================
    # СТАТИСТИКА И МЕТРИКИ
    # ========================================================================
    
    def compute_summary_statistics(self):
        """
        Вычисление сводной статистики по всем обработанным кадрам.
        
        Returns:
            dict: Словарь со сводной статистикой или None если нет данных
        """
        if not self.frame_metrics:
            return None
        
        left_percents = [m['left_percent'] for m in self.frame_metrics]
        right_percents = [m['right_percent'] for m in self.frame_metrics]
        total_areas = [m['total_area'] for m in self.frame_metrics]
        
        # Статистика по 3 зонам
        zone1_percents = [m['zone1_percent'] for m in self.frame_metrics]
        zone2_percents = [m['zone2_percent'] for m in self.frame_metrics]
        zone3_percents = [m['zone3_percent'] for m in self.frame_metrics]
        
        summary = {
            'total_frames': len(self.frame_metrics),
            'perspective_corrected': self.perspective_transformer is not None,
            'min_contour_area': self.min_contour_area,
            'near_zone_ratio': self.near_zone_ratio,
            'near_zone_c_offset': self.near_zone_c_offset,
            'far_zone_c_offset': self.far_zone_c_offset,
            'near_zone_area_multiplier': self.near_zone_area_multiplier,
            'adaptive_flame_detection': self.use_adaptive_flame_detection,
            'far_c_boost_no_flame': self.far_c_boost_no_flame,
            'flame_detection_threshold': self.flame_detection_threshold,
            # Статистика по пламени
            'flame_stats': {
                'frames_with_flame': sum(1 for m in self.frame_metrics if m.get('has_flame', False)),
                'frames_without_flame': sum(1 for m in self.frame_metrics if not m.get('has_flame', False)),
                'avg_flame_percent': float(np.mean([m.get('flame_percent', 0) for m in self.frame_metrics])),
                'avg_brightness': float(np.mean([m.get('brightness', 0) for m in self.frame_metrics]))
            },
            'left': {
                'mean': float(np.mean(left_percents)),
                'std': float(np.std(left_percents)),
                'min': float(np.min(left_percents)),
                'max': float(np.max(left_percents)),
                'median': float(np.median(left_percents))
            },
            'right': {
                'mean': float(np.mean(right_percents)),
                'std': float(np.std(right_percents)),
                'min': float(np.min(right_percents)),
                'max': float(np.max(right_percents)),
                'median': float(np.median(right_percents))
            },
            # Статистика по 3 зонам
            'zone1': {
                'name': 'Дальняя зона (0-30%)',
                'description': 'Процент от общей шихты в этой зоне',
                'mean': float(np.mean(zone1_percents)),
                'std': float(np.std(zone1_percents)),
                'min': float(np.min(zone1_percents)),
                'max': float(np.max(zone1_percents)),
                'median': float(np.median(zone1_percents))
            },
            'zone2': {
                'name': 'Средняя зона (30-70%)',
                'description': 'Процент от общей шихты в этой зоне',
                'mean': float(np.mean(zone2_percents)),
                'std': float(np.std(zone2_percents)),
                'min': float(np.min(zone2_percents)),
                'max': float(np.max(zone2_percents)),
                'median': float(np.median(zone2_percents))
            },
            'zone3': {
                'name': 'Ближняя зона (70-100%)',
                'description': 'Процент от общей шихты в этой зоне',
                'mean': float(np.mean(zone3_percents)),
                'std': float(np.std(zone3_percents)),
                'min': float(np.min(zone3_percents)),
                'max': float(np.max(zone3_percents)),
                'median': float(np.median(zone3_percents))
            },
            'total_area': {
                'mean': float(np.mean(total_areas)),
                'std': float(np.std(total_areas)),
                'min': float(np.min(total_areas)),
                'max': float(np.max(total_areas))
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def save_metrics(self, output_path):
        """
        Сохранение метрик в JSON с автоматической генерацией графиков.
        
        Args:
            output_path: Путь для сохранения JSON файла
            
        Returns:
            dict: Сводная статистика
        """
        summary = self.compute_summary_statistics()
        
        data = {
            'summary': summary,
            'frames': self.frame_metrics
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Метрики сохранены в {output_path}")
        
        # Автоматическая генерация графиков
        plot_path = output_path.replace('.json', '_plot.png')
        self.plot_metrics(plot_path)
        
        return summary
    
    # ========================================================================
    # ПОСТРОЕНИЕ ГРАФИКОВ
    # ========================================================================
    
    def plot_metrics(self, output_path):
        """
        Построение графиков с выделением мин/макс и статистикой по зонам.
        
        Args:
            output_path: Путь для сохранения графика
        """
        if not self.frame_metrics:
            print("Нет данных для построения графиков")
            return
        
        # Извлечение данных
        frames = [m['frame_idx'] for m in self.frame_metrics]
        left_percents = [m['left_percent'] for m in self.frame_metrics]
        right_percents = [m['right_percent'] for m in self.frame_metrics]
        total_areas = [m['total_area'] for m in self.frame_metrics]
        
        # Данные по 3 зонам
        zone1_percents = [m.get('zone1_percent', 0) for m in self.frame_metrics]
        zone2_percents = [m.get('zone2_percent', 0) for m in self.frame_metrics]
        zone3_percents = [m.get('zone3_percent', 0) for m in self.frame_metrics]
        
        # Поиск экстремумов
        left_min_idx = np.argmin(left_percents)
        left_max_idx = np.argmax(left_percents)
        right_min_idx = np.argmin(right_percents)
        right_max_idx = np.argmax(right_percents)
        
        # Создание 3 графиков
        fig, axes = plt.subplots(3, 1, figsize=(14, 14))
        
        # Заголовок
        title = 'Анализ шихты (улучшенный алгоритм)'
        if self.perspective_transformer:
            title += ' с перспективной коррекцией'
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        
        # ========== График 1: Лево/Право ==========
        ax1 = axes[0]
        ax1.plot(frames, left_percents, label='Левая часть',
                 color='#2E86DE', linewidth=2, alpha=0.8)
        ax1.plot(frames, right_percents, label='Правая часть',
                 color='#EE5A6F', linewidth=2, alpha=0.8)
        ax1.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5,
                    label='Баланс 50/50')
        
        # Выделение экстремумов
        for idx, val, color, name in [
            (left_min_idx, left_percents[left_min_idx], 'blue', 'MIN L'),
            (left_max_idx, left_percents[left_max_idx], 'blue', 'MAX L'),
            (right_min_idx, right_percents[right_min_idx], 'red', 'MIN R'),
            (right_max_idx, right_percents[right_max_idx], 'red', 'MAX R')
        ]:
            marker = 'v' if 'MIN' in name else '^'
            ax1.scatter(frames[idx], val, color=color, s=150,
                        marker=marker, zorder=5, edgecolors='black', linewidth=2)
            ax1.annotate(f'{name}: {val:.1f}%',
                         xy=(frames[idx], val),
                         xytext=(10, -20 if 'MIN' in name else 20),
                         textcoords='offset points',
                         bbox=dict(boxstyle='round,pad=0.5',
                                   facecolor='lightblue' if 'L' in name else 'lightcoral',
                                   alpha=0.8),
                         arrowprops=dict(arrowstyle='->', color=color, lw=2))
        
        ax1.set_xlabel('Номер кадра', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Процент шихты, %', fontsize=11, fontweight='bold')
        ax1.set_title('Распределение шихты по левой и правой частям',
                      fontsize=13, fontweight='bold', pad=10)
        ax1.legend(loc='best', fontsize=9, framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        ax1.set_ylim(0, 100)
        
        # Статистика
        stats_text = (f'Левая: μ={np.mean(left_percents):.1f}% σ={np.std(left_percents):.1f}%\n'
                      f'Правая: μ={np.mean(right_percents):.1f}% σ={np.std(right_percents):.1f}%\n'
                      f'MinArea={self.min_contour_area} (x{self.near_zone_area_multiplier} ближ.)\n'
                      f'C: ближ={self.near_zone_c_offset} дал={self.far_zone_c_offset}')
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                 fontsize=8, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # ========== График 2: 3 зоны ==========
        ax2 = axes[1]
        
        ax2.plot(frames, zone1_percents,
                 color='#3498DB', linewidth=2.5, alpha=0.9,
                 label='Зона 1 (дальняя 0-30%)')
        ax2.plot(frames, zone2_percents,
                 color='#F39C12', linewidth=2.5, alpha=0.9,
                 label='Зона 2 (средняя 30-70%)')
        ax2.plot(frames, zone3_percents,
                 color='#E74C3C', linewidth=2.5, alpha=0.9,
                 label='Зона 3 (ближняя 70-100%)')
        
        # Средние значения
        ax2.axhline(y=np.mean(zone1_percents), color='#2980B9',
                    linestyle='--', linewidth=1.5, alpha=0.6)
        ax2.axhline(y=np.mean(zone2_percents), color='#E67E22',
                    linestyle='--', linewidth=1.5, alpha=0.6)
        ax2.axhline(y=np.mean(zone3_percents), color='#C0392B',
                    linestyle='--', linewidth=1.5, alpha=0.6)
        
        ax2.set_xlabel('Номер кадра', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Процент шихты в зоне, %', fontsize=11, fontweight='bold')
        ax2.set_title('Распределение шихты по зонам печи (от камеры)',
                      fontsize=13, fontweight='bold', pad=10)
        ax2.legend(loc='best', fontsize=9, framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        ax2.set_ylim(0, max(max(zone1_percents), max(zone2_percents), max(zone3_percents)) * 1.1)
        
        # Статистика зон
        zones_sum_mean = np.mean(zone1_percents) + np.mean(zone2_percents) + np.mean(zone3_percents)
        zone_stats_text = (f'Зона 1: μ={np.mean(zone1_percents):.1f}% σ={np.std(zone1_percents):.1f}% '
                           f'[{np.min(zone1_percents):.1f}%-{np.max(zone1_percents):.1f}%]\n'
                           f'Зона 2: μ={np.mean(zone2_percents):.1f}% σ={np.std(zone2_percents):.1f}% '
                           f'[{np.min(zone2_percents):.1f}%-{np.max(zone2_percents):.1f}%]\n'
                           f'Зона 3: μ={np.mean(zone3_percents):.1f}% σ={np.std(zone3_percents):.1f}% '
                           f'[{np.min(zone3_percents):.1f}%-{np.max(zone3_percents):.1f}%]\n'
                           f'Сумма зон: {zones_sum_mean:.1f}% (должно быть ≈100%)')
        ax2.text(0.02, 0.98, zone_stats_text, transform=ax2.transAxes,
                 fontsize=8, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))
        
        # ========== График 3: Общая площадь ==========
        ax3 = axes[2]
        ax3.fill_between(frames, total_areas, alpha=0.3, color='#10AC84')
        ax3.plot(frames, total_areas, color='#10AC84', linewidth=2)
        
        area_min_idx = np.argmin(total_areas)
        area_max_idx = np.argmax(total_areas)
        
        for idx, marker, name in [(area_min_idx, 'v', 'MIN'),
                                   (area_max_idx, '^', 'MAX')]:
            ax3.scatter(frames[idx], total_areas[idx], color='green',
                        s=150, marker=marker, zorder=5,
                        edgecolors='black', linewidth=2)
            ax3.annotate(f'{name}: {total_areas[idx]:.0f}',
                         xy=(frames[idx], total_areas[idx]),
                         xytext=(10, -20 if marker == 'v' else 20),
                         textcoords='offset points',
                         bbox=dict(boxstyle='round,pad=0.5',
                                   facecolor='lightgreen', alpha=0.8),
                         arrowprops=dict(arrowstyle='->', color='green', lw=2))
        
        ax3.set_xlabel('Номер кадра', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Площадь, пикс²', fontsize=11, fontweight='bold')
        ax3.set_title('Общая площадь шихты во времени',
                      fontsize=13, fontweight='bold', pad=10)
        ax3.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        
        area_stats = f'μ={np.mean(total_areas):.0f} σ={np.std(total_areas):.0f}'
        ax3.text(0.02, 0.98, area_stats, transform=ax3.transAxes,
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ График сохранён: {output_path}")
        plt.close()


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def load_metrics_from_json(json_path):
    """
    Загрузка метрик из JSON файла.
    
    Args:
        json_path: Путь к JSON файлу
        
    Returns:
        dict: Загруженные данные с метриками
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_metrics(metrics1, metrics2, output_path=None):
    """
    Сравнение метрик двух анализов.
    
    Args:
        metrics1: Первый набор метрик (dict)
        metrics2: Второй набор метрик (dict)
        output_path: Путь для сохранения графика сравнения
        
    Returns:
        dict: Словарь с результатами сравнения
    """
    summary1 = metrics1['summary']
    summary2 = metrics2['summary']
    
    comparison = {
        'left_mean_diff': summary2['left']['mean'] - summary1['left']['mean'],
        'right_mean_diff': summary2['right']['mean'] - summary1['right']['mean'],
        'total_area_mean_diff': summary2['total_area']['mean'] - summary1['total_area']['mean'],
        'zone1_mean_diff': summary2['zone1']['mean'] - summary1['zone1']['mean'],
        'zone2_mean_diff': summary2['zone2']['mean'] - summary1['zone2']['mean'],
        'zone3_mean_diff': summary2['zone3']['mean'] - summary1['zone3']['mean'],
    }
    
    if output_path:
        _plot_comparison(metrics1, metrics2, comparison, output_path)
    
    return comparison


def _plot_comparison(metrics1, metrics2, comparison, output_path):
    """
    Построение графика сравнения двух анализов.
    
    Args:
        metrics1: Первый набор метрик
        metrics2: Второй набор метрик
        comparison: Результаты сравнения
        output_path: Путь для сохранения
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Сравнение двух анализов', fontsize=16, fontweight='bold')
    
    # График 1: Лево/Право
    ax1 = axes[0, 0]
    frames1 = [m['frame_idx'] for m in metrics1['frames']]
    frames2 = [m['frame_idx'] for m in metrics2['frames']]
    left1 = [m['left_percent'] for m in metrics1['frames']]
    left2 = [m['left_percent'] for m in metrics2['frames']]
    
    ax1.plot(frames1, left1, label='Анализ 1', alpha=0.7, linewidth=2)
    ax1.plot(frames2, left2, label='Анализ 2', alpha=0.7, linewidth=2)
    ax1.set_title('Левая часть')
    ax1.set_ylabel('Процент, %')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График 2: Зоны
    ax2 = axes[0, 1]
    categories = ['Зона 1', 'Зона 2', 'Зона 3']
    values1 = [
        metrics1['summary']['zone1']['mean'],
        metrics1['summary']['zone2']['mean'],
        metrics1['summary']['zone3']['mean']
    ]
    values2 = [
        metrics2['summary']['zone2']['mean'],
        metrics2['summary']['zone2']['mean'],
        metrics2['summary']['zone3']['mean']
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    ax2.bar(x - width/2, values1, width, label='Анализ 1', alpha=0.8)
    ax2.bar(x + width/2, values2, width, label='Анализ 2', alpha=0.8)
    ax2.set_ylabel('Средний процент, %')
    ax2.set_title('Сравнение по зонам')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # График 3: Общая площадь
    ax3 = axes[1, 0]
    total1 = [m['total_area'] for m in metrics1['frames']]
    total2 = [m['total_area'] for m in metrics2['frames']]
    
    ax3.plot(frames1, total1, label='Анализ 1', alpha=0.7, linewidth=2)
    ax3.plot(frames2, total2, label='Анализ 2', alpha=0.7, linewidth=2)
    ax3.set_title('Общая площадь')
    ax3.set_xlabel('Номер кадра')
    ax3.set_ylabel('Площадь, пикс²')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # График 4: Таблица различий
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    diff_data = [
        ['Метрика', 'Разница'],
        ['Левая часть', f"{comparison['left_mean_diff']:+.2f}%"],
        ['Правая часть', f"{comparison['right_mean_diff']:+.2f}%"],
        ['Зона 1', f"{comparison['zone1_mean_diff']:+.2f}%"],
        ['Зона 2', f"{comparison['zone2_mean_diff']:+.2f}%"],
        ['Зона 3', f"{comparison['zone3_mean_diff']:+.2f}%"],
        ['Площадь', f"{comparison['total_area_mean_diff']:+.0f} пикс²"]
    ]
    
    table = ax4.table(cellText=diff_data, loc='center', cellLoc='left',
                      colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Форматирование заголовка
    for i in range(2):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ График сравнения сохранён: {output_path}")
    plt.close()


# ============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================================

if __name__ == "__main__":
    """
    Пример использования ImprovedShikhtaAnalyzer.
    
    Демонстрирует базовую настройку и запуск анализа.
    """
    
    print("="*70)
    print("УЛУЧШЕННЫЙ АНАЛИЗАТОР ШИХТЫ v2.0")
    print("="*70)
    
    # Создание анализатора с настройками по умолчанию
    analyzer = ImprovedShikhtaAnalyzer(
        min_contour_area=100,
        near_zone_ratio=0.5,
        near_zone_c_offset=-5,
        far_zone_c_offset=5,
        near_zone_area_multiplier=2.0,
        use_adaptive_flame_detection=True,
        far_c_boost_no_flame=5,
        flame_detection_threshold=15.0
    )
    
    print("\n✓ Анализатор инициализирован")
    print(f"  • Адаптивная детекция пламени: {'включена' if analyzer.use_adaptive_flame_detection else 'выключена'}")
    print(f"  • Полигон: {len(analyzer.polygon)} точек")
    print(f"  • Целевой размер: {analyzer.target_size}")
    
    # Пример обработки директории с кадрами
    # frames_dir = "path/to/frames"
    # output_dir = "path/to/output"
    # 
    # metrics = analyzer.process_video_frames(
    #     frames_dir=frames_dir,
    #     output_dir=output_dir,
    #     save_every_n=10,
    #     use_parallel=True,
    #     max_workers=4
    # )
    # 
    # # Сохранение результатов
    # analyzer.save_metrics("results/metrics.json")
    
    print("\n" + "="*70)
    print("Для использования раскомментируйте код в блоке __main__")
    print("="*70)