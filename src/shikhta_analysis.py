"""
Анализ шихты с поддержкой перспективной коррекции
Улучшенная версия с преобразованием "вид сверху"
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from skimage import img_as_float64, restoration
from skimage import img_as_ubyte

USE_IMPROVED_ANALYZER = True  # Измените на False для возврата к старому

try:
    from improved_shikhta_analyzer import ImprovedShikhtaAnalyzer
    IMPROVED_AVAILABLE = True
    print("✓ Улучшенный анализатор импортирован успешно")
except ImportError:
    IMPROVED_AVAILABLE = False
    ImprovedShikhtaAnalyzer = None
    print("⚠ Улучшенный анализатор не найден, используется стандартный")

# Ð˜Ð¼Ð¿Ð¾Ñ€Ñ‚ Ð¼Ð¾Ð´ÑƒÐ»Ñ Ð¿ÐµÑ€ÑÐ¿ÐµÐºÑ‚Ð¸Ð²Ð½Ð¾Ð¹ ÐºÐ¾Ñ€Ñ€ÐµÐºÑ†Ð¸Ð¸
try:
    from perspective_transform import PerspectiveTransformer, load_or_setup_perspective
except ImportError:
    print("Модуль perspective_transform.py не найден. Работа без перспективной коррекции.")
    PerspectiveTransformer = None
    load_or_setup_perspective = None

try:
    from perspective_transform_hexagon import (
        HexagonPerspectiveTransformer,
        setup_hexagon_perspective_gui
    )
    HEXAGON_PERSPECTIVE = True
except ImportError:
    HEXAGON_PERSPECTIVE = False
    HexagonPerspectiveTransformer = None
    setup_hexagon_perspective_gui = None
    print("⚠ Модуль perspective_transform_hexagon не найден")


def analyze_video_shikhta(frames_dir, output_dir, polygon=None,
                          save_visualizations=True, save_every_n=10,
                          use_parallel=True, max_workers=4,
                          use_perspective=True, perspective_config=None,
                          perspective_method='hexagon',min_contour_area=100,
                          near_zone_ratio=0.5):
    """
    Главная функция для анализа шихты в видео с перспективной коррекцией

    Args:
        frames_dir: директория с извлеченными кадрами
        output_dir: директория для сохранения результатов
        polygon: пользовательский полигон (опционально)
        save_visualizations: сохранять ли визуализации
        save_every_n: сохранять каждый N-й кадр
        use_parallel: использовать параллельную обработку
        max_workers: количество потоков для параллельной обработки
        use_perspective: использовать ли перспективную коррекцию
        perspective_config: путь к конфигурации перспективы или объект трансформера
        perspective_method: 'standard' (4 точки) или 'hexagon' (6 точек)

    Returns:
        summary: словарь со статистикой
    """
    start_time = time.time()

    # ÐÐ°ÑÑ‚Ñ€Ð¾Ð¹ÐºÐ° Ð¿ÐµÑ€ÑÐ¿ÐµÐºÑ‚Ð¸Ð²Ð½Ð¾Ð¹ ÐºÐ¾Ñ€Ñ€ÐµÐºÑ†Ð¸Ð¸ ÐµÑÐ»Ð¸ Ñ‚Ñ€ÐµÐ±ÑƒÐµÑ‚ÑÑ
    transformer = None
    if use_perspective:
        if perspective_method == 'hexagon' and HEXAGON_PERSPECTIVE:
            # ========== 6-Ð¢ÐžÐ§Ð•Ð§ÐÐÐ¯ Ð¢Ð ÐÐÐ¡Ð¤ÐžÐ ÐœÐÐ¦Ð˜Ð¯ ==========
            if isinstance(perspective_config, HexagonPerspectiveTransformer):
                transformer = perspective_config
                print("✓ Используется переданный HexagonPerspectiveTransformer")
            elif perspective_config and isinstance(perspective_config, str) and os.path.exists(perspective_config):
                transformer = HexagonPerspectiveTransformer.load_config(
                    perspective_config)
                print(
                    f"✓ Загружена 6-точечная конфигурация: {perspective_config}")
            else:
                # ÐÐ²Ñ‚Ð¾-Ð½Ð°ÑÑ‚Ñ€Ð¾Ð¹ÐºÐ° Ñ‡ÐµÑ€ÐµÐ· GUI
                video_name = Path(frames_dir).name
                first_frame = next(Path(frames_dir).glob("*.jpg"), None)
                if first_frame:
                    print(
                        f"Настройка 6-точечной перспективы для {video_name}...")
                    img = cv2.imread(str(first_frame))
                    if img is not None:
                        points = setup_hexagon_perspective_gui(img)
                        if points:
                            transformer = HexagonPerspectiveTransformer(
                                points,
                                dst_width=1920,
                                dst_height=720
                            )
                            # Ð¡Ð¾Ñ…Ñ€Ð°Ð½ÐµÐ½Ð¸Ðµ ÐºÐ¾Ð½Ñ„Ð¸Ð³Ð°
                            config_dir = os.path.join(
                                output_dir, "perspective_configs")
                            os.makedirs(config_dir, exist_ok=True)
                            config_path = os.path.join(
                                config_dir,
                                f"{video_name}_hexagon_perspective.json"
                            )
                            transformer.save_config(config_path)
                            print(
                                f"✓ 6-точечная конфигурация сохранена: {config_path}")
                        else:
                            print("⚠ Настройка 6-точечной перспективы отменена")

        elif perspective_method == 'standard' or not HEXAGON_PERSPECTIVE:
            # ========== Ð¡Ð¢ÐÐÐ”ÐÐ Ð¢ÐÐÐ¯ 4-Ð¢ÐžÐ§Ð•Ð§ÐÐÐ¯ Ð¢Ð ÐÐÐ¡Ð¤ÐžÐ ÐœÐÐ¦Ð˜Ð¯ ==========
            if perspective_method == 'hexagon' and not HEXAGON_PERSPECTIVE:
                print(
                    "⚠ HexagonPerspectiveTransformer недоступен, используется стандартный метод")

            if PerspectiveTransformer is not None:
                if isinstance(perspective_config, PerspectiveTransformer):
                    transformer = perspective_config
                    print("✓ Используется переданный PerspectiveTransformer")
                elif perspective_config and isinstance(perspective_config, str) and os.path.exists(perspective_config):
                    transformer = PerspectiveTransformer.load_config(
                        perspective_config)
                    print(
                        f"✓ Загружена 4-точечная конфигурация: {perspective_config}")
                else:
                    # ÐŸÐ¾Ð¿Ñ‹Ñ‚ÐºÐ° Ð°Ð²Ñ‚Ð¾Ð¼Ð°Ñ‚Ð¸Ñ‡ÐµÑÐºÐ¾Ð¹ Ð·Ð°Ð³Ñ€ÑƒÐ·ÐºÐ¸ Ð¸Ð»Ð¸ Ð½Ð°ÑÑ‚Ñ€Ð¾Ð¹ÐºÐ¸
                    video_name = Path(frames_dir).name
                    first_frame = next(Path(frames_dir).glob("*.jpg"), None)
                    if first_frame and load_or_setup_perspective:
                        transformer = load_or_setup_perspective(
                            video_name, str(first_frame),
                            config_dir=os.path.join(
                                output_dir, "perspective_configs")
                        )

        # ÐžÐ±Ñ‰Ð°Ñ Ð¾Ð±Ñ€Ð°Ð±Ð¾Ñ‚ÐºÐ° Ð¿Ð¾ÑÐ»Ðµ Ð½Ð°ÑÑ‚Ñ€Ð¾Ð¹ÐºÐ¸ Ñ‚Ñ€Ð°Ð½ÑÑ„Ð¾Ñ€Ð¼ÐµÑ€Ð°
        if transformer:
            print(
                f"✓ Перспективная коррекция активирована ({perspective_method})")
            # Ð•ÑÐ»Ð¸ Ð¸ÑÐ¿Ð¾Ð»ÑŒÐ·ÑƒÐµÑ‚ÑÑ Ð¿ÐµÑ€ÑÐ¿ÐµÐºÑ‚Ð¸Ð²Ð°, Ð½ÑƒÐ¶Ð½Ð¾ Ð¿Ñ€ÐµÐ¾Ð±Ñ€Ð°Ð·Ð¾Ð²Ð°Ñ‚ÑŒ Ð¿Ð¾Ð»Ð¸Ð³Ð¾Ð½
            if polygon is not None:
                try:
                    polygon = transformer.transform_polygon(polygon)
                    print("✓ Полигон адаптирован к скорректированной перспективе")
                except Exception as e:
                    print(f"⚠ Ошибка при трансформации полигона: {e}")
                    print("  Полигон будет использован без трансформации")
        else:
            print("ℹ Перспективная коррекция не настроена")

    # Создание анализатора
        if USE_IMPROVED_ANALYZER and IMPROVED_AVAILABLE:
            AnalyzerClass = ImprovedShikhtaAnalyzer
            print("Используется улучшенный анализатор")
            extra_params = {
                'min_contour_area': min_contour_area,
                'near_zone_ratio': near_zone_ratio
            }
    else:
        AnalyzerClass = ShikhtaAnalyzer
        print("Используется стандартный анализатор")
        extra_params = {}
    
    # Создание экземпляра
    if transformer:
        analyzer = AnalyzerClass(
            polygon=polygon,
            target_size=(transformer.dst_width, transformer.dst_height),
            perspective_transformer=transformer,
            **extra_params
        )
    else:
        analyzer = AnalyzerClass(
            polygon=polygon,
            **extra_params
        )

    # Ð¡Ð¾Ð·Ð´Ð°Ð½Ð¸Ðµ Ð¿Ð¾Ð´Ð´Ð¸Ñ€ÐµÐºÑ‚Ð¾Ñ€Ð¸Ð¹
    vis_dir = os.path.join(
        output_dir, "visualizations") if save_visualizations else None

    # ÐžÐ±Ñ€Ð°Ð±Ð¾Ñ‚ÐºÐ° ÐºÐ°Ð´Ñ€Ð¾Ð²
    metrics = analyzer.process_video_frames(
        frames_dir,
        output_dir=vis_dir,
        save_every_n=save_every_n,
        use_parallel=use_parallel,
        max_workers=max_workers
    )

    if not metrics:
        print("Нет метрик для сохранения")
        return None

    # Ð¡Ð¾Ñ…Ñ€Ð°Ð½ÐµÐ½Ð¸Ðµ Ð¼ÐµÑ‚Ñ€Ð¸Ðº
    video_name = Path(frames_dir).name
    metrics_path = os.path.join(output_dir, f"{video_name}_metrics.json")
    summary = analyzer.save_metrics(metrics_path)

    elapsed_time = time.time() - start_time

    # Ð’Ñ‹Ð²Ð¾Ð´ ÑÑ‚Ð°Ñ‚Ð¸ÑÑ‚Ð¸ÐºÐ¸
    print("\n" + "="*60)
    print(f"СТАТИСТИКА ПО ШИХТЕ: {video_name}")
    if transformer:
        method_name = "6-точечная" if perspective_method == 'hexagon' else "4-точечная"
        print(f"( перспективной коррекцией: {method_name})")
    print("="*60)
    print(f"Всего кадров: {summary['total_frames']}")
    print(
        f"Время обработки: {elapsed_time:.1f} сек ({summary['total_frames']/elapsed_time:.1f} кадров/сек)")
    print("\nЛевая часть:")
    print(f"  Среднее: {summary['left']['mean']:.2f}%")
    print(
        f"  Мин/Макс: {summary['left']['min']:.2f}% / {summary['left']['max']:.2f}%")
    print(f"  Медиана: {summary['left']['median']:.2f}%")
    print("\nПравая часть::")
    print(f"  Среднее: {summary['right']['mean']:.2f}%")
    print(
        f"  Мин/Макс: {summary['right']['min']:.2f}% / {summary['right']['max']:.2f}%")
    print(f"  Медиана: {summary['right']['median']:.2f}%")
    print("="*70 + "\n")

    return summary


# Ð”ÐµÑ„Ð¾Ð»Ñ‚Ð½Ñ‹Ð¹ Ð¿Ð¾Ð»Ð¸Ð³Ð¾Ð½
DEFAULT_POLYGON = np.array([
    [215, 113],
    [625, 118],
    [733, 270],
    [577, 529],
    [144, 530],
    [54, 277]
], np.int32)


class ShikhtaAnalyzer:
    """Анализатор шихты с поддержкой перспективной коррекции"""

    def __init__(self, polygon=None, target_size=(928, 576), perspective_transformer=None):
        self.polygon = polygon if polygon is not None else DEFAULT_POLYGON.copy()
        if not isinstance(self.polygon, np.ndarray):
            try:
                self.polygon = np.array(self.polygon, dtype=np.int32)
            except Exception:
                self.polygon = np.array(DEFAULT_POLYGON, dtype=np.int32)
        else:
            self.polygon = self.polygon.astype(np.int32)

        self.target_size = target_size
        self.perspective_transformer = perspective_transformer
        self.frame_metrics = []

        # ÐŸÑ€ÐµÐ´Ð²Ñ‹Ñ‡Ð¸ÑÐ»ÐµÐ½Ð½Ñ‹Ðµ Ð¼Ð°ÑÐºÐ¸
        self._polygon_mask = None
        self._left_mask = None
        self._right_mask = None
        self._mid_x = None
        self._setup_masks()

    def _setup_masks(self):
        """ÐŸÑ€ÐµÐ´Ð²Ð°Ñ€Ð¸Ñ‚ÐµÐ»ÑŒÐ½Ð¾Ðµ ÑÐ¾Ð·Ð´Ð°Ð½Ð¸Ðµ Ð¼Ð°ÑÐ¾Ðº"""
        dummy = np.zeros(self.target_size[::-1], dtype=np.uint8)

        # ÐœÐ°ÑÐºÐ° Ð¿Ð¾Ð»Ð¸Ð³Ð¾Ð½Ð°
        self._polygon_mask = np.zeros_like(dummy, dtype=np.uint8)
        cv2.fillPoly(self._polygon_mask, [self.polygon], 255)

        # Ð Ð°Ð·Ð´ÐµÐ»ÐµÐ½Ð¸Ðµ Ð½Ð° Ð»ÐµÐ²ÑƒÑŽ Ð¸ Ð¿Ñ€Ð°Ð²ÑƒÑŽ Ñ‡Ð°ÑÑ‚Ð¸
        self._mid_x = (self.polygon[:, 0].min() +
                       self.polygon[:, 0].max()) // 2

        left_polygon = []
        right_polygon = []
        for pt in self.polygon:
            if pt[0] < self._mid_x:
                left_polygon.append(pt)
            else:
                right_polygon.append(pt)

        left_polygon = np.array(left_polygon, np.int32)
        right_polygon = np.array(right_polygon, np.int32)

        self._left_mask = np.zeros_like(dummy, dtype=np.uint8)
        self._right_mask = np.zeros_like(dummy, dtype=np.uint8)
        cv2.fillPoly(self._left_mask, [left_polygon], 255)
        cv2.fillPoly(self._right_mask, [right_polygon], 255)

    def preprocess_frame(self, frame):
        """ÐŸÑ€ÐµÐ´Ð¾Ð±Ñ€Ð°Ð±Ð¾Ñ‚ÐºÐ° Ð¸Ð·Ð¾Ð±Ñ€Ð°Ð¶ÐµÐ½Ð¸Ñ Ñ Ð¾Ð¿Ñ†Ð¸Ð¾Ð½Ð°Ð»ÑŒÐ½Ð¾Ð¹ Ð¿ÐµÑ€ÑÐ¿ÐµÐºÑ‚Ð¸Ð²Ð½Ð¾Ð¹ ÐºÐ¾Ñ€Ñ€ÐµÐºÑ†Ð¸ÐµÐ¹"""
        # ÐŸÑ€Ð¸Ð¼ÐµÐ½ÑÐµÐ¼ Ð¿ÐµÑ€ÑÐ¿ÐµÐºÑ‚Ð¸Ð²Ð½ÑƒÑŽ ÐºÐ¾Ñ€Ñ€ÐµÐºÑ†Ð¸ÑŽ ÐµÑÐ»Ð¸ ÐµÑÑ‚ÑŒ Ñ‚Ñ€Ð°Ð½ÑÑ„Ð¾Ñ€Ð¼ÐµÑ€
        if self.perspective_transformer is not None:
            frame = self.perspective_transformer.transform(frame)

        # ÐšÐ¾Ð½Ð²ÐµÑ€Ñ‚Ð°Ñ†Ð¸Ñ Ð² grayscale ÐµÑÐ»Ð¸ Ð½ÑƒÐ¶Ð½Ð¾
        if len(frame.shape) == 3:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            img = frame

        # Resize Ñ‚Ð¾Ð»ÑŒÐºÐ¾ ÐµÑÐ»Ð¸ Ñ€Ð°Ð·Ð¼ÐµÑ€ Ð¾Ñ‚Ð»Ð¸Ñ‡Ð°ÐµÑ‚ÑÑ
        if img.shape[:2] != self.target_size[::-1]:
            img = cv2.resize(img, self.target_size)

        # ÐÐºÐ²Ð°Ð»Ð¸Ð·Ð°Ñ†Ð¸Ñ Ð³Ð¸ÑÑ‚Ð¾Ð³Ñ€Ð°Ð¼Ð¼Ñ‹
        img_hist_eq = cv2.equalizeHist(img)
        img_float = img_as_float64(img_hist_eq)

        # Ð¤Ð¸Ð»ÑŒÑ‚Ñ€ Ð’Ð¸Ð½ÐµÑ€Ð°
        kernel = np.ones((15, 15), np.float64)
        image_filtered = scipy.signal.convolve2d(img_float, kernel, 'same')
        img_wiener = restoration.wiener(image_filtered, kernel, 5.1e4)
        img_wiener = img_as_ubyte(img_wiener)

        # CLAHE
        image_filtered = img_as_ubyte(img_wiener)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        image_filtered = clahe.apply(image_filtered)

        return img, image_filtered

    def segment_shikhta(self, preprocessed_img, polygon_mask):
        """Ð¡ÐµÐ³Ð¼ÐµÐ½Ñ‚Ð°Ñ†Ð¸Ñ Ð¿ÑÑ‚ÐµÐ½ ÑˆÐ¸Ñ…Ñ‚Ñ‹"""
        # ÐÐ´Ð°Ð¿Ñ‚Ð¸Ð²Ð½Ð°Ñ Ð±Ð¸Ð½Ð°Ñ€Ð¸Ð·Ð°Ñ†Ð¸Ñ
        thresh = cv2.adaptiveThreshold(
            preprocessed_img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 191, 0
        )

        # ÐœÐ¾Ñ€Ñ„Ð¾Ð»Ð¾Ð³Ð¸Ñ‡ÐµÑÐºÐ¾Ðµ Ð·Ð°ÐºÑ€Ñ‹Ñ‚Ð¸Ðµ
        thresh = cv2.morphologyEx(
            thresh, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        # ÐŸÑ€Ð¸Ð¼ÐµÐ½ÐµÐ½Ð¸Ðµ Ð¼Ð°ÑÐºÐ¸ Ð¿Ð¾Ð»Ð¸Ð³Ð¾Ð½Ð°
        thresh = cv2.bitwise_and(thresh, polygon_mask)

        return thresh

    def analyze_frame(self, frame, frame_idx=0, save_visualization=False, output_path=None):
        """ÐÐ½Ð°Ð»Ð¸Ð· Ð¾Ð´Ð½Ð¾Ð³Ð¾ ÐºÐ°Ð´Ñ€Ð° Ñ Ð¿ÐµÑ€ÑÐ¿ÐµÐºÑ‚Ð¸Ð²Ð½Ð¾Ð¹ ÐºÐ¾Ñ€Ñ€ÐµÐºÑ†Ð¸ÐµÐ¹"""
        # Ð¡Ð¾Ñ…Ñ€Ð°Ð½ÑÐµÐ¼ Ð¾Ñ€Ð¸Ð³Ð¸Ð½Ð°Ð»ÑŒÐ½Ñ‹Ð¹ ÐºÐ°Ð´Ñ€ Ð´Ð»Ñ Ð²Ð¸Ð·ÑƒÐ°Ð»Ð¸Ð·Ð°Ñ†Ð¸Ð¸
        original_frame = frame.copy()

        # ÐŸÑ€ÐµÐ´Ð¾Ð±Ñ€Ð°Ð±Ð¾Ñ‚ÐºÐ° (Ð²ÐºÐ»ÑŽÑ‡Ð°ÐµÑ‚ Ð¿ÐµÑ€ÑÐ¿ÐµÐºÑ‚Ð¸Ð²Ð½ÑƒÑŽ ÐºÐ¾Ñ€Ñ€ÐµÐºÑ†Ð¸ÑŽ ÐµÑÐ»Ð¸ Ð²ÐºÐ»ÑŽÑ‡ÐµÐ½Ð°)
        img_gray, preprocessed = self.preprocess_frame(frame)

        # Ð¡ÐµÐ³Ð¼ÐµÐ½Ñ‚Ð°Ñ†Ð¸Ñ
        thresh = self.segment_shikhta(preprocessed, self._polygon_mask)

        # ÐŸÐ¾Ð¸ÑÐº ÐºÐ¾Ð½Ñ‚ÑƒÑ€Ð¾Ð²
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Ð Ð°Ð·Ð´ÐµÐ»ÐµÐ½Ð¸Ðµ ÐºÐ¾Ð½Ñ‚ÑƒÑ€Ð¾Ð²
        left_thresh = cv2.bitwise_and(thresh, self._left_mask)
        right_thresh = cv2.bitwise_and(thresh, self._right_mask)

        left_contours, _ = cv2.findContours(
            left_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        right_contours, _ = cv2.findContours(
            right_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Ð Ð°ÑÑ‡ÐµÑ‚ Ð¿Ð»Ð¾Ñ‰Ð°Ð´ÐµÐ¹
        left_area = sum(cv2.contourArea(cnt) for cnt in left_contours)
        right_area = sum(cv2.contourArea(cnt) for cnt in right_contours)
        total_area = left_area + right_area

        # ÐŸÑ€Ð¾Ñ†ÐµÐ½Ñ‚Ð½Ð¾Ðµ ÑÐ¾Ð¾Ñ‚Ð½Ð¾ÑˆÐµÐ½Ð¸Ðµ
        if total_area > 0:
            left_percent = (left_area / total_area) * 100
            right_percent = (right_area / total_area) * 100
        else:
            left_percent = right_percent = 0

        metrics = {
            'frame_idx': frame_idx,
            'left_area': float(left_area),
            'right_area': float(right_area),
            'total_area': float(total_area),
            'left_percent': float(left_percent),
            'right_percent': float(right_percent),
            'left_contours_count': len(left_contours),
            'right_contours_count': len(right_contours),
            'perspective_corrected': self.perspective_transformer is not None
        }

        # Ð’Ð¸Ð·ÑƒÐ°Ð»Ð¸Ð·Ð°Ñ†Ð¸Ñ
        if save_visualization and output_path:
            # ÐŸÐ¾Ð´Ð³Ð¾Ñ‚Ð¾Ð²ÐºÐ° Ð¸Ð·Ð¾Ð±Ñ€Ð°Ð¶ÐµÐ½Ð¸Ñ Ð´Ð»Ñ Ð²Ð¸Ð·ÑƒÐ°Ð»Ð¸Ð·Ð°Ñ†Ð¸Ð¸
            if self.perspective_transformer:
                # Ð•ÑÐ»Ð¸ Ð¸ÑÐ¿Ð¾Ð»ÑŒÐ·Ð¾Ð²Ð°Ð»Ð°ÑÑŒ Ð¿ÐµÑ€ÑÐ¿ÐµÐºÑ‚Ð¸Ð²Ð°, Ð¿Ð¾ÐºÐ°Ð·Ñ‹Ð²Ð°ÐµÐ¼ ÑÐºÐ¾Ñ€Ñ€ÐµÐºÑ‚Ð¸Ñ€Ð¾Ð²Ð°Ð½Ð½Ð¾Ðµ Ð¸Ð·Ð¾Ð±Ñ€Ð°Ð¶ÐµÐ½Ð¸Ðµ
                img_color = self.perspective_transformer.transform(
                    original_frame)
            else:
                img_color = cv2.resize(original_frame.copy(), self.target_size)

            min_y = self.polygon[:, 1].min()
            max_y = self.polygon[:, 1].max()

            # Ð Ð¸ÑÑƒÐµÐ¼ ÐºÐ¾Ð½Ñ‚ÑƒÑ€Ñ‹ Ð¸ Ð³Ñ€Ð°Ð½Ð¸Ñ†Ñ‹
            cv2.drawContours(img_color, contours, -1, (0, 255, 0), 1)
            cv2.polylines(img_color, [self.polygon], True, (255, 0, 0), 2)
            cv2.line(img_color, (self._mid_x, min_y),
                     (self._mid_x, max_y), (0, 0, 255), 2)

            # Ð¢ÐµÐºÑÑ‚ Ñ Ð¼ÐµÑ‚Ñ€Ð¸ÐºÐ°Ð¼Ð¸
            text = f"L: {left_percent:.1f}% | R: {right_percent:.1f}%"
            if self.perspective_transformer:
                text += " [PERSP]"
            cv2.putText(img_color, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)

            # Ð¡Ð¾Ð·Ð´Ð°Ñ‘Ð¼ ÐºÐ¾Ð¼Ð¿Ð¾Ð·Ð¸Ñ‚Ð½Ð¾Ðµ Ð¸Ð·Ð¾Ð±Ñ€Ð°Ð¶ÐµÐ½Ð¸Ðµ ÐµÑÐ»Ð¸ Ð¸ÑÐ¿Ð¾Ð»ÑŒÐ·ÑƒÐµÑ‚ÑÑ Ð¿ÐµÑ€ÑÐ¿ÐµÐºÑ‚Ð¸Ð²Ð°
            if self.perspective_transformer:
                # Ð¡Ð¾Ð·Ð´Ð°Ñ‘Ð¼ ÑÑ€Ð°Ð²Ð½Ð¸Ñ‚ÐµÐ»ÑŒÐ½ÑƒÑŽ Ð²Ð¸Ð·ÑƒÐ°Ð»Ð¸Ð·Ð°Ñ†Ð¸ÑŽ
                h, w = img_color.shape[:2]
                composite = np.zeros((h, w * 2 + 10, 3), dtype=np.uint8)

                # ÐžÑ€Ð¸Ð³Ð¸Ð½Ð°Ð» ÑÐ»ÐµÐ²Ð°
                orig_resized = cv2.resize(original_frame, (w, h))
                composite[:, :w] = orig_resized

                # Ð Ð°Ð·Ð´ÐµÐ»Ð¸Ñ‚ÐµÐ»ÑŒ
                composite[:, w:w+10] = 128

                # Ð¡ÐºÐ¾Ñ€Ñ€ÐµÐºÑ‚Ð¸Ñ€Ð¾Ð²Ð°Ð½Ð½Ð¾Ðµ ÑÐ¿Ñ€Ð°Ð²Ð°
                composite[:, w+10:] = img_color

                # Ð¡Ð¾Ñ…Ñ€Ð°Ð½ÐµÐ½Ð¸Ðµ
                cv2.imwrite(output_path, composite)
            else:
                cv2.imwrite(output_path, img_color)

        return metrics

    def _process_single_frame_wrapper(self, task):
        """ÐžÐ±ÐµÑ€Ñ‚ÐºÐ° Ð´Ð»Ñ Ð¿Ð°Ñ€Ð°Ð»Ð»ÐµÐ»ÑŒÐ½Ð¾Ð¹ Ð¾Ð±Ñ€Ð°Ð±Ð¾Ñ‚ÐºÐ¸"""
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
        """ÐžÐ±Ñ€Ð°Ð±Ð¾Ñ‚ÐºÐ° Ð²ÑÐµÑ… ÐºÐ°Ð´Ñ€Ð¾Ð² Ð²Ð¸Ð´ÐµÐ¾ Ñ Ð¿Ð¾Ð´Ð´ÐµÑ€Ð¶ÐºÐ¾Ð¹ Ð¿Ð°Ñ€Ð°Ð»Ð»ÐµÐ»Ð¸Ð·Ð¼Ð°"""
        frame_files = sorted(Path(frames_dir).glob("*.jpg"))
        if max_frames:
            frame_files = frame_files[:max_frames]

        if not frame_files:
            print(f"Нет кадров в {frames_dir}")
            return []

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self.frame_metrics = []

        print(
            f"Обработка {len(frame_files)} кадров из {Path(frames_dir).name}...")
        if self.perspective_transformer:
            print("  • Перспективная коррекция: включена")

        if use_parallel and len(frame_files) > 50:
            # ÐŸÐ°Ñ€Ð°Ð»Ð»ÐµÐ»ÑŒÐ½Ð°Ñ Ð¾Ð±Ñ€Ð°Ð±Ð¾Ñ‚ÐºÐ°
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
                        print(
                            f"    Обработано {processed}/{len(frame_files)} кадров")

            # Ð¡Ð¾Ñ€Ñ‚Ð¸Ñ€Ð¾Ð²ÐºÐ° Ð¿Ð¾ frame_idx
            self.frame_metrics.sort(key=lambda x: x['frame_idx'])
        else:
            # ÐŸÐ¾ÑÐ»ÐµÐ´Ð¾Ð²Ð°Ñ‚ÐµÐ»ÑŒÐ½Ð°Ñ Ð¾Ð±Ñ€Ð°Ð±Ð¾Ñ‚ÐºÐ°
            print(f"  • Режим: последовательный")
            for idx, frame_path in enumerate(frame_files):
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    continue

                save_vis = output_dir and (idx % save_every_n == 0)
                vis_path = None
                if save_vis:
                    vis_path = os.path.join(
                        output_dir, f"shikhta_{idx:06d}.jpg")

                metrics = self.analyze_frame(frame, frame_idx=idx,
                                             save_visualization=save_vis,
                                             output_path=vis_path)
                self.frame_metrics.append(metrics)

                if (idx + 1) % 100 == 0:
                    print(
                        f"    Обработано {idx + 1}/{len(frame_files)} кадров")

        print(f"✓ Обработано {len(self.frame_metrics)} кадров")
        return self.frame_metrics

    def compute_summary_statistics(self):
        """Ð’Ñ‹Ñ‡Ð¸ÑÐ»ÐµÐ½Ð¸Ðµ ÑÐ²Ð¾Ð´Ð½Ð¾Ð¹ ÑÑ‚Ð°Ñ‚Ð¸ÑÑ‚Ð¸ÐºÐ¸"""
        if not self.frame_metrics:
            return None

        left_percents = [m['left_percent'] for m in self.frame_metrics]
        right_percents = [m['right_percent'] for m in self.frame_metrics]
        total_areas = [m['total_area'] for m in self.frame_metrics]

        summary = {
            'total_frames': len(self.frame_metrics),
            'perspective_corrected': self.perspective_transformer is not None,
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
        """Ð¡Ð¾Ñ…Ñ€Ð°Ð½ÐµÐ½Ð¸Ðµ Ð¼ÐµÑ‚Ñ€Ð¸Ðº Ð² JSON Ñ Ð°Ð²Ñ‚Ð¾Ð¼Ð°Ñ‚Ð¸Ñ‡ÐµÑÐºÐ¾Ð¹ Ð³ÐµÐ½ÐµÑ€Ð°Ñ†Ð¸ÐµÐ¹ Ð³Ñ€Ð°Ñ„Ð¸ÐºÐ¾Ð²"""
        summary = self.compute_summary_statistics()

        data = {
            'summary': summary,
            'frames': self.frame_metrics
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✓ Метрики сохранены в {output_path}")

        # ÐÐ²Ñ‚Ð¾Ð¼Ð°Ñ‚Ð¸Ñ‡ÐµÑÐºÐ°Ñ Ð³ÐµÐ½ÐµÑ€Ð°Ñ†Ð¸Ñ Ð³Ñ€Ð°Ñ„Ð¸ÐºÐ¾Ð²
        plot_path = output_path.replace('.json', '_plot.png')
        self.plot_metrics(plot_path)

        return summary

    def plot_metrics(self, output_path):
        """ÐŸÐ¾ÑÑ‚Ñ€Ð¾ÐµÐ½Ð¸Ðµ Ð³Ñ€Ð°Ñ„Ð¸ÐºÐ¾Ð² Ñ Ð²Ñ‹Ð´ÐµÐ»ÐµÐ½Ð¸ÐµÐ¼ Ð¼Ð¸Ð½/Ð¼Ð°ÐºÑ"""
        if not self.frame_metrics:
            print("Нет данных для построения графиков")
            return

        # Ð˜Ð·Ð²Ð»ÐµÑ‡ÐµÐ½Ð¸Ðµ Ð´Ð°Ð½Ð½Ñ‹Ñ…
        frames = [m['frame_idx'] for m in self.frame_metrics]
        left_percents = [m['left_percent'] for m in self.frame_metrics]
        right_percents = [m['right_percent'] for m in self.frame_metrics]
        total_areas = [m['total_area'] for m in self.frame_metrics]

        # ÐŸÐ¾Ð¸ÑÐº ÑÐºÑÑ‚Ñ€ÐµÐ¼ÑƒÐ¼Ð¾Ð²
        left_min_idx = np.argmin(left_percents)
        left_max_idx = np.argmax(left_percents)
        right_min_idx = np.argmin(right_percents)
        right_max_idx = np.argmax(right_percents)

        # Ð¡Ð¾Ð·Ð´Ð°Ð½Ð¸Ðµ Ð³Ñ€Ð°Ñ„Ð¸ÐºÐ¾Ð²
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Ð”Ð¾Ð±Ð°Ð²Ð»ÐµÐ½Ð¸Ðµ Ð·Ð°Ð³Ð¾Ð»Ð¾Ð²ÐºÐ° ÐµÑÐ»Ð¸ Ð¸ÑÐ¿Ð¾Ð»ÑŒÐ·Ð¾Ð²Ð°Ð»Ð°ÑÑŒ Ð¿ÐµÑ€ÑÐ¿ÐµÐºÑ‚Ð¸Ð²Ð½Ð°Ñ ÐºÐ¾Ñ€Ñ€ÐµÐºÑ†Ð¸Ñ
        if self.perspective_transformer:
            fig.suptitle('Анализ шихты с перспективной коррекцией',
                         fontsize=16, fontweight='bold', y=1.02)

        # Ð“Ñ€Ð°Ñ„Ð¸Ðº 1: ÐŸÑ€Ð¾Ñ†ÐµÐ½Ñ‚Ð½Ð¾Ðµ ÑÐ¾Ð¾Ñ‚Ð½Ð¾ÑˆÐµÐ½Ð¸Ðµ
        ax1 = axes[0]
        ax1.plot(frames, left_percents, label='Левая часть',
                 color='#2E86DE', linewidth=2, alpha=0.8)
        ax1.plot(frames, right_percents, label='Правая часть',
                 color='#EE5A6F', linewidth=2, alpha=0.8)

        # Ð›Ð¸Ð½Ð¸Ñ Ð±Ð°Ð»Ð°Ð½ÑÐ° 50%
        ax1.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5,
                    label='Баланс 50/50')

        # Ð’Ñ‹Ð´ÐµÐ»ÐµÐ½Ð¸Ðµ Ð¼Ð¸Ð½Ð¸Ð¼ÑƒÐ¼Ð¾Ð² Ð¸ Ð¼Ð°ÐºÑÐ¸Ð¼ÑƒÐ¼Ð¾Ð²
        # Ð›ÐµÐ²Ð°Ñ Ñ‡Ð°ÑÑ‚ÑŒ
        ax1.scatter(frames[left_min_idx], left_percents[left_min_idx],
                    color='blue', s=150, marker='v', zorder=5,
                    edgecolors='black', linewidth=2)
        ax1.annotate(f'MIN: {left_percents[left_min_idx]:.1f}%',
                     xy=(frames[left_min_idx], left_percents[left_min_idx]),
                     xytext=(10, -20), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5',
                               facecolor='lightblue', alpha=0.8),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                     color='blue', lw=2))

        ax1.scatter(frames[left_max_idx], left_percents[left_max_idx],
                    color='blue', s=150, marker='^', zorder=5,
                    edgecolors='black', linewidth=2)
        ax1.annotate(f'MAX: {left_percents[left_max_idx]:.1f}%',
                     xy=(frames[left_max_idx], left_percents[left_max_idx]),
                     xytext=(10, 20), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5',
                               facecolor='lightblue', alpha=0.8),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                     color='blue', lw=2))

        # ÐŸÑ€Ð°Ð²Ð°Ñ Ñ‡Ð°ÑÑ‚ÑŒ
        ax1.scatter(frames[right_min_idx], right_percents[right_min_idx],
                    color='red', s=150, marker='v', zorder=5,
                    edgecolors='black', linewidth=2)
        ax1.annotate(f'MIN: {right_percents[right_min_idx]:.1f}%',
                     xy=(frames[right_min_idx], right_percents[right_min_idx]),
                     xytext=(10, -20), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5',
                               facecolor='lightcoral', alpha=0.8),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                     color='red', lw=2))

        ax1.scatter(frames[right_max_idx], right_percents[right_max_idx],
                    color='red', s=150, marker='^', zorder=5,
                    edgecolors='black', linewidth=2)
        ax1.annotate(f'MAX: {right_percents[right_max_idx]:.1f}%',
                     xy=(frames[right_max_idx], right_percents[right_max_idx]),
                     xytext=(10, 20), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5',
                               facecolor='lightcoral', alpha=0.8),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                     color='red', lw=2))

        ax1.set_xlabel('Номер кадра', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Процент шихты, %', fontsize=12, fontweight='bold')
        title = 'Распределение шихты по левой и правой частям'
        if self.perspective_transformer:
            title += ' (скорректированная перспектива)'
        ax1.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax1.legend(loc='best', fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        ax1.set_ylim(0, 100)

        # Добавление статистики в углу
        stats_text = (f'Левая: μ={np.mean(left_percents):.1f}% σ={np.std(left_percents):.1f}%\n'
                      f'Правая: μ={np.mean(right_percents):.1f}% σ={np.std(right_percents):.1f}%')
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        # Ð“Ñ€Ð°Ñ„Ð¸Ðº 2: ÐžÐ±Ñ‰Ð°Ñ Ð¿Ð»Ð¾Ñ‰Ð°Ð´ÑŒ ÑˆÐ¸Ñ…Ñ‚Ñ‹
        ax2 = axes[1]
        ax2.fill_between(frames, total_areas, alpha=0.3, color='#10AC84')
        ax2.plot(frames, total_areas, color='#10AC84', linewidth=2)

        # Ð’Ñ‹Ð´ÐµÐ»ÐµÐ½Ð¸Ðµ ÑÐºÑÑ‚Ñ€ÐµÐ¼ÑƒÐ¼Ð¾Ð² Ð¿Ð»Ð¾Ñ‰Ð°Ð´Ð¸
        area_min_idx = np.argmin(total_areas)
        area_max_idx = np.argmax(total_areas)

        ax2.scatter(frames[area_min_idx], total_areas[area_min_idx],
                    color='green', s=150, marker='v', zorder=5,
                    edgecolors='black', linewidth=2)
        ax2.annotate(f'MIN: {total_areas[area_min_idx]:.0f}',
                     xy=(frames[area_min_idx], total_areas[area_min_idx]),
                     xytext=(10, -20), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5',
                               facecolor='lightgreen', alpha=0.8),
                     arrowprops=dict(arrowstyle='->', color='green', lw=2))

        ax2.scatter(frames[area_max_idx], total_areas[area_max_idx],
                    color='green', s=150, marker='^', zorder=5,
                    edgecolors='black', linewidth=2)
        ax2.annotate(f'MAX: {total_areas[area_max_idx]:.0f}',
                     xy=(frames[area_max_idx], total_areas[area_max_idx]),
                     xytext=(10, 20), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5',
                               facecolor='lightgreen', alpha=0.8),
                     arrowprops=dict(arrowstyle='->', color='green', lw=2))

        ax2.set_xlabel('Номер кадра', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Площадь, пикс²', fontsize=12, fontweight='bold')
        ax2.set_title('Общая площадь шихты во времени',
                      fontsize=14, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

        # Ð¡Ñ‚Ð°Ñ‚Ð¸ÑÑ‚Ð¸ÐºÐ° Ð¿Ð»Ð¾Ñ‰Ð°Ð´Ð¸
        area_stats = f'Î¼={np.mean(total_areas):.0f} Ïƒ={np.std(total_areas):.0f}'
        ax2.text(0.02, 0.98, area_stats, transform=ax2.transAxes,
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ График сохранен: {output_path}")
        plt.close()


if __name__ == "__main__":
    # ÐŸÑ€Ð¸Ð¼ÐµÑ€ Ð¸ÑÐ¿Ð¾Ð»ÑŒÐ·Ð¾Ð²Ð°Ð½Ð¸Ñ Ñ Ð¿ÐµÑ€ÑÐ¿ÐµÐºÑ‚Ð¸Ð²Ð½Ð¾Ð¹ ÐºÐ¾Ñ€Ñ€ÐµÐºÑ†Ð¸ÐµÐ¹
    import sys
    if len(sys.argv) < 3:
        print(
            "Usage: python shikhta_analysis_with_perspective.py <frames_dir> <output_dir> [--no-perspective]")
        sys.exit(1)

    use_perspective = "--no-perspective" not in sys.argv

    analyze_video_shikhta(
        sys.argv[1],
        sys.argv[2],
        use_perspective=use_perspective
    )
