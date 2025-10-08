# shikhta_analysis.py
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

# Импорт модуля перспективной коррекции
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
                          perspective_method='hexagon'):
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

    # Настройка перспективной коррекции если требуется
    transformer = None
    if use_perspective:
        if perspective_method == 'hexagon' and HEXAGON_PERSPECTIVE:
            # ========== 6-ТОЧЕЧНАЯ ТРАНСФОРМАЦИЯ ==========
            if isinstance(perspective_config, HexagonPerspectiveTransformer):
                transformer = perspective_config
                print("✓ Используется переданный HexagonPerspectiveTransformer")
            elif perspective_config and isinstance(perspective_config, str) and os.path.exists(perspective_config):
                transformer = HexagonPerspectiveTransformer.load_config(
                    perspective_config)
                print(
                    f"✓ Загружена 6-точечная конфигурация: {perspective_config}")
            else:
                # Авто-настройка через GUI
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
                                dst_width=800,
                                dst_height=600
                            )
                            # Сохранение конфига
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
            # ========== СТАНДАРТНАЯ 4-ТОЧЕЧНАЯ ТРАНСФОРМАЦИЯ ==========
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
                    # Попытка автоматической загрузки или настройки
                    video_name = Path(frames_dir).name
                    first_frame = next(Path(frames_dir).glob("*.jpg"), None)
                    if first_frame and load_or_setup_perspective:
                        transformer = load_or_setup_perspective(
                            video_name, str(first_frame),
                            config_dir=os.path.join(
                                output_dir, "perspective_configs")
                        )

        # Общая обработка после настройки трансформера
        if transformer:
            print(
                f"✓ Перспективная коррекция активирована ({perspective_method})")
            # Если используется перспектива, нужно преобразовать полигон
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
    if transformer:
        # При использовании перспективы меняем target_size на размер выхода трансформера
        analyzer = ShikhtaAnalyzer(
            polygon=polygon,
            target_size=(transformer.dst_width, transformer.dst_height),
            perspective_transformer=transformer
        )
    else:
        analyzer = ShikhtaAnalyzer(polygon=polygon)

    # Создание поддиректорий
    vis_dir = os.path.join(
        output_dir, "visualizations") if save_visualizations else None

    # Обработка кадров
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

    # Сохранение метрик
    video_name = Path(frames_dir).name
    metrics_path = os.path.join(output_dir, f"{video_name}_metrics.json")
    summary = analyzer.save_metrics(metrics_path)

    elapsed_time = time.time() - start_time

    # Вывод статистики
    print("\n" + "="*60)
    print(f"СТАТИСТИКА ПО ШИХТЕ: {video_name}")
    if transformer:
        method_name = "6-точечная" if perspective_method == 'hexagon' else "4-точечная"
        print(f"(с перспективной коррекцией: {method_name})")
    print("="*60)
    print(f"Всего кадров: {summary['total_frames']}")
    print(
        f"Время обработки: {elapsed_time:.1f} сек ({summary['total_frames']/elapsed_time:.1f} кадров/сек)")
    print("\nЛевая часть:")
    print(f"  Среднее: {summary['left']['mean']:.2f}%")
    print(
        f"  Мин/Макс: {summary['left']['min']:.2f}% / {summary['left']['max']:.2f}%")
    print(f"  Медиана: {summary['left']['median']:.2f}%")
    print("\nПравая часть:")
    print(f"  Среднее: {summary['right']['mean']:.2f}%")
    print(
        f"  Мин/Макс: {summary['right']['min']:.2f}% / {summary['right']['max']:.2f}%")
    print(f"  Медиана: {summary['right']['median']:.2f}%")
    print("="*70 + "\n")

    return summary


# Дефолтный полигон
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

        # Предвычисленные маски
        self._polygon_mask = None
        self._left_mask = None
        self._right_mask = None
        self._mid_x = None
        self._setup_masks()

    def _setup_masks(self):
        """Предварительное создание масок"""
        dummy = np.zeros(self.target_size[::-1], dtype=np.uint8)

        # Маска полигона
        self._polygon_mask = np.zeros_like(dummy, dtype=np.uint8)
        cv2.fillPoly(self._polygon_mask, [self.polygon], 255)

        # Разделение на левую и правую части
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
        """Предобработка изображения с опциональной перспективной коррекцией"""
        # Применяем перспективную коррекцию если есть трансформер
        if self.perspective_transformer is not None:
            frame = self.perspective_transformer.transform(frame)

        # Конвертация в grayscale если нужно
        if len(frame.shape) == 3:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            img = frame

        # Resize только если размер отличается
        if img.shape[:2] != self.target_size[::-1]:
            img = cv2.resize(img, self.target_size)

        # Эквализация гистограммы
        img_hist_eq = cv2.equalizeHist(img)
        img_float = img_as_float64(img_hist_eq)

        # Фильтр Винера
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
        """Сегментация пятен шихты"""
        # Адаптивная бинаризация
        thresh = cv2.adaptiveThreshold(
            preprocessed_img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 191, 0
        )

        # Морфологическое закрытие
        thresh = cv2.morphologyEx(
            thresh, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        # Применение маски полигона
        thresh = cv2.bitwise_and(thresh, polygon_mask)

        return thresh

    def analyze_frame(self, frame, frame_idx=0, save_visualization=False, output_path=None):
        """Анализ одного кадра с перспективной коррекцией"""
        # Сохраняем оригинальный кадр для визуализации
        original_frame = frame.copy()

        # Предобработка (включает перспективную коррекцию если включена)
        img_gray, preprocessed = self.preprocess_frame(frame)

        # Сегментация
        thresh = self.segment_shikhta(preprocessed, self._polygon_mask)

        # Поиск контуров
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Разделение контуров
        left_thresh = cv2.bitwise_and(thresh, self._left_mask)
        right_thresh = cv2.bitwise_and(thresh, self._right_mask)

        left_contours, _ = cv2.findContours(
            left_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        right_contours, _ = cv2.findContours(
            right_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Расчет площадей
        left_area = sum(cv2.contourArea(cnt) for cnt in left_contours)
        right_area = sum(cv2.contourArea(cnt) for cnt in right_contours)
        total_area = left_area + right_area

        # Процентное соотношение
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

        # Визуализация
        if save_visualization and output_path:
            # Подготовка изображения для визуализации
            if self.perspective_transformer:
                # Если использовалась перспектива, показываем скорректированное изображение
                img_color = self.perspective_transformer.transform(
                    original_frame)
            else:
                img_color = cv2.resize(original_frame.copy(), self.target_size)

            min_y = self.polygon[:, 1].min()
            max_y = self.polygon[:, 1].max()

            # Рисуем контуры и границы
            cv2.drawContours(img_color, contours, -1, (0, 255, 0), 1)
            cv2.polylines(img_color, [self.polygon], True, (255, 0, 0), 2)
            cv2.line(img_color, (self._mid_x, min_y),
                     (self._mid_x, max_y), (0, 0, 255), 2)

            # Текст с метриками
            text = f"L: {left_percent:.1f}% | R: {right_percent:.1f}%"
            if self.perspective_transformer:
                text += " [PERSP]"
            cv2.putText(img_color, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)

            # Создаём композитное изображение если используется перспектива
            if self.perspective_transformer:
                # Создаём сравнительную визуализацию
                h, w = img_color.shape[:2]
                composite = np.zeros((h, w * 2 + 10, 3), dtype=np.uint8)

                # Оригинал слева
                orig_resized = cv2.resize(original_frame, (w, h))
                composite[:, :w] = orig_resized

                # Разделитель
                composite[:, w:w+10] = 128

                # Скорректированное справа
                composite[:, w+10:] = img_color

                # Сохранение
                cv2.imwrite(output_path, composite)
            else:
                cv2.imwrite(output_path, img_color)

        return metrics

    def _process_single_frame_wrapper(self, task):
        """Обертка для параллельной обработки"""
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
        """Обработка всех кадров видео с поддержкой параллелизма"""
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
                        print(
                            f"    Обработано {processed}/{len(frame_files)} кадров")

            # Сортировка по frame_idx
            self.frame_metrics.sort(key=lambda x: x['frame_idx'])
        else:
            # Последовательная обработка
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
        """Вычисление сводной статистики"""
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
        """Сохранение метрик в JSON с автоматической генерацией графиков"""
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

    def plot_metrics(self, output_path):
        """Построение графиков с выделением мин/макс"""
        if not self.frame_metrics:
            print("Нет данных для построения графиков")
            return

        # Извлечение данных
        frames = [m['frame_idx'] for m in self.frame_metrics]
        left_percents = [m['left_percent'] for m in self.frame_metrics]
        right_percents = [m['right_percent'] for m in self.frame_metrics]
        total_areas = [m['total_area'] for m in self.frame_metrics]

        # Поиск экстремумов
        left_min_idx = np.argmin(left_percents)
        left_max_idx = np.argmax(left_percents)
        right_min_idx = np.argmin(right_percents)
        right_max_idx = np.argmax(right_percents)

        # Создание графиков
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Добавление заголовка если использовалась перспективная коррекция
        if self.perspective_transformer:
            fig.suptitle('Анализ шихты с перспективной коррекцией',
                         fontsize=16, fontweight='bold', y=1.02)

        # График 1: Процентное соотношение
        ax1 = axes[0]
        ax1.plot(frames, left_percents, label='Левая часть',
                 color='#2E86DE', linewidth=2, alpha=0.8)
        ax1.plot(frames, right_percents, label='Правая часть',
                 color='#EE5A6F', linewidth=2, alpha=0.8)

        # Линия баланса 50%
        ax1.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5,
                    label='Баланс 50/50')

        # Выделение минимумов и максимумов
        # Левая часть
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

        # Правая часть
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

        # График 2: Общая площадь шихты
        ax2 = axes[1]
        ax2.fill_between(frames, total_areas, alpha=0.3, color='#10AC84')
        ax2.plot(frames, total_areas, color='#10AC84', linewidth=2)

        # Выделение экстремумов площади
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

        # Статистика площади
        area_stats = f'μ={np.mean(total_areas):.0f} σ={np.std(total_areas):.0f}'
        ax2.text(0.02, 0.98, area_stats, transform=ax2.transAxes,
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ График сохранен: {output_path}")
        plt.close()


if __name__ == "__main__":
    # Пример использования с перспективной коррекцией
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
