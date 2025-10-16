"""
Улучшенный анализатор шихты с зональной адаптивной обработкой
ПОЛНАЯ ВЕРСИЯ с всеми методами для интеграции
"""

import cv2
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage import img_as_float64, restoration, img_as_ubyte
import scipy.signal


class ImprovedShikhtaAnalyzer:
    """Улучшенный анализатор с зональной обработкой"""
    
    def __init__(self, polygon=None, target_size=(928, 576), 
                 perspective_transformer=None,
                 min_contour_area=100,  # Минимальная площадь контура
                 near_zone_ratio=0.5):  # Доля "ближней" зоны
        
        self.polygon = polygon if polygon is not None else self._get_default_polygon()
        if not isinstance(self.polygon, np.ndarray):
            try:
                self.polygon = np.array(self.polygon, dtype=np.int32)
            except Exception:
                self.polygon = np.array(self._get_default_polygon(), dtype=np.int32)
        else:
            self.polygon = self.polygon.astype(np.int32)
            
        self.target_size = target_size
        self.perspective_transformer = perspective_transformer
        self.min_contour_area = min_contour_area
        self.near_zone_ratio = near_zone_ratio
        self.frame_metrics = []
        
        # Предвычисленные маски
        self._polygon_mask = None
        self._left_mask = None
        self._right_mask = None
        self._near_mask = None
        self._far_mask = None
        self._mid_x = None
        self._setup_masks()
    
    def _get_default_polygon(self):
        return np.array([
            [215, 113], [625, 118], [733, 270],
            [577, 529], [144, 530], [54, 277]
        ], np.int32)
    
    def _setup_masks(self):
        """Создание масок включая зональное разделение"""
        dummy = np.zeros(self.target_size[::-1], dtype=np.uint8)
        
        # Основная маска полигона
        self._polygon_mask = np.zeros_like(dummy, dtype=np.uint8)
        cv2.fillPoly(self._polygon_mask, [self.polygon], 255)
        
        # Разделение на левую/правую части
        self._mid_x = (self.polygon[:, 0].min() + self.polygon[:, 0].max()) // 2
        
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
        
        # Зональное разделение (ближняя/дальняя части)
        y_coords = self.polygon[:, 1]
        y_min, y_max = y_coords.min(), y_coords.max()
        y_threshold = y_min + (y_max - y_min) * self.near_zone_ratio
        
        # Маска для ближней зоны (нижняя часть изображения)
        self._near_mask = self._polygon_mask.copy()
        self._near_mask[:int(y_threshold), :] = 0
        
        # Маска для дальней зоны (верхняя часть)
        self._far_mask = self._polygon_mask.copy()
        self._far_mask[int(y_threshold):, :] = 0
    
    def preprocess_frame(self, frame):
        """Предобработка с перспективной коррекцией"""
        if self.perspective_transformer is not None:
            frame = self.perspective_transformer.transform(frame)
        
        if len(frame.shape) == 3:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            img = frame
        
        if img.shape[:2] != self.target_size[::-1]:
            img = cv2.resize(img, self.target_size)
        
        # Улучшенная предобработка
        img_hist_eq = cv2.equalizeHist(img)
        img_float = img_as_float64(img_hist_eq)
        
        # Фильтр Винера
        kernel = np.ones((15, 15), np.float64)
        image_filtered = scipy.signal.convolve2d(img_float, kernel, 'same')
        img_wiener = restoration.wiener(image_filtered, kernel, 5.1e4)
        img_wiener = img_as_ubyte(img_wiener)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        image_filtered = clahe.apply(img_wiener)
        
        return img, image_filtered
    
    def detect_flame_regions(self, preprocessed_img):
        """Детекция областей с ярким пламенем"""
        _, flame_mask = cv2.threshold(preprocessed_img, 220, 255, cv2.THRESH_BINARY)
        kernel = np.ones((15, 15), np.uint8)
        flame_mask = cv2.dilate(flame_mask, kernel, iterations=2)
        return flame_mask
    
    def segment_shikhta(self, preprocessed_img, polygon_mask):
        """СТАРЫЙ метод для обратной совместимости"""
        return self.segment_shikhta_adaptive(preprocessed_img, polygon_mask)
    
    def segment_shikhta_adaptive(self, preprocessed_img, polygon_mask):
        """Адаптивная сегментация шихты с зональными параметрами"""
        # 1. Детекция пламени
        flame_mask = self.detect_flame_regions(preprocessed_img)
        
        # 2. Сегментация для ближней зоны (более строгие параметры)
        near_thresh = cv2.adaptiveThreshold(
            preprocessed_img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 
            191, -5  # C = -5 (более строгий порог)
        )
        
        # 3. Сегментация для дальней зоны (более мягкие параметры)
        far_thresh = cv2.adaptiveThreshold(
            preprocessed_img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            191, 5  # C = 5 (более мягкий порог)
        )
        
        # 4. Комбинируем результаты по зонам
        combined_thresh = np.zeros_like(near_thresh)
        combined_thresh = cv2.bitwise_or(
            combined_thresh,
            cv2.bitwise_and(near_thresh, self._near_mask)
        )
        combined_thresh = cv2.bitwise_or(
            combined_thresh,
            cv2.bitwise_and(far_thresh, self._far_mask)
        )
        
        # 5. Исключаем области с пламенем
        combined_thresh = cv2.bitwise_and(
            combined_thresh,
            cv2.bitwise_not(flame_mask)
        )
        
        # 6. Морфологическая обработка
        kernel_close = np.ones((7, 7), np.uint8)
        combined_thresh = cv2.morphologyEx(
            combined_thresh, 
            cv2.MORPH_CLOSE, 
            kernel_close
        )
        
        # 7. Удаление мелких объектов в ближней зоне
        near_region = cv2.bitwise_and(combined_thresh, self._near_mask)
        near_region = self._remove_small_contours(
            near_region, 
            min_area=self.min_contour_area * 2
        )
        
        # 8. Удаление мелких объектов в дальней зоне
        far_region = cv2.bitwise_and(combined_thresh, self._far_mask)
        far_region = self._remove_small_contours(
            far_region,
            min_area=self.min_contour_area
        )
        
        # 9. Объединяем обработанные зоны
        combined_thresh = cv2.bitwise_or(near_region, far_region)
        
        # 10. Применяем маску полигона
        combined_thresh = cv2.bitwise_and(combined_thresh, polygon_mask)
        
        return combined_thresh
    
    def _remove_small_contours(self, binary_img, min_area=50):
        """Удаление контуров меньше заданной площади"""
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
    
    def analyze_frame(self, frame, frame_idx=0, save_visualization=False, 
                     output_path=None):
        """Анализ кадра с улучшенной сегментацией"""
        original_frame = frame.copy()
        
        # Предобработка
        img_gray, preprocessed = self.preprocess_frame(frame)
        
        # Улучшенная сегментация
        thresh = self.segment_shikhta_adaptive(preprocessed, self._polygon_mask)
        
        # Поиск контуров
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Разделение контуров
        left_thresh = cv2.bitwise_and(thresh, self._left_mask)
        right_thresh = cv2.bitwise_and(thresh, self._right_mask)
        
        left_contours, _ = cv2.findContours(
            left_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        right_contours, _ = cv2.findContours(
            right_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
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
            self._save_visualization(
                original_frame, contours, 
                left_percent, right_percent, output_path
            )
        
        return metrics
    
    def _save_visualization(self, original_frame, contours, 
                           left_percent, right_percent, output_path):
        """Сохранение визуализации с дополнительной информацией"""
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
        
        # Показываем зональное разделение
        y_threshold = int(min_y + (max_y - min_y) * self.near_zone_ratio)
        cv2.line(img_color, (self.polygon[:, 0].min(), y_threshold),
                (self.polygon[:, 0].max(), y_threshold), (255, 255, 0), 1)
        
        # Текст с метриками
        text = f"L: {left_percent:.1f}% | R: {right_percent:.1f}%"
        if self.perspective_transformer:
            text += " [PERSP]"
        cv2.putText(img_color, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Дополнительная информация
        info_text = f"Contours: {len(contours)} | MinArea: {self.min_contour_area}"
        cv2.putText(img_color, info_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
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
        
        print(f"Обработка {len(frame_files)} кадров из {Path(frames_dir).name}...")
        if self.perspective_transformer:
            print("  • Перспективная коррекция: включена")
        print(f"  • Зональная адаптация: включена (порог={self.near_zone_ratio:.0%})")
        print(f"  • Мин. площадь контура: {self.min_contour_area} пикс²")
        
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
            print(f"  • Режим: последовательный")
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
            'min_contour_area': self.min_contour_area,
            'near_zone_ratio': self.near_zone_ratio,
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
        
        # Заголовок с информацией об улучшениях
        title = 'Анализ шихты (улучшенный алгоритм)'
        if self.perspective_transformer:
            title += ' с перспективной коррекцией'
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        
        # График 1: Процентное соотношение
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
        
        ax1.set_xlabel('Номер кадра', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Процент шихты, %', fontsize=12, fontweight='bold')
        ax1.set_title('Распределение шихты по левой и правой частям',
                     fontsize=14, fontweight='bold', pad=15)
        ax1.legend(loc='best', fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        ax1.set_ylim(0, 100)
        
        # Статистика
        stats_text = (f'Левая: μ={np.mean(left_percents):.1f}% σ={np.std(left_percents):.1f}%\n'
                     f'Правая: μ={np.mean(right_percents):.1f}% σ={np.std(right_percents):.1f}%\n'
                     f'MinArea={self.min_contour_area} | Zone={self.near_zone_ratio:.0%}')
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # График 2: Общая площадь шихты
        ax2 = axes[1]
        ax2.fill_between(frames, total_areas, alpha=0.3, color='#10AC84')
        ax2.plot(frames, total_areas, color='#10AC84', linewidth=2)
        
        area_min_idx = np.argmin(total_areas)
        area_max_idx = np.argmax(total_areas)
        
        for idx, marker, name in [(area_min_idx, 'v', 'MIN'), 
                                  (area_max_idx, '^', 'MAX')]:
            ax2.scatter(frames[idx], total_areas[idx], color='green', 
                       s=150, marker=marker, zorder=5, 
                       edgecolors='black', linewidth=2)
            ax2.annotate(f'{name}: {total_areas[idx]:.0f}',
                        xy=(frames[idx], total_areas[idx]),
                        xytext=(10, -20 if marker == 'v' else 20), 
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5',
                                facecolor='lightgreen', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', color='green', lw=2))
        
        ax2.set_xlabel('Номер кадра', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Площадь, пикс²', fontsize=12, fontweight='bold')
        ax2.set_title('Общая площадь шихты во времени',
                     fontsize=14, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        
        area_stats = f'μ={np.mean(total_areas):.0f} σ={np.std(total_areas):.0f}'
        ax2.text(0.02, 0.98, area_stats, transform=ax2.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ График сохранен: {output_path}")
        plt.close()