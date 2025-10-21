# perspective_transform.py
"""
Модуль для перспективной коррекции изображений печи
Преобразует вид под углом в вид сверху для улучшения анализа шихты
"""

import cv2
import json
import os
from pathlib import Path

import numpy as np


class PerspectiveTransformer:
    """Класс для перспективной трансформации изображений"""

    def __init__(self, src_points=None, dst_width=800, dst_height=600):
        """
        Args:
            src_points: 4 точки исходного изображения (углы области интереса)
            dst_width: ширина выходного изображения
            dst_height: высота выходного изображения
        """
        self.src_points = src_points
        self.dst_width = dst_width
        self.dst_height = dst_height
        self.transform_matrix = None
        self.inverse_matrix = None

        # Целевые точки (прямоугольник)
        self.dst_points = np.float32([
            [0, 0],
            [dst_width - 1, 0],
            [dst_width - 1, dst_height - 1],
            [0, dst_height - 1]
        ])

        if src_points is not None:
            self._calculate_transform()

    def _calculate_transform(self):
        """Вычисление матрицы преобразования"""
        if self.src_points is None:
            return

        src_pts = np.float32(self.src_points)

        # Прямое преобразование
        self.transform_matrix = cv2.getPerspectiveTransform(
            src_pts, self.dst_points)

        # Обратное преобразование (для маппинга координат обратно)
        self.inverse_matrix = cv2.getPerspectiveTransform(
            self.dst_points, src_pts)

    def set_source_points(self, points):
        """Установка исходных точек и пересчёт матрицы"""
        self.src_points = points
        self._calculate_transform()

    def transform(self, image):
        """
        Применение перспективного преобразования к изображению

        Args:
            image: входное изображение

        Returns:
            transformed: преобразованное изображение
        """
        if self.transform_matrix is None:
            raise ValueError(
                "Transform matrix not calculated. Set source points first.")

        transformed = cv2.warpPerspective(
            image,
            self.transform_matrix,
            (self.dst_width, self.dst_height),
            flags=cv2.INTER_LINEAR
        )

        return transformed

    def transform_point(self, point):
        """Преобразование одной точки"""
        if self.transform_matrix is None:
            return point

        pt = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.transform_matrix)
        return tuple(transformed[0][0].astype(int))

    def inverse_transform_point(self, point):
        """Обратное преобразование точки (из скорректированного в исходное)"""
        if self.inverse_matrix is None:
            return point

        pt = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.inverse_matrix)
        return tuple(transformed[0][0].astype(int))

    def transform_polygon(self, polygon):
        """Преобразование полигона"""
        if self.transform_matrix is None:
            return polygon

        polygon_array = np.array(polygon, dtype=np.float32).reshape((-1, 1, 2))
        transformed = cv2.perspectiveTransform(
            polygon_array, self.transform_matrix)
        return transformed.reshape((-1, 2)).astype(np.int32)

    def save_config(self, filepath):
        """Сохранение конфигурации в JSON"""
        config = {
            'src_points': self.src_points.tolist() if isinstance(self.src_points, np.ndarray) else self.src_points,
            'dst_width': self.dst_width,
            'dst_height': self.dst_height
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load_config(cls, filepath):
        """Загрузка конфигурации из JSON"""
        with open(filepath, 'r') as f:
            config = json.load(f)

        return cls(
            src_points=config['src_points'],
            dst_width=config['dst_width'],
            dst_height=config['dst_height']
        )


def setup_perspective_gui(image, window_name="Setup Perspective Correction"):
    """
    GUI для настройки перспективной коррекции
    Пользователь выбирает 4 угловые точки области интереса

    Returns:
        points: список из 4 точек или None при отмене
    """
    points = []
    clone = image.copy()
    h, w = image.shape[:2]

    # Подсказка по умолчанию - примерный прямоугольник
    default_points = [
        [int(w * 0.2), int(h * 0.2)],
        [int(w * 0.8), int(h * 0.2)],
        [int(w * 0.8), int(h * 0.8)],
        [int(w * 0.2), int(h * 0.8)]
    ]

    def mouse_callback(event, x, y, flags, param):
        nonlocal points

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN:
            if points:
                points.pop()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    instructions = [
        "НАСТРОЙКА ПЕРСПЕКТИВНОЙ КОРРЕКЦИИ",
        "Выберите 4 угловые точки области печи:",
        "ЛКМ - добавить точку | ПКМ - удалить последнюю",
        "D - использовать по умолчанию | R - сброс",
        "S - сохранить | ESC - отмена"
    ]

    while True:
        display = clone.copy()

        # Инструкции
        y_offset = 30
        for instruction in instructions:
            cv2.putText(display, instruction, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, instruction, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            y_offset += 25

        # Рисуем точки и линии
        for i, point in enumerate(points):
            cv2.circle(display, tuple(point), 5, (0, 255, 0), -1)
            cv2.putText(display, f"{i+1}", (point[0] + 10, point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Соединяем точки линиями
        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(display, tuple(points[i]), tuple(
                    points[i + 1]), (0, 255, 0), 2)

        # Замыкаем контур если 4 точки
        if len(points) == 4:
            cv2.line(display, tuple(points[3]),
                     tuple(points[0]), (0, 255, 0), 2)

            # Показываем предпросмотр преобразования
            transformer = PerspectiveTransformer(points)
            preview = transformer.transform(clone)
            preview_small = cv2.resize(preview, (300, 225))
            display[10:235, w-310:w-10] = preview_small
            cv2.rectangle(display, (w-310, 10), (w-10, 235), (0, 255, 0), 2)
            cv2.putText(display, "Preview", (w-305, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            cv2.destroyWindow(window_name)
            return None
        elif key == ord('s') and len(points) == 4:
            cv2.destroyWindow(window_name)
            return points
        elif key == ord('r'):
            points = []
        elif key == ord('d'):
            points = default_points.copy()

    cv2.destroyWindow(window_name)
    return None


def load_or_setup_perspective(video_name, frame_path, config_dir="data/perspective_configs"):
    """
    Загрузка существующей конфигурации или создание новой через GUI

    Args:
        video_name: имя видео для сохранения конфигурации
        frame_path: путь к первому кадру для настройки
        config_dir: директория для сохранения конфигураций

    Returns:
        PerspectiveTransformer object or None
    """
    config_path = os.path.join(config_dir, f"{video_name}_perspective.json")

    # Попытка загрузить существующую конфигурацию
    if os.path.exists(config_path):
        print(
            f"Загружена сохранённая конфигурация перспективы для {video_name}")
        return PerspectiveTransformer.load_config(config_path)

    # Если нет - показываем GUI для настройки
    print(f"Настройка перспективной коррекции для {video_name}")
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Не удалось загрузить кадр: {frame_path}")
        return None

    points = setup_perspective_gui(frame)

    if points is None:
        print("Настройка перспективы отменена")
        return None

    # Создаём трансформер
    transformer = PerspectiveTransformer(points)

    # Сохраняем конфигурацию
    transformer.save_config(config_path)
    print(f"Конфигурация перспективы сохранена: {config_path}")

    return transformer


def apply_perspective_to_video(frames_dir, transformer, output_dir=None):
    """
    Применение перспективной коррекции ко всем кадрам видео

    Args:
        frames_dir: директория с исходными кадрами
        transformer: объект PerspectiveTransformer
        output_dir: директория для сохранения преобразованных кадров (опционально)

    Returns:
        success: bool
    """
    if transformer is None:
        print("Трансформер не инициализирован")
        return False

    frames_dir = Path(frames_dir)
    frame_files = sorted(frames_dir.glob("*.jpg"))

    if not frame_files:
        print(f"Не найдено кадров в {frames_dir}")
        return False

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Применение перспективной коррекции к {len(frame_files)} кадрам...")

    for i, frame_path in enumerate(frame_files):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        # Применяем трансформацию
        corrected = transformer.transform(frame)

        if output_dir:
            output_path = Path(output_dir) / frame_path.name
            cv2.imwrite(str(output_path), corrected)

        if (i + 1) % 100 == 0:
            print(f"  Обработано {i + 1}/{len(frame_files)} кадров")

    print(f"✓ Перспективная коррекция завершена")
    return True


if __name__ == "__main__":
    # Пример использования
    import sys

    if len(sys.argv) < 2:
        print("Usage: python perspective_transform.py <frame_image>")
        sys.exit(1)

    # Загрузка изображения
    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Не удалось загрузить изображение: {sys.argv[1]}")
        sys.exit(1)

    # Настройка через GUI
    points = setup_perspective_gui(img)

    if points:
        # Создание трансформера
        transformer = PerspectiveTransformer(points)

        # Применение трансформации
        corrected = transformer.transform(img)

        # Показ результата
        cv2.imshow("Original", cv2.resize(img, (600, 400)))
        cv2.imshow("Corrected", cv2.resize(corrected, (600, 400)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
