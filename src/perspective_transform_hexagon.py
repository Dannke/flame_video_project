# perspective_transform_hexagon.py
"""
Модуль для перспективной коррекции шестиугольной печи
Использует разбиение на треугольники для более точного преобразования
"""

import cv2
import json
import os
from pathlib import Path
import numpy as np


class HexagonPerspectiveTransformer:
    """Класс для перспективной трансформации шестиугольных областей"""

    def __init__(self, src_points=None, dst_width=1920, dst_height=1280):
        """
        Args:
            src_points: 6 точек исходного шестиугольника (по часовой стрелке)
            dst_width: ширина выходного изображения
            dst_height: высота выходного изображения
        """
        self.src_points = src_points
        self.dst_width = dst_width
        self.dst_height = dst_height

        # Целевой правильный шестиугольник
        self.dst_points = self._create_regular_hexagon(dst_width, dst_height)

        # Треугольники для разбиения
        self.triangles = None
        self.affine_matrices = []

        if src_points is not None:
            self._calculate_transform()

    def _create_regular_hexagon(self, width, height):
        """Создание правильного шестиугольника для целевого изображения"""
        # Центр
        cx, cy = width / 2, height / 2

        # Радиус (берём минимум, чтобы вписаться)
        radius = min(width, height) * 0.45

        # 6 точек по кругу (начиная сверху, по часовой стрелке)
        # Поворот на 60° влево (против часовой) = добавляем π/3 к углу
        points = []
        for i in range(6):
            angle = np.pi / 2 + np.pi / 6 - i * np.pi / 3  # +30° поворот
            x = cx + radius * np.cos(angle)
            y = cy - radius * np.sin(angle)
            points.append([x, y])

        return np.float32(points)

    def _calculate_transform(self):
        """Вычисление аффинных преобразований для треугольников"""
        if self.src_points is None:
            return

        src_pts = np.float32(self.src_points)

        # Центральная точка (центроид)
        src_center = np.mean(src_pts, axis=0)
        dst_center = np.mean(self.dst_points, axis=0)

        # Разбиение на 6 треугольников (центр + каждое ребро)
        self.triangles = []
        self.affine_matrices = []

        for i in range(6):
            next_i = (i + 1) % 6

            # Треугольник: центр -> точка i -> точка i+1
            src_tri = np.float32([
                src_center,
                src_pts[i],
                src_pts[next_i]
            ])

            dst_tri = np.float32([
                dst_center,
                self.dst_points[i],
                self.dst_points[next_i]
            ])

            # Аффинное преобразование для этого треугольника
            affine_mat = cv2.getAffineTransform(src_tri, dst_tri)

            self.triangles.append((src_tri, dst_tri))
            self.affine_matrices.append(affine_mat)

    def set_source_points(self, points):
        """Установка исходных точек и пересчёт матриц"""
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
        if self.triangles is None:
            raise ValueError(
                "Transform not calculated. Set source points first.")

        # Создаём выходное изображение
        result = np.zeros((self.dst_height, self.dst_width, 3), dtype=np.uint8)

        # Создаём маску для отслеживания заполненных пикселей
        mask = np.zeros((self.dst_height, self.dst_width), dtype=np.uint8)

        # Преобразуем каждый треугольник отдельно
        for i, (src_tri, dst_tri) in enumerate(self.triangles):
            # Создаём маску для текущего треугольника в целевом изображении
            tri_mask = np.zeros(
                (self.dst_height, self.dst_width), dtype=np.uint8)
            cv2.fillConvexPoly(tri_mask, dst_tri.astype(np.int32), 255)

            # Применяем аффинное преобразование
            warped = cv2.warpAffine(
                image,
                self.affine_matrices[i],
                (self.dst_width, self.dst_height),
                flags=cv2.INTER_LINEAR
            )

            # Копируем только пиксели внутри треугольника
            result[tri_mask > 0] = warped[tri_mask > 0]
            mask[tri_mask > 0] = 255

        return result

    def transform_point(self, point):
        """Преобразование одной точки (находим треугольник и применяем его аффинное преобразование)"""
        if self.triangles is None:
            return point

        pt = np.array(point, dtype=np.float32)

        # Определяем, в каком треугольнике находится точка
        for i, (src_tri, dst_tri) in enumerate(self.triangles):
            # Проверка принадлежности точки треугольнику
            if self._point_in_triangle(pt, src_tri):
                # Применяем аффинное преобразование
                pt_homogeneous = np.array([pt[0], pt[1], 1.0])
                transformed = self.affine_matrices[i] @ pt_homogeneous
                return tuple(transformed.astype(int))

        # Если точка не в одном из треугольников, используем первое преобразование
        pt_homogeneous = np.array([pt[0], pt[1], 1.0])
        transformed = self.affine_matrices[0] @ pt_homogeneous
        return tuple(transformed.astype(int))

    def _point_in_triangle(self, pt, tri):
        """Проверка, находится ли точка внутри треугольника (барицентрические координаты)"""
        v0 = tri[2] - tri[0]
        v1 = tri[1] - tri[0]
        v2 = pt - tri[0]

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        return (u >= 0) and (v >= 0) and (u + v <= 1)

    def transform_polygon(self, polygon):
        """Преобразование полигона"""
        if self.triangles is None:
            return polygon

        transformed = []
        for point in polygon:
            transformed.append(self.transform_point(point))

        return np.array(transformed, dtype=np.int32)

    def save_config(self, filepath):
        """Сохранение конфигурации в JSON"""
        config = {
            'src_points': self.src_points.tolist() if isinstance(self.src_points, np.ndarray) else self.src_points,
            'dst_width': self.dst_width,
            'dst_height': self.dst_height,
            'method': 'hexagon_triangulation'
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


def setup_hexagon_perspective_gui(image, window_name="Setup Hexagon Perspective"):
    """
    GUI для настройки перспективной коррекции шестиугольника
    Пользователь выбирает 6 угловых точек

    Returns:
        points: список из 6 точек или None при отмене
    """
    points = []
    clone = image.copy()
    h, w = image.shape[:2]

    # Подсказка по умолчанию - правильный шестиугольник
    cx, cy = w // 2, h // 2
    radius = min(w, h) * 0.35
    default_points = []
    for i in range(6):
        angle = np.pi / 2 - i * np.pi / 3
        x = int(cx + radius * np.cos(angle))
        y = int(cy - radius * np.sin(angle))
        default_points.append([x, y])

    def mouse_callback(event, x, y, flags, param):
        nonlocal points

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 6:
                points.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN:
            if points:
                points.pop()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    instructions = [
        "НАСТРОЙКА ПЕРСПЕКТИВЫ ШЕСТИУГОЛЬНИКА",
        "Выберите 6 угловых точек печи (по часовой стрелке):",
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

        # Замыкаем контур если 6 точек
        if len(points) == 6:
            cv2.line(display, tuple(points[5]),
                     tuple(points[0]), (0, 255, 0), 2)

            # Показываем предпросмотр преобразования
            transformer = HexagonPerspectiveTransformer(points)
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
        elif key == ord('s') and len(points) == 6:
            cv2.destroyWindow(window_name)
            return points
        elif key == ord('r'):
            points = []
        elif key == ord('d'):
            points = default_points.copy()

    cv2.destroyWindow(window_name)
    return None


if __name__ == "__main__":
    # Пример использования
    import sys

    if len(sys.argv) < 2:
        print("Usage: python perspective_transform_hexagon.py <frame_image>")
        sys.exit(1)

    # Загрузка изображения
    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Не удалось загрузить изображение: {sys.argv[1]}")
        sys.exit(1)

    # Настройка через GUI
    points = setup_hexagon_perspective_gui(img)

    if points:
        # Создание трансформера
        transformer = HexagonPerspectiveTransformer(
            points, dst_width=1920, dst_height=1280)

        # Применение трансформации
        corrected = transformer.transform(img)

        # Показ результата
        cv2.imshow("Original", cv2.resize(img, (1920, 1280)))
        cv2.imshow("Corrected (Hexagon)", cv2.resize(corrected, (1920, 1280)))

        # Сохранение конфига
        transformer.save_config("hexagon_transform_config.json")
        print("✓ Конфигурация сохранена: hexagon_transform_config.json")

        cv2.waitKey(0)
        cv2.destroyAllWindows()
