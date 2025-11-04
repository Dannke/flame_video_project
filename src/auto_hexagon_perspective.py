# auto_hexagon_perspective.py
"""
Автоматическая генерация полигона для перспективной коррекции
на основе основного полигона печи с предпросмотром и ручной коррекцией
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path


class AutoHexagonGenerator:
    """Генератор hexagon полигона на основе базового полигона"""
    
    # Оптимальный профиль расширения (широкое расширение)
    WIDE_PROFILE = {
        'name': 'Wide expansion',
        'description': 'Оптимизированное расширение для захвата искажений',
        'zones': [
            {'points': [0, 1], 'scale': 1.3},   # Верхние точки
            {'points': [2, 5], 'scale': 1.35},  # Боковые точки (максимальное расширение)
            {'points': [3, 4], 'scale': 1.2}    # Нижние точки
        ]
    }
    
    def __init__(self, base_polygon, image_shape):
        """
        Args:
            base_polygon: Основной полигон печи (6 точек)
            image_shape: (height, width) исходного изображения
        """
        self.base_polygon = np.array(base_polygon, dtype=np.float32)
        self.image_shape = image_shape
        self.expanded_polygon = None
        
        if len(self.base_polygon) != 6:
            raise ValueError("Требуется полигон из 6 точек")
    
    def generate(self):
        """Генерация расширенного полигона по профилю Wide"""
        profile_config = self.WIDE_PROFILE
        
        # Вычисляем центроид базового полигона
        centroid = np.mean(self.base_polygon, axis=0)
        
        expanded_points = []
        
        # Зонированное расширение (Wide profile)
        point_scales = {}
        for zone in profile_config['zones']:
            for idx in zone['points']:
                point_scales[idx] = zone['scale']
        
        for idx, point in enumerate(self.base_polygon):
            scale = point_scales.get(idx, 1.20)  # Fallback scale
            direction = point - centroid
            new_point = centroid + direction * scale
            expanded_points.append(new_point)
        
        self.expanded_polygon = np.array(expanded_points, dtype=np.float32)
        
        # Проверка границ изображения
        self._clip_to_image_bounds()
        
        # Сглаживание для плавности
        self._smooth_polygon()
        
        return self.expanded_polygon.astype(np.int32)
    
    def _clip_to_image_bounds(self):
        """Обрезка точек по границам изображения"""
        height, width = self.image_shape[:2]
        
        self.expanded_polygon[:, 0] = np.clip(
            self.expanded_polygon[:, 0], 0, width - 1
        )
        self.expanded_polygon[:, 1] = np.clip(
            self.expanded_polygon[:, 1], 0, height - 1
        )
    
    def _smooth_polygon(self, strength=0.1):
        """
        Сглаживание полигона для плавности углов
        
        Args:
            strength: Сила сглаживания (0-1)
        """
        smoothed = self.expanded_polygon.copy()
        n = len(smoothed)
        
        for i in range(n):
            prev_idx = (i - 1) % n
            next_idx = (i + 1) % n
            
            # Среднее с соседними точками
            neighbor_avg = (
                self.expanded_polygon[prev_idx] + 
                self.expanded_polygon[next_idx]
            ) / 2
            
            # Смешивание текущей точки и среднего
            smoothed[i] = (
                (1 - strength) * self.expanded_polygon[i] + 
                strength * neighbor_avg
            )
        
        self.expanded_polygon = smoothed
    
    def visualize(self, image):
        """
        Визуализация обоих полигонов
        
        Returns:
            Изображение с нарисованными полигонами
        """
        vis = image.copy()
        
        # Базовый полигон (зелёный)
        cv2.polylines(
            vis, 
            [self.base_polygon.astype(np.int32)], 
            True, 
            (0, 255, 0), 
            2
        )
        
        # Расширенный полигон (синий)
        if self.expanded_polygon is not None:
            cv2.polylines(
                vis, 
                [self.expanded_polygon.astype(np.int32)], 
                True, 
                (255, 0, 0), 
                2
            )
            
            # Стрелки от базового к расширенному
            for base_pt, exp_pt in zip(
                self.base_polygon.astype(np.int32), 
                self.expanded_polygon.astype(np.int32)
            ):
                cv2.arrowedLine(
                    vis, 
                    tuple(base_pt), 
                    tuple(exp_pt), 
                    (0, 255, 255), 
                    1, 
                    tipLength=0.3
                )
        
        # Легенда
        cv2.putText(
            vis, 
            "Green: Base polygon", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        cv2.putText(
            vis, 
            "Blue: Expanded polygon (for perspective)", 
            (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 0, 0), 
            2
        )
        
        return vis


def setup_auto_hexagon_with_preview(
    image, 
    base_polygon,
    initial_points=None,
    window_name="Auto Hexagon Setup",
    window_width=1400,  # Ширина окна по умолчанию
    window_height=900   # Высота окна по умолчанию
):
    """
    Автоматическая установка hexagon полигона с предпросмотром и редактированием
    
    Args:
        image: Исходное изображение
        base_polygon: Базовый полигон печи (6 точек)
        initial_points: Начальные точки (если есть существующий полигон)
        window_name: Название окна
        window_width: Ширина окна (None = автоматически)
        window_height: Высота окна (None = автоматически)
        
    Returns:
        expanded_polygon или None (отмена)
    """
    
    # Генерация расширенного полигона (если нет начальных точек)
    if initial_points is None:
        generator = AutoHexagonGenerator(base_polygon, image.shape)
        points = generator.generate()
    else:
        points = np.array(initial_points, dtype=np.float32)
    
    # Состояние редактирования
    editing = False
    selected_point = None
    drag_offset = None
    
    def redraw():
        """Перерисовка изображения с полигонами"""
        vis = image.copy()
        h, w = vis.shape[:2]
        
        # Рисуем базовый полигон (зелёный)
        if base_polygon is not None:
            base_array = np.array(base_polygon, dtype=np.int32)
            cv2.polylines(vis, [base_array], True, (0, 255, 0), 2)
        
        # Рисуем расширенный полигон (синий или жёлтый в режиме редактирования)
        color = (0, 255, 255) if editing else (255, 0, 0)
        points_int = points.astype(np.int32)
        cv2.polylines(vis, [points_int], True, color, 3)
        
        # Рисуем точки
        for i, pt in enumerate(points_int):
            # Подсветка выбранной точки
            point_color = (255, 255, 0) if (editing and i == selected_point) else color
            point_radius = 8 if (editing and i == selected_point) else 6
            
            cv2.circle(vis, tuple(pt), point_radius, point_color, -1)
            cv2.circle(vis, tuple(pt), point_radius + 2, (0, 0, 0), 2)
            
            # Номер точки
            cv2.putText(
                vis,
                f"{i+1}",
                (pt[0] + 12, pt[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                point_color,
                2
            )
        
        # Стрелки от базового к расширенному (только если не редактируем)
        if not editing and base_polygon is not None:
            base_array = np.array(base_polygon, dtype=np.int32)
            for base_pt, exp_pt in zip(base_array, points_int):
                cv2.arrowedLine(
                    vis, 
                    tuple(base_pt), 
                    tuple(exp_pt), 
                    (0, 255, 255), 
                    1, 
                    tipLength=0.3
                )
        
        # Легенда
        legend_y = 30
        if editing:
            cv2.putText(
                vis, 
                "EDITING MODE - Drag points to adjust", 
                (10, legend_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 255), 
                2
            )
            legend_y += 30
            cv2.putText(
                vis, 
                "Click on a point and drag to move it", 
                (10, legend_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                1
            )
        else:
            cv2.putText(
                vis, 
                "Green: Base polygon | Blue: Expanded (for perspective)", 
                (10, legend_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2
            )
        
        return vis
    
    # Панель с кнопками
    def create_panel(w, h):
        panel_height = 120
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        
        # Заголовок
        title = "EDIT MODE" if editing else "AUTO HEXAGON PERSPECTIVE - WIDE EXPANSION"
        cv2.putText(
            panel, 
            title,
            (10, 25), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 255) if editing else (255, 255, 255), 
            2
        )
        
        if not editing:
            cv2.putText(
                panel, 
                "Optimized expansion: top +30%, sides +35%, bottom +20%", 
                (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (200, 200, 200), 
                1
            )
        
        # Кнопки
        button_accept_rect = (w // 2 - 320, 70, 300, 40)
        button_edit_rect = (w // 2 + 20, 70, 300, 40)
        
        # Кнопка "Accept"
        cv2.rectangle(
            panel, 
            (button_accept_rect[0], button_accept_rect[1]),
            (button_accept_rect[0] + button_accept_rect[2], 
             button_accept_rect[1] + button_accept_rect[3]),
            (0, 200, 0), 
            -1
        )
        accept_text = "SAVE" if editing else "ACCEPT"
        cv2.putText(
            panel,
            accept_text,
            (button_accept_rect[0] + 90, button_accept_rect[1] + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Кнопка "Edit" / "Cancel Edit"
        edit_color = (100, 100, 100) if editing else (0, 100, 200)
        cv2.rectangle(
            panel, 
            (button_edit_rect[0], button_edit_rect[1]),
            (button_edit_rect[0] + button_edit_rect[2], 
             button_edit_rect[1] + button_edit_rect[3]),
            edit_color, 
            -1
        )
        edit_text = "CANCEL EDIT" if editing else "EDIT"
        text_offset = 40 if editing else 110
        cv2.putText(
            panel,
            edit_text,
            (button_edit_rect[0] + text_offset, button_edit_rect[1] + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        return panel, button_accept_rect, button_edit_rect, panel_height
    
    # Обработчик мыши
    mouse_state = {
        'action': None,
        'dragging': False
    }
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_point, drag_offset, editing
        
        h, w = image.shape[:2]
        
        # Клик по кнопкам (только если не в режиме перетаскивания)
        if event == cv2.EVENT_LBUTTONDOWN and not mouse_state['dragging']:
            # Проверяем клик по кнопкам
            panel_h = param['panel_height']
            button_accept = param['button_accept']
            button_edit = param['button_edit']
            
            # Клик по Accept/Save
            if (button_accept[0] <= x <= button_accept[0] + button_accept[2] and
                h + button_accept[1] <= y <= h + button_accept[1] + button_accept[3]):
                mouse_state['action'] = 'ACCEPT'
                return
            
            # Клик по Edit/Cancel Edit
            elif (button_edit[0] <= x <= button_edit[0] + button_edit[2] and
                  h + button_edit[1] <= y <= h + button_edit[1] + button_edit[3]):
                if editing:
                    # Отмена редактирования - восстанавливаем исходные точки
                    if initial_points is None:
                        generator = AutoHexagonGenerator(base_polygon, image.shape)
                        param['points'][:] = generator.generate()
                    else:
                        param['points'][:] = np.array(initial_points, dtype=np.float32)
                    editing = False
                    selected_point = None
                else:
                    editing = True
                return
        
        # Редактирование точек (только в режиме редактирования)
        if editing:
            if event == cv2.EVENT_LBUTTONDOWN:
                # Проверяем клик по точке
                for i, pt in enumerate(param['points']):
                    dist = np.linalg.norm(np.array([x, y]) - pt)
                    if dist < 15:  # Радиус захвата
                        selected_point = i
                        drag_offset = pt - np.array([x, y])
                        mouse_state['dragging'] = True
                        break
            
            elif event == cv2.EVENT_MOUSEMOVE and mouse_state['dragging'] and selected_point is not None:
                # Перетаскивание точки
                new_pos = np.array([x, y], dtype=np.float32) + drag_offset
                
                # Ограничиваем границами изображения
                new_pos[0] = np.clip(new_pos[0], 0, w - 1)
                new_pos[1] = np.clip(new_pos[1], 0, h - 1)
                
                param['points'][selected_point] = new_pos
            
            elif event == cv2.EVENT_LBUTTONUP:
                # Отпускаем точку
                mouse_state['dragging'] = False
                selected_point = None
                drag_offset = None
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Устанавливаем размер окна если указан
    if window_width and window_height:
        cv2.resizeWindow(window_name, window_width, window_height)
    
    while True:
        # Рисуем изображение
        vis = redraw()
        h, w = vis.shape[:2]
        
        # Создаём панель
        panel, btn_accept, btn_edit, panel_h = create_panel(w, h)
        
        # Объединяем
        full_preview = np.vstack([vis, panel])
        
        # Настраиваем callback с параметрами
        callback_params = {
            'points': points,
            'button_accept': btn_accept,
            'button_edit': btn_edit,
            'panel_height': panel_h
        }
        cv2.setMouseCallback(window_name, mouse_callback, callback_params)
        
        cv2.imshow(window_name, full_preview)
        key = cv2.waitKey(1) & 0xFF
        
        # Проверяем результат
        if mouse_state['action'] == 'ACCEPT' or key == ord('1'):
            cv2.destroyWindow(window_name)
            return points.astype(np.int32).tolist()
        
        elif key == 27:  # ESC
            cv2.destroyWindow(window_name)
            return None
    
    cv2.destroyWindow(window_name)
    return None


def show_existing_polygon_preview(
    image,
    existing_points,
    base_polygon=None,
    window_name="Existing Perspective Polygon",
    window_width=1400,
    window_height=900
):
    """
    Показ предпросмотра существующего полигона с возможностью редактирования
    
    Args:
        image: Исходное изображение
        existing_points: Существующие точки hexagon полигона
        base_polygon: Базовый полигон печи (для контекста)
        window_name: Название окна
        window_width: Ширина окна
        window_height: Высота окна
        
    Returns:
        points: Отредактированные точки или None (отмена)
    """
    
    # Используем ту же функцию редактирования, но с существующими точками
    return setup_auto_hexagon_with_preview(
        image,
        base_polygon,
        initial_points=existing_points,
        window_name=window_name,
        window_width=window_width,
        window_height=window_height
    )


def integrate_with_existing_gui(
    image, 
    base_polygon, 
    video_name, 
    save_dir,
    existing_config_path=None
):
    """
    Интеграция с существующей системой - включает проверку существующего полигона
    
    Args:
        image: Исходное изображение
        base_polygon: Базовый полигон печи
        video_name: Имя видео
        save_dir: Директория для сохранения
        existing_config_path: Путь к существующему конфигу (если есть)
        
    Returns:
        points: Список точек для hexagon или None
    """
    
    # === ШАГ 0: Проверка существующего полигона ===
    if existing_config_path and os.path.exists(existing_config_path):
        try:
            with open(existing_config_path, 'r') as f:
                config = json.load(f)
            
            existing_points = config.get('src_points')
            
            if existing_points and len(existing_points) == 6:
                print(f"✓ Найден существующий hexagon полигон: {existing_config_path}")
                
                # Показываем интерактивное окно с возможностью редактирования
                result = show_existing_polygon_preview(
                    image, 
                    existing_points,
                    base_polygon=base_polygon
                )
                
                # Теперь result это либо отредактированные точки, либо None
                if result is not None:
                    print("✓ Полигон принят/отредактирован пользователем")
                    return result
                else:
                    print("→ Отмена пользователем")
                    return None
        
        except Exception as e:
            print(f"⚠ Ошибка загрузки существующего конфига: {e}")
    
    # === ШАГ 1: Автоматическая генерация с интерактивным редактированием ===
    result = setup_auto_hexagon_with_preview(
        image, 
        base_polygon,
        initial_points=None
    )
    
    return result


# =============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("AUTO HEXAGON PERSPECTIVE GENERATOR - WIDE EXPANSION")
    print("="*70)
    
    # Пример: загрузка изображения и базового полигона
    # В реальном проекте это будет из polygon_gui
    
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python auto_hexagon_perspective.py <image_path>")
        print("\nТестовый режим с демо-данными...")
        
        # Создаём демо-изображение
        demo_image = np.zeros((720, 1280, 3), dtype=np.uint8)
        demo_image[:] = (50, 50, 50)
        
        # Демо-полигон (центр изображения)
        cx, cy = 640, 360
        radius = 200
        demo_polygon = []
        for i in range(6):
            angle = np.pi / 2 - i * np.pi / 3
            x = int(cx + radius * np.cos(angle))
            y = int(cy - radius * np.sin(angle))
            demo_polygon.append([x, y])
        
        # Рисуем базовый полигон
        cv2.polylines(
            demo_image, 
            [np.array(demo_polygon, np.int32)], 
            True, 
            (0, 200, 0), 
            2
        )
        
        image = demo_image
        base_polygon = demo_polygon
    
    else:
        # Загрузка реального изображения
        image = cv2.imread(sys.argv[1])
        if image is None:
            print(f"Ошибка: не удалось загрузить {sys.argv[1]}")
            sys.exit(1)
        
        # В реальном проекте полигон загружается из polygon_gui
        # Здесь используем демо-значения
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2
        radius = min(w, h) // 3
        
        base_polygon = []
        for i in range(6):
            angle = np.pi / 2 - i * np.pi / 3
            x = int(cx + radius * np.cos(angle))
            y = int(cy - radius * np.sin(angle))
            base_polygon.append([x, y])
    
    # Демонстрация профиля Wide Expansion
    print("\n📊 Используется профиль: Wide Expansion")
    print("  • Верхние точки (0,1): +30%")
    print("  • Боковые точки (2,5): +35% (максимальное расширение)")
    print("  • Нижние точки (3,4): +20%")
    
    # Интерактивный выбор с предпросмотром
    result = integrate_with_existing_gui(
        image,
        base_polygon,
        video_name="demo",
        save_dir="./output"
    )
    
    if result:
        print("\n✓ Полигон успешно создан!")
        print(f"  Точки: {result}")
        
        # Визуализация финального результата
        final_vis = image.copy()
        cv2.polylines(
            final_vis,
            [np.array(result, np.int32)],
            True,
            (255, 0, 0),
            3
        )
        cv2.imshow("Final Result", final_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    else:
        print("\n⚠ Установка полигона отменена")