#!/usr/bin/env python3
"""
Скрипт для тестирования и настройки алгоритма детекции пламени
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Добавляем src в путь
sys.path.append('src')

try:
    from image_processing import generate_flame_mask, FlameDetector
except ImportError:
    print("Ошибка импорта. Убедитесь что файл image_processing.py находится в папке src/")
    sys.exit(1)


def test_single_frame(frame_path, roi_file=None, output_dir="test_output"):
    """Тестирование алгоритма на одном кадре"""
    print(f"Тестирование кадра: {frame_path}")

    # Загружаем кадр
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Не удалось загрузить кадр: {frame_path}")
        return

    # Загружаем ROI если указан
    polygon = None
    if roi_file and os.path.exists(roi_file):
        polygon = np.load(roi_file).tolist()
        print(f"Загружен ROI: {len(polygon)} точек")

    # Создаем выходную папку
    os.makedirs(output_dir, exist_ok=True)

    # Тестируем разные варианты
    test_variants = [
        {"name": "basic", "resize_to": (928, 576)},
        {"name": "high_res", "resize_to": (1280, 720)},
        {"name": "low_res", "resize_to": (640, 480)},
    ]

    results = {}

    for variant in test_variants:
        print(f"Тестирование варианта: {variant['name']}")

        mask, percent = generate_flame_mask(
            frame,
            polygon_vertices=polygon,
            resize_to=variant['resize_to']
        )

        results[variant['name']] = {
            'mask': mask,
            'percent': percent,
            'resize_to': variant['resize_to']
        }

        print(f"  Процент пламени: {percent:.2f}%")
        print(f"  Размер маски: {mask.shape}")
        print(f"  Пиксели пламени: {cv2.countNonZero(mask)}")

        # Сохраняем маску
        mask_path = os.path.join(output_dir, f"mask_{variant['name']}.png")
        cv2.imwrite(mask_path, mask)

        # Создаем визуализацию
        create_test_visualization(frame, mask, variant, polygon,
                                  os.path.join(output_dir, f"viz_{variant['name']}.jpg"))

    # Создаем сравнительную визуализацию
    create_comparison_view(frame, results, polygon,
                           os.path.join(output_dir, "comparison.jpg"))

    print(f"Результаты сохранены в: {output_dir}")
    return results


def create_test_visualization(frame, mask, variant, polygon, output_path):
    """Создание детальной визуализации для тестирования"""
    # Масштабируем кадр под маску для корректного отображения
    frame_resized = cv2.resize(frame, variant['resize_to'])

    # Создаем цветную маску
    mask_colored = np.zeros_like(frame_resized)
    mask_colored[:, :, 2] = mask  # Красный канал
    mask_colored[:, :, 1] = mask // 2  # Немного зеленого

    # Создаем overlay
    overlay = cv2.addWeighted(frame_resized, 0.7, mask_colored, 0.3, 0)

    # Рисуем ROI если есть
    if polygon:
        # Масштабируем координаты
        scale_x = variant['resize_to'][0] / frame.shape[1]
        scale_y = variant['resize_to'][1] / frame.shape[0]

        scaled_polygon = []
        for x, y in polygon:
            scaled_polygon.append((int(x * scale_x), int(y * scale_y)))

        pts = np.array(scaled_polygon, dtype=np.int32)
        cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)

    # Добавляем информацию
    info_texts = [
        f"Variant: {variant['name']}",
        f"Resolution: {variant['resize_to'][0]}x{variant['resize_to'][1]}",
        f"Flame: {(cv2.countNonZero(mask) / (mask.shape[0] * mask.shape[1])) * 100:.1f}%",
        f"Pixels: {cv2.countNonZero(mask)}",
    ]

    # Рисуем текст на темном фоне
    for i, text in enumerate(info_texts):
        y_pos = 30 + i * 25
        # Темный фон для текста
        cv2.rectangle(overlay, (5, y_pos - 20),
                      (400, y_pos + 5), (0, 0, 0), -1)
        cv2.putText(overlay, text, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imwrite(output_path, overlay)


def create_comparison_view(frame, results, polygon, output_path):
    """Создание сравнительного вида всех вариантов"""
    # Создаем сетку изображений
    images = []

    # Добавляем оригинальный кадр
    # Стандартный размер для сравнения
    orig_resized = cv2.resize(frame, (640, 480))
    cv2.putText(orig_resized, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    images.append(orig_resized)

    # Добавляем результаты для каждого варианта
    for name, result in results.items():
        mask = result['mask']
        percent = result['percent']

        # Приводим к стандартному размеру
        mask_resized = cv2.resize(mask, (640, 480))
        frame_resized = cv2.resize(frame, (640, 480))

        # Создаем overlay
        mask_colored = np.zeros_like(frame_resized)
        mask_colored[:, :, 2] = mask_resized
        mask_colored[:, :, 1] = mask_resized // 2

        overlay = cv2.addWeighted(frame_resized, 0.7, mask_colored, 0.3, 0)

        # Добавляем подпись
        cv2.putText(overlay, f"{name}: {percent:.1f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        images.append(overlay)

    # Компонуем в сетку 2x2
    if len(images) >= 4:
        top_row = np.hstack(images[:2])
        bottom_row = np.hstack(images[2:4])
        combined = np.vstack([top_row, bottom_row])
    else:
        combined = np.hstack(images)

    cv2.imwrite(output_path, combined)


def test_sequence(frames_dir, roi_file=None, output_dir="sequence_test", max_frames=20):
    """Тестирование алгоритма на последовательности кадров"""
    print(f"Тестирование последовательности кадров из: {frames_dir}")

    import glob
    frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))[:max_frames]

    if len(frames) == 0:
        print("Кадры не найдены")
        return

    # Загружаем ROI если указан
    polygon = None
    if roi_file and os.path.exists(roi_file):
        polygon = np.load(roi_file).tolist()

    os.makedirs(output_dir, exist_ok=True)

    # Инициализируем детектор с временной информацией
    detector = FlameDetector(history_size=5)

    results = []

    for i, frame_path in enumerate(frames):
        print(
            f"Обработка кадра {i+1}/{len(frames)}: {os.path.basename(frame_path)}")

        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        # Тестируем без временной информации
        mask_simple, percent_simple = generate_flame_mask(
            frame, polygon_vertices=polygon)

        # Тестируем с временной информацией
        mask_temporal, percent_temporal = detector.detect(
            frame, polygon_vertices=polygon)

        results.append({
            'frame': i,
            'path': frame_path,
            'simple_percent': percent_simple,
            'temporal_percent': percent_temporal,
            'mask_simple': mask_simple,
            'mask_temporal': mask_temporal
        })

        # Сохраняем визуализацию каждые 5 кадров
        if i % 5 == 0:
            create_sequence_comparison(frame, mask_simple, mask_temporal,
                                       percent_simple, percent_temporal, i,
                                       os.path.join(output_dir, f"frame_{i:03d}_comparison.jpg"))

    # Создаем график изменения процента пламени во времени
    create_temporal_analysis_plot(results, os.path.join(
        output_dir, "temporal_analysis.jpg"))

    # Сохраняем статистику
    save_sequence_statistics(
        results, os.path.join(output_dir, "statistics.txt"))

    print(
        f"Результаты тестирования последовательности сохранены в: {output_dir}")
    return results


def create_sequence_comparison(frame, mask_simple, mask_temporal,
                               percent_simple, percent_temporal, frame_idx, output_path):
    """Создание сравнения методов для кадра из последовательности"""
    # Приводим все к одному размеру
    target_size = (640, 480)
    frame_resized = cv2.resize(frame, target_size)
    mask_simple_resized = cv2.resize(mask_simple, target_size)
    mask_temporal_resized = cv2.resize(mask_temporal, target_size)

    # Создаем визуализации
    images = []

    # Оригинальный кадр
    orig = frame_resized.copy()
    cv2.putText(orig, f"Frame {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    images.append(orig)

    # Простой метод
    mask_colored_simple = np.zeros_like(frame_resized)
    mask_colored_simple[:, :, 2] = mask_simple_resized
    mask_colored_simple[:, :, 1] = mask_simple_resized // 2
    overlay_simple = cv2.addWeighted(
        frame_resized, 0.7, mask_colored_simple, 0.3, 0)
    cv2.putText(overlay_simple, f"Simple: {percent_simple:.1f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    images.append(overlay_simple)

    # Временной метод
    mask_colored_temporal = np.zeros_like(frame_resized)
    mask_colored_temporal[:, :, 2] = mask_temporal_resized
    mask_colored_temporal[:, :, 1] = mask_temporal_resized // 2
    overlay_temporal = cv2.addWeighted(
        frame_resized, 0.7, mask_colored_temporal, 0.3, 0)
    cv2.putText(overlay_temporal, f"Temporal: {percent_temporal:.1f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    images.append(overlay_temporal)

    # Разность масок
    diff = cv2.absdiff(mask_simple_resized, mask_temporal_resized)
    diff_colored = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
    cv2.putText(diff_colored, f"Difference", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    images.append(diff_colored)

    # Компонуем в сетку 2x2
    top_row = np.hstack(images[:2])
    bottom_row = np.hstack(images[2:])
    combined = np.vstack([top_row, bottom_row])

    cv2.imwrite(output_path, combined)


def create_temporal_analysis_plot(results, output_path):
    """Создание графика временного анализа (без matplotlib, используем OpenCV)"""
    if len(results) == 0:
        return

    # Размеры графика
    width, height = 800, 400
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Данные для графика
    simple_percents = [r['simple_percent'] for r in results]
    temporal_percents = [r['temporal_percent'] for r in results]

    max_percent = max(max(simple_percents), max(temporal_percents))
    if max_percent == 0:
        max_percent = 1

    # Масштабируем данные
    margin = 50
    graph_width = width - 2 * margin
    graph_height = height - 2 * margin

    # Рисуем оси
    cv2.line(img, (margin, height - margin),
             (width - margin, height - margin), (0, 0, 0), 2)  # X
    cv2.line(img, (margin, margin),
             (margin, height - margin), (0, 0, 0), 2)  # Y

    # Подписи осей
    cv2.putText(img, "Frame", (width//2 - 30, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(img, "Flame %", (5, height//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Рисуем сетку
    for i in range(1, 5):
        y = margin + (graph_height * i) // 5
        cv2.line(img, (margin, y), (width - margin, y), (200, 200, 200), 1)
        cv2.putText(img, f"{max_percent * (5-i) / 5:.1f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Рисуем данные
    if len(results) > 1:
        for i in range(len(results) - 1):
            # Координаты точек
            x1 = margin + (i * graph_width) // (len(results) - 1)
            x2 = margin + ((i + 1) * graph_width) // (len(results) - 1)

            # Simple method - синий
            y1_simple = height - margin - \
                int((simple_percents[i] / max_percent) * graph_height)
            y2_simple = height - margin - \
                int((simple_percents[i + 1] / max_percent) * graph_height)
            cv2.line(img, (x1, y1_simple), (x2, y2_simple), (255, 0, 0), 2)

            # Temporal method - красный
            y1_temporal = height - margin - \
                int((temporal_percents[i] / max_percent) * graph_height)
            y2_temporal = height - margin - \
                int((temporal_percents[i + 1] / max_percent) * graph_height)
            cv2.line(img, (x1, y1_temporal), (x2, y2_temporal), (0, 0, 255), 2)

    # Легенда
    cv2.line(img, (width - 150, 30), (width - 120, 30), (255, 0, 0), 2)
    cv2.putText(img, "Simple", (width - 115, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.line(img, (width - 150, 50), (width - 120, 50), (0, 0, 255), 2)
    cv2.putText(img, "Temporal", (width - 115, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Заголовок
    cv2.putText(img, "Flame Detection Comparison", (width//2 - 120, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.imwrite(output_path, img)


def save_sequence_statistics(results, output_path):
    """Сохранение статистики тестирования последовательности"""
    if len(results) == 0:
        return

    simple_percents = [r['simple_percent'] for r in results]
    temporal_percents = [r['temporal_percent'] for r in results]

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("СТАТИСТИКА ТЕСТИРОВАНИЯ ДЕТЕКЦИИ ПЛАМЕНИ\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Количество кадров: {len(results)}\n\n")

        # Статистика простого метода
        f.write("ПРОСТОЙ МЕТОД:\n")
        f.write(f"- Среднее: {np.mean(simple_percents):.2f}%\n")
        f.write(f"- Медиана: {np.median(simple_percents):.2f}%\n")
        f.write(f"- Минимум: {min(simple_percents):.2f}%\n")
        f.write(f"- Максимум: {max(simple_percents):.2f}%\n")
        f.write(f"- Ст. отклонение: {np.std(simple_percents):.2f}%\n\n")

        # Статистика временного метода
        f.write("ВРЕМЕННОЙ МЕТОД:\n")
        f.write(f"- Среднее: {np.mean(temporal_percents):.2f}%\n")
        f.write(f"- Медиана: {np.median(temporal_percents):.2f}%\n")
        f.write(f"- Минимум: {min(temporal_percents):.2f}%\n")
        f.write(f"- Максимум: {max(temporal_percents):.2f}%\n")
        f.write(f"- Ст. отклонение: {np.std(temporal_percents):.2f}%\n\n")

        # Сравнение методов
        differences = [abs(s - t)
                       for s, t in zip(simple_percents, temporal_percents)]
        f.write("СРАВНЕНИЕ МЕТОДОВ:\n")
        f.write(f"- Средняя разность: {np.mean(differences):.2f}%\n")
        f.write(f"- Макс. разность: {max(differences):.2f}%\n")
        f.write(
            f"- Корреляция: {np.corrcoef(simple_percents, temporal_percents)[0, 1]:.3f}\n\n")

        # Детальные данные
        f.write("ДЕТАЛЬНЫЕ ДАННЫЕ:\n")
        f.write("Frame | Simple% | Temporal% | Diff% | Filename\n")
        f.write("-" * 60 + "\n")

        for r in results:
            diff = abs(r['simple_percent'] - r['temporal_percent'])
            f.write(f"{r['frame']:5d} | "
                    f"{r['simple_percent']:7.1f} | "
                    f"{r['temporal_percent']:9.1f} | "
                    f"{diff:5.1f} | "
                    f"{os.path.basename(r['path'])}\n")


def interactive_parameter_tuning(frame_path, roi_file=None):
    """Интерактивная настройка параметров детекции пламени"""
    print("Интерактивная настройка параметров (нажмите ESC для выхода)")

    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Не удалось загрузить кадр: {frame_path}")
        return

    # Загружаем ROI если есть
    polygon = None
    if roi_file and os.path.exists(roi_file):
        polygon = np.load(roi_file).tolist()

    # Создаем окна
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Overlay', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Parameters', cv2.WINDOW_NORMAL)

    # Создаем трекбары для настройки параметров
    # (Это упрощенная версия - в реальности нужно модифицировать generate_flame_mask)
    cv2.createTrackbar('Hue Low', 'Parameters', 0, 30, lambda x: None)
    cv2.createTrackbar('Hue High', 'Parameters', 30, 180, lambda x: None)
    cv2.createTrackbar('Sat Low', 'Parameters', 100, 255, lambda x: None)
    cv2.createTrackbar('Val Low', 'Parameters', 150, 255, lambda x: None)
    cv2.createTrackbar('Area Min', 'Parameters', 200, 2000, lambda x: None)

    while True:
        # Получаем значения параметров
        hue_low = cv2.getTrackbarPos('Hue Low', 'Parameters')
        hue_high = cv2.getTrackbarPos('Hue High', 'Parameters')
        sat_low = cv2.getTrackbarPos('Sat Low', 'Parameters')
        val_low = cv2.getTrackbarPos('Val Low', 'Parameters')
        area_min = cv2.getTrackbarPos('Area Min', 'Parameters')

        # Применяем детекцию (упрощенная версия)
        mask, percent = generate_flame_mask(frame, polygon_vertices=polygon)

        # Отображаем результаты
        frame_display = cv2.resize(frame, (640, 480))
        mask_display = cv2.resize(mask, (640, 480))

        # Создаем overlay
        mask_colored = np.zeros_like(frame_display)
        mask_colored[:, :, 2] = mask_display
        mask_colored[:, :, 1] = mask_display // 2
        overlay = cv2.addWeighted(frame_display, 0.7, mask_colored, 0.3, 0)

        # Добавляем информацию
        cv2.putText(overlay, f"Flame: {percent:.1f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(overlay, f"Pixels: {cv2.countNonZero(mask)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Показываем изображения
        cv2.imshow('Original', frame_display)
        cv2.imshow('Mask', mask_display)
        cv2.imshow('Overlay', overlay)

        # Проверяем нажатия клавиш
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s'):  # Сохранить текущий результат
            cv2.imwrite('tuned_mask.png', mask)
            cv2.imwrite('tuned_overlay.jpg', overlay)
            print("Результат сохранен: tuned_mask.png, tuned_overlay.jpg")

    cv2.destroyAllWindows()


def main():
    """Основная функция для запуска тестов"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Тестирование алгоритма детекции пламени')
    parser.add_argument('--mode', choices=['single', 'sequence', 'interactive'],
                        default='single', help='Режим тестирования')
    parser.add_argument('--input', required=True,
                        help='Путь к кадру (single) или папке с кадрами (sequence)')
    parser.add_argument('--roi', help='Файл с ROI (опционально)')
    parser.add_argument('--output', default='test_output',
                        help='Папка для результатов')
    parser.add_argument('--max_frames', type=int, default=20,
                        help='Макс. количество кадров для тестирования последовательности')

    args = parser.parse_args()

    if args.mode == 'single':
        test_single_frame(args.input, args.roi, args.output)
    elif args.mode == 'sequence':
        test_sequence(args.input, args.roi, args.output, args.max_frames)
    elif args.mode == 'interactive':
        interactive_parameter_tuning(args.input, args.roi)


if __name__ == "__main__":
    # Если запускается без аргументов, используем значения по умолчанию
    if len(sys.argv) == 1:
        print("Запуск тестирования с параметрами по умолчанию...")
        print("Для настройки параметров используйте: python test_flame_detection.py --help")

        # Проверяем наличие тестовых данных
        if os.path.exists("data/frames/1video"):
            test_sequence(
                "data/frames/1video", "data/roi_1video.npy" if os.path.exists("data/roi_1video.npy") else None)
        elif os.path.exists("data/videos"):
            print("Сначала извлеките кадры из видео с помощью extract_frames.py")
        else:
            print(
                "Тестовые данные не найдены. Убедитесь что папка data/frames/1video существует")
    else:
        main()
