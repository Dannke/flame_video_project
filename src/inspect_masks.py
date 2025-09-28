import cv2
import glob
import os
import numpy as np


def create_flame_overlay(frame, mask, style='fire'):
    """
    Создание overlay с разными стилями визуализации пламени

    Args:
        frame: исходный кадр
        mask: маска пламени
        style: стиль визуализации ('fire', 'heat', 'simple', 'contour')
    """
    if style == 'fire':
        # Огненный стиль - красно-оранжевые цвета
        mask_colored = np.zeros_like(frame)
        mask_colored[:, :, 2] = mask  # Красный канал
        mask_colored[:, :, 1] = mask // 2  # Зеленый для оранжевого оттенка
        overlay = cv2.addWeighted(frame, 0.6, mask_colored, 0.4, 0)

    elif style == 'heat':
        # Тепловая карта
        mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
        overlay = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)

    elif style == 'simple':
        # Простая полупрозрачная маска
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)

    elif style == 'contour':
        # Только контуры пламени
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = frame.copy()
        cv2.drawContours(overlay, contours, -1,
                         (0, 0, 255), 3)  # Красные контуры

        # Добавляем заливку с низкой прозрачностью
        mask_colored = np.zeros_like(frame)
        mask_colored[:, :, 2] = mask
        overlay = cv2.addWeighted(overlay, 0.9, mask_colored, 0.1, 0)

    else:
        # По умолчанию fire стиль
        return create_flame_overlay(frame, mask, 'fire')

    return overlay


def add_statistics_text(image, mask, frame_idx, additional_info=None):
    """Добавление статистической информации на изображение"""
    # Вычисляем статистики
    total_pixels = mask.shape[0] * mask.shape[1]
    flame_pixels = cv2.countNonZero(mask)
    flame_percent = (flame_pixels / total_pixels) * 100

    # Анализ контуров
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_regions = len(contours)

    # Средний размер области пламени
    if num_regions > 0:
        areas = [cv2.contourArea(c) for c in contours]
        avg_area = np.mean(areas)
        max_area = max(areas)
    else:
        avg_area = 0
        max_area = 0

    # Подготавливаем текст
    texts = [
        f"Frame: {frame_idx}",
        f"Flame: {flame_percent:.1f}%",
        f"Pixels: {flame_pixels}",
        f"Regions: {num_regions}",
        f"Avg area: {avg_area:.0f}",
        f"Max area: {max_area:.0f}"
    ]

    if additional_info:
        texts.extend(additional_info)

    # Рисуем полупрозрачный фон для текста
    text_bg = np.zeros((len(texts) * 35 + 20, 300, 3), dtype=np.uint8)
    text_bg[:] = (0, 0, 0)

    # Добавляем текст
    for i, text in enumerate(texts):
        cv2.putText(text_bg, text, (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Накладываем текстовый фон на изображение
    h, w = text_bg.shape[:2]
    if h < image.shape[0] and w < image.shape[1]:
        roi = image[0:h, 0:w]
        text_bg_gray = cv2.cvtColor(text_bg, cv2.COLOR_BGR2GRAY)
        _, mask_text = cv2.threshold(text_bg_gray, 1, 255, cv2.THRESH_BINARY)

        # Полупрозрачное наложение
        alpha = 0.7
        for c in range(3):
            roi[:, :, c] = roi[:, :, c] * \
                (1 - alpha) + text_bg[:, :, c] * alpha

    return image


def inspect_flame_masks(frames_dir, masks_dir, output_dir="data/flame_preview",
                        max_images=50, styles=['fire', 'heat'], save_individual=True):
    """
    Улучшенная инспекция масок пламени с множественными стилями визуализации

    Args:
        frames_dir: папка с кадрами
        masks_dir: папка с масками
        output_dir: папка для сохранения результатов
        max_images: максимальное количество изображений для обработки
        styles: список стилей визуализации
        save_individual: сохранять ли отдельные изображения для каждого стиля
    """
    frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    masks = sorted(glob.glob(os.path.join(masks_dir, "*.png")))

    if len(frames) == 0:
        print(f"Кадры не найдены в {frames_dir}")
        return

    if len(masks) == 0:
        print(f"Маски не найдены в {masks_dir}")
        return

    print(f"Найдено {len(frames)} кадров и {len(masks)} масок")

    os.makedirs(output_dir, exist_ok=True)

    # Создаем подпапки для каждого стиля если нужно
    if save_individual:
        for style in styles:
            os.makedirs(os.path.join(output_dir, style), exist_ok=True)

    # Статистика для анализа
    flame_percentages = []

    process_count = min(max_images, len(frames), len(masks))

    for i in range(process_count):
        print(
            f"Обработка {i+1}/{process_count}: {os.path.basename(frames[i])}")

        # Загружаем кадр и маску
        frame = cv2.imread(frames[i])
        mask = cv2.imread(masks[i], cv2.IMREAD_GRAYSCALE)

        if frame is None or mask is None:
            print(f"Ошибка загрузки файлов для индекса {i}")
            continue

        # Проверяем и корректируем размеры
        if frame.shape[:2] != mask.shape:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        # Вычисляем статистику
        flame_percent = (cv2.countNonZero(mask) /
                         (mask.shape[0] * mask.shape[1])) * 100
        flame_percentages.append(flame_percent)

        # Создаем визуализации для каждого стиля
        for style in styles:
            overlay = create_flame_overlay(frame, mask, style)

            # Добавляем статистическую информацию
            overlay_with_stats = add_statistics_text(overlay, mask, i,
                                                     [f"Style: {style}"])

            # Сохраняем
            if save_individual:
                filename = f"flame_{style}_{i:04d}.jpg"
                filepath = os.path.join(output_dir, style, filename)
            else:
                filename = f"flame_{style}_{i:04d}.jpg"
                filepath = os.path.join(output_dir, filename)

            cv2.imwrite(filepath, overlay_with_stats)

        # Создаем комбинированное изображение со всеми стилями
        if len(styles) > 1:
            create_combined_view(frame, mask, i, styles,
                                 os.path.join(output_dir, f"combined_{i:04d}.jpg"))

    # Выводим итоговую статистику
    if flame_percentages:
        print(f"\n{'='*50}")
        print("СТАТИСТИКА ИНСПЕКЦИИ МАСОК ПЛАМЕНИ:")
        print(f"Обработано изображений: {len(flame_percentages)}")
        print(f"Средний процент пламени: {np.mean(flame_percentages):.2f}%")
        print(f"Медиана: {np.median(flame_percentages):.2f}%")
        print(f"Минимум: {min(flame_percentages):.2f}%")
        print(f"Максимум: {max(flame_percentages):.2f}%")
        print(f"Стандартное отклонение: {np.std(flame_percentages):.2f}%")

        # Гистограмма распределения
        print(f"\nРаспределение по интервалам:")
        intervals = [(0, 1), (1, 5), (5, 15), (15, 30), (30, 100)]
        for low, high in intervals:
            count = sum(1 for x in flame_percentages if low <= x < high)
            pct = (count / len(flame_percentages)) * 100
            print(f"  {low:2d}-{high:2d}%: {count:3d} изображений ({pct:5.1f}%)")

        print(f"Результаты сохранены в: {output_dir}")
        print(f"{'='*50}")


def create_combined_view(frame, mask, frame_idx, styles, output_path):
    """Создание комбинированного вида со всеми стилями визуализации"""
    overlays = []

    # Создаем оригинальный кадр с подписью
    orig_with_text = frame.copy()
    cv2.putText(orig_with_text, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    overlays.append(orig_with_text)

    # Создаем маску с подписью
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(mask_colored, "Mask", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    overlays.append(mask_colored)

    # Создаем overlay для каждого стиля
    for style in styles:
        overlay = create_flame_overlay(frame, mask, style)
        cv2.putText(overlay, style.capitalize(), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        overlays.append(overlay)

    # Компонуем изображения в сетку
    if len(overlays) <= 2:
        combined = np.hstack(overlays)
    elif len(overlays) <= 4:
        if len(overlays) == 3:
            # Заполняем пустое место
            overlays.append(np.zeros_like(overlays[0]))
        top_row = np.hstack(overlays[:2])
        bottom_row = np.hstack(overlays[2:4])
        combined = np.vstack([top_row, bottom_row])
    else:
        # Более 4 изображений - делаем прямоугольную сетку
        rows = []
        cols = 3
        for i in range(0, len(overlays), cols):
            row_images = overlays[i:i+cols]
            # Дополняем строку пустыми изображениями если нужно
            while len(row_images) < cols:
                row_images.append(np.zeros_like(overlays[0]))
            rows.append(np.hstack(row_images))
        combined = np.vstack(rows)

    # Добавляем общую информацию
    flame_percent = (cv2.countNonZero(mask) /
                     (mask.shape[0] * mask.shape[1])) * 100
    info_text = f"Frame {frame_idx}: Flame {flame_percent:.1f}%"
    cv2.putText(combined, info_text, (10, combined.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imwrite(output_path, combined)


def create_flame_analysis_report(masks_dir, output_file="flame_analysis_report.txt"):
    """Создание подробного отчета об анализе пламени"""
    masks = sorted(glob.glob(os.path.join(masks_dir, "*.png")))

    if len(masks) == 0:
        print("Маски не найдены для анализа")
        return

    print(f"Анализ {len(masks)} масок пламени...")

    analysis_data = []

    for i, mask_path in enumerate(masks):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        # Базовые статистики
        total_pixels = mask.shape[0] * mask.shape[1]
        flame_pixels = cv2.countNonZero(mask)
        flame_percent = (flame_pixels / total_pixels) * 100

        # Анализ контуров
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_regions = len(contours)

        if num_regions > 0:
            areas = [cv2.contourArea(c) for c in contours]
            perimeters = [cv2.arcLength(c, True) for c in contours]

            total_area = sum(areas)
            avg_area = np.mean(areas)
            max_area = max(areas)

            # Коэффициенты формы
            circularities = []
            for j, contour in enumerate(contours):
                if perimeters[j] > 0:
                    circularity = 4 * np.pi * areas[j] / (perimeters[j] ** 2)
                    circularities.append(circularity)

            avg_circularity = np.mean(circularities) if circularities else 0

        else:
            total_area = 0
            avg_area = 0
            max_area = 0
            avg_circularity = 0

        analysis_data.append({
            'frame': i,
            'flame_percent': flame_percent,
            'flame_pixels': flame_pixels,
            'num_regions': num_regions,
            'total_area': total_area,
            'avg_area': avg_area,
            'max_area': max_area,
            'avg_circularity': avg_circularity
        })

    # Сохраняем отчет
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("ОТЧЕТ ОБ АНАЛИЗЕ ДЕТЕКЦИИ ПЛАМЕНИ\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Общая информация:\n")
        f.write(f"- Проанализировано масок: {len(analysis_data)}\n")
        f.write(f"- Размер изображения: {mask.shape}\n\n")

        # Статистики по проценту пламени
        flame_percents = [d['flame_percent'] for d in analysis_data]
        f.write(f"Статистики по проценту пламени:\n")
        f.write(f"- Среднее: {np.mean(flame_percents):.2f}%\n")
        f.write(f"- Медиана: {np.median(flame_percents):.2f}%\n")
        f.write(f"- Минимум: {min(flame_percents):.2f}%\n")
        f.write(f"- Максимум: {max(flame_percents):.2f}%\n")
        f.write(f"- Ст. отклонение: {np.std(flame_percents):.2f}%\n\n")

        # Распределение по интервалам
        f.write("Распределение по интервалам:\n")
        intervals = [(0, 1), (1, 5), (5, 15), (15, 30), (30, 100)]
        for low, high in intervals:
            count = sum(1 for x in flame_percents if low <= x < high)
            pct = (count / len(flame_percents)) * 100
            f.write(f"- {low:2d}-{high:2d}%: {count:3d} кадров ({pct:5.1f}%)\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("ДЕТАЛЬНЫЕ ДАННЫЕ ПО КАДРАМ:\n\n")

        f.write("Frame | Flame% | Pixels | Regions | AvgArea | MaxArea | Circularity\n")
        f.write("-" * 70 + "\n")

        for data in analysis_data:
            f.write(f"{data['frame']:5d} | "
                    f"{data['flame_percent']:6.1f} | "
                    f"{data['flame_pixels']:6d} | "
                    f"{data['num_regions']:7d} | "
                    f"{data['avg_area']:7.0f} | "
                    f"{data['max_area']:7.0f} | "
                    f"{data['avg_circularity']:11.3f}\n")

    print(f"Отчет сохранен в файл: {output_file}")

# Основная функция для запуска инспекции


def main():
    frames_dir = "data/frames/1video"
    masks_dir = "data/masks/1video"
    output_dir = "data/flame_preview"

    # Проверяем существование папок
    if not os.path.exists(frames_dir):
        print(f"Папка с кадрами не найдена: {frames_dir}")
        return

    if not os.path.exists(masks_dir):
        print(f"Папка с масками не найдена: {masks_dir}")
        return

    print("Запуск инспекции масок пламени...")

    # Запускаем инспекцию с разными стилями
    inspect_flame_masks(
        frames_dir=frames_dir,
        masks_dir=masks_dir,
        output_dir=output_dir,
        max_images=50,
        styles=['fire', 'heat', 'contour'],
        save_individual=True
    )

    # Создаем аналитический отчет
    report_file = os.path.join(output_dir, "flame_analysis_report.txt")
    create_flame_analysis_report(masks_dir, report_file)

    print("Инспекция завершена!")


if __name__ == "__main__":
    main()
