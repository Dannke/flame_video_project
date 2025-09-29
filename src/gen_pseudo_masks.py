import os
import glob
import cv2
import numpy as np
import traceback
from collections import deque

try:
    # пытаемся импортировать улучшённую функцию, если она есть
    from image_processing import generate_flame_mask_improved, apply_perspective_transform
except Exception:
    try:
        from .image_processing import generate_flame_mask_improved, apply_perspective_transform
    except Exception:
        import sys
        sys.path.append(os.path.dirname(__file__))
        from image_processing import generate_flame_mask_improved, apply_perspective_transform

RESIZE_TO = (928, 576)


def gen_masks_temporal(frames_dir, masks_out_dir, polygon=None,
                       use_homography=False, src_pts=None, use_roi=None,
                       # Новые параметры для настройки
                       buffer_size=10,          # Размер буфера кадров
                       brightness_thresh=210,    # Порог яркости (210-240)
                       saturation_thresh=110,    # Порог насыщенности (100-140)
                       flicker_weight=0.3,       # Вес мерцания (0-0.5)
                       min_flicker_frames=5,     # Минимум кадров для анализа мерцания
                       use_color_filter=True,    # Использовать цветовой фильтр
                       confidence_threshold=20.0,  # Минимальная уверенность для "пламени"
                       flame_percent_threshold=0.2,  # Минимальный процент для "пламени"
                       negative_keep_ratio=0.05):   # Доля % кадров без пламени
    """
    Генерация масок пламени с улучшенным временным анализом

    Args:
        frames_dir: папка с кадрами (.jpg)
        masks_out_dir: куда сохранять .png маски
        polygon: список вершин ROI или None
        use_homography: применять ли перспективное преобразование
        src_pts: точки для гомографии (если нужно)
        use_roi: принудительно включить/выключить ROI (None - авто)
        buffer_size: размер буфера для хранения предыдущих кадров (5-20)
        brightness_thresh: порог яркости для детекции (210-240)
        saturation_thresh: порог насыщенности (100-140)
        flicker_weight: вес компонента мерцания (0-0.5)
        min_flicker_frames: минимум кадров для анализа мерцания (3-10)
        use_color_filter: использовать ли строгую цветовую фильтрацию
        confidence_threshold: минимальная уверенность для классификации как пламя
        flame_percent_threshold: минимальный процент площади для пламени
    """
    try:
        os.makedirs(masks_out_dir, exist_ok=True)
        frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
        if len(frames) == 0:
            print("No frames found in", frames_dir)
            return
        
        # --- вычислим лимит на negative (no_flame) кадры ---
        max_no_flame_to_keep = max(
            0, int(len(frames) * float(negative_keep_ratio)))
        kept_no_flame = 0
        processed_no_flame = 0

        if use_roi is None:
            use_roi = bool(polygon) and len(polygon) > 2
        else:
            use_roi = bool(use_roi)

        print(f"Найдено {len(frames)} кадров для обработки")
        print(f"Параметры детекции:")
        print(f"  - Размер буфера: {buffer_size} кадров")
        print(f"  - Порог яркости: {brightness_thresh}")
        print(f"  - Порог насыщенности: {saturation_thresh}")
        print(f"  - Вес мерцания: {flicker_weight}")
        print(f"  - Мин. кадров для мерцания: {min_flicker_frames}")
        print(f"  - Цветовой фильтр: {'Да' if use_color_filter else 'Нет'}")
        print(f"  - ROI: {'задан' if use_roi else 'не используется'}")
        print(f"  - Порог уверенности: {confidence_threshold}%")
        print(f"  - Порог площади пламени: {flame_percent_threshold}%")
        print(
            f"  - negative_keep_ratio: {negative_keep_ratio} -> max_no_flame_to_keep={max_no_flame_to_keep}")

        # Инициализация буфера кадров
        frame_buffer = deque(maxlen=buffer_size)

        # Статистические переменные
        total_flame_percent = 0.0
        total_confidence = 0.0
        max_flame_percent = 0.0
        min_flame_percent = 100.0
        frames_with_flame = 0
        frames_without_flame = 0
        uncertain_frames = 0

        flame_percentages = []
        confidences = []
        statuses = []

        for i, frame_path in enumerate(frames):
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Не удалось загрузить кадр: {frame_path}")
                continue

            if use_homography and src_pts is not None:
                frame = apply_perspective_transform(frame, src_pts, RESIZE_TO)

            # Добавляем кадр в буфер
            frame_buffer.append(frame.copy())

            # Генерация маски с использованием буфера
            mask, flame_percent, confidence = generate_flame_mask_improved(
                frame,
                frame_buffer=list(frame_buffer),  # Передаем копию буфера
                polygon_vertices=polygon if use_roi else None,
                resize_to=RESIZE_TO,
                use_roi=use_roi,
                brightness_thresh=brightness_thresh,
                saturation_thresh=saturation_thresh,
                flicker_weight=flicker_weight if len(
                    frame_buffer) >= min_flicker_frames else 0,
                min_flicker_frames=min_flicker_frames,
                use_color_filter=use_color_filter
            )

            # Классификация с настраиваемыми порогами
            if confidence >= confidence_threshold and flame_percent > flame_percent_threshold:
                flame_status = "flame"
            elif confidence < confidence_threshold/2 or flame_percent < flame_percent_threshold/2:
                flame_status = "no_flame"
            else:
                flame_status = "uncertain"

           # Сохранение маски — но применяем ограничение на no_flame
            save_mask = True
            if flame_status == "no_flame":
                processed_no_flame += 1
                if kept_no_flame < max_no_flame_to_keep:
                    kept_no_flame += 1
                    save_mask = True
                else:
                    # Превышен лимит negative — не сохраняем маску и удаляем кадр,
                    # чтобы не засорять датасет лишними negative-кадрами.
                    save_mask = False

            if save_mask:
                mask_filename = f"mask_{i:06d}.png"
                mask_path = os.path.join(masks_out_dir, mask_filename)
                cv2.imwrite(mask_path, mask)
            else:
                # удаляем исходный кадр (либо можно переместить в отдельную папку)
                try:
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
                except Exception:
                    pass

            # Обновление статистики
            total_flame_percent += flame_percent
            total_confidence += confidence
            max_flame_percent = max(max_flame_percent, flame_percent)
            min_flame_percent = min(
                min_flame_percent, flame_percent) if flame_percent > 0 else min_flame_percent

            flame_percentages.append(flame_percent)
            confidences.append(confidence)
            statuses.append(flame_status)

            if flame_status == "flame":
                frames_with_flame += 1
            elif flame_status == "no_flame":
                frames_without_flame += 1
            else:
                uncertain_frames += 1

            # Вывод прогресса
            if i % 50 == 0 or i == len(frames) - 1:
                flame_pixels = cv2.countNonZero(mask)
                buffer_status = f"Буфер: {len(frame_buffer)}/{buffer_size}"
                print(f"Кадр {i+1}/{len(frames)} | {buffer_status} | "
                      f"Пламя: {flame_percent:.2f}% | "
                      f"Уверенность: {confidence:.1f}% | "
                      f"Пиксели: {flame_pixels} | "
                      f"Статус: {flame_status}")

        # Вычисление итоговой статистики
        n = len(flame_percentages) if len(flame_percentages) > 0 else 1
        avg_flame_percent = total_flame_percent / n
        avg_confidence = total_confidence / n
        flame_coverage = (frames_with_flame / n) * 100
        no_flame_coverage = (frames_without_flame / n) * 100
        uncertain_coverage = (uncertain_frames / n) * 100

        print("\n" + "=" * 70)
        print("СТАТИСТИКА ДЕТЕКЦИИ ПЛАМЕНИ (С ВРЕМЕННЫМ АНАЛИЗОМ):")
        print(f"Обработано кадров: {len(frames)}")
        print(
            f"  Кадры с пламенем: {frames_with_flame} ({flame_coverage:.1f}%)")
        print(
            f"  Кадры без пламени: {frames_without_flame} ({no_flame_coverage:.1f}%)")
        print(
            f"  Неопределенные кадры: {uncertain_frames} ({uncertain_coverage:.1f}%)")
        print(f"Средний процент пламени: {avg_flame_percent:.2f}%")
        print(f"Средняя уверенность: {avg_confidence:.1f}%")
        print(f"Максимальный процент: {max_flame_percent:.2f}%")
        print(f"Минимальный процент: {min_flame_percent:.2f}%")

        # Распределение по интервалам
        intervals = [(0, 0.1), (0.1, 1), (1, 5), (5, 15), (15, 100)]
        print("\nРАСПРЕДЕЛЕНИЕ ПО ИНТЕРВАЛАМ ПЛАМЕНИ:")
        for low, high in intervals:
            count = sum(1 for x in flame_percentages if low <= x < high)
            percentage = (count / n) * 100
            print(
                f"  {low:.1f}-{high:.1f}%: {count:4d} кадров ({percentage:5.1f}%)")

        conf_intervals = [(0, 10), (10, 25), (25, 50), (50, 75), (75, 100)]
        print("\nРАСПРЕДЕЛЕНИЕ ПО УВЕРЕННОСТИ:")
        for low, high in conf_intervals:
            count = sum(1 for x in confidences if low <= x < high)
            percentage = (count / n) * 100
            print(f"  {low:2d}-{high:2d}%: {count:4d} кадров ({percentage:5.1f}%)")

        print(f"\nМаски сохранены в: {masks_out_dir}")
        print("=" * 70)

        # Сохранение детальной статистики
        save_detailed_statistics_with_params(
            masks_out_dir, flame_percentages, confidences, statuses, frames,
            {
                'buffer_size': buffer_size,
                'brightness_thresh': brightness_thresh,
                'saturation_thresh': saturation_thresh,
                'flicker_weight': flicker_weight,
                'min_flicker_frames': min_flicker_frames,
                'use_color_filter': use_color_filter,
                'confidence_threshold': confidence_threshold,
                'flame_percent_threshold': flame_percent_threshold,
                'negative_keep_ratio': negative_keep_ratio
            }
        )

    except Exception as exc:
        print("Ошибка в gen_masks_temporal:", exc)
        traceback.print_exc()


def save_detailed_statistics_with_params(output_dir, flame_percentages, confidences,
                                         statuses, frame_paths, params):
    """Сохранение детальной статистики с параметрами детекции"""
    stats_file = os.path.join(output_dir, "detection_statistics.txt")
    try:
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("ДЕТАЛЬНАЯ СТАТИСТИКА ДЕТЕКЦИИ ПЛАМЕНИ\n")
            f.write("=" * 60 + "\n\n")

            # Сохраняем параметры детекции
            f.write("ПАРАМЕТРЫ ДЕТЕКЦИИ:\n")
            f.write("-" * 60 + "\n")
            for key, value in params.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

            f.write("РЕЗУЛЬТАТЫ ПО КАДРАМ:\n")
            f.write("-" * 60 + "\n")
            f.write("Frame | Flame% | Confidence% | Status | Filename\n")
            f.write("-" * 60 + "\n")
            for i, (flame_pct, conf, status) in enumerate(zip(flame_percentages, confidences, statuses)):
                filename = os.path.basename(frame_paths[i]) if i < len(
                    frame_paths) else f"frame_{i:06d}.jpg"
                if status == 'flame':
                    status_symbol = 'FLAME'
                elif status == 'no_flame':
                    status_symbol = 'CLEAR'
                else:
                    status_symbol = '?????'
                f.write(
                    f"{i:5d} | {flame_pct:6.2f} | {conf:11.1f} | {status:8s} {status_symbol} | {filename}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("СВОДНАЯ СТАТИСТИКА:\n")
            f.write(f"Всего кадров: {len(flame_percentages)}\n")
            f.write(
                f"С пламенем: {sum(1 for s in statuses if s == 'flame')}\n")
            f.write(
                f"Без пламени: {sum(1 for s in statuses if s == 'no_flame')}\n")
            f.write(
                f"Неопределенные: {sum(1 for s in statuses if s == 'uncertain')}\n")
            if flame_percentages:
                f.write(
                    f"Средний процент пламени: {np.mean(flame_percentages):.2f}%\n")
                f.write(f"Средняя уверенность: {np.mean(confidences):.2f}%\n")
                f.write(
                    f"Стандартное отклонение: {np.std(flame_percentages):.2f}%\n")
        print(f"Детальная статистика сохранена в: {stats_file}")
    except Exception as e:
        print("Ошибка при сохранении статистики:", e)
        traceback.print_exc()


# Для обратной совместимости оставляем старую функцию
def gen_masks(frames_dir, masks_out_dir, polygon=None,
              use_homography=False, src_pts=None, use_roi=None):
    """
    Старая функция для обратной совместимости - перенаправляет на новую
    """
    print("Используется улучшенная версия с временным анализом")
    return gen_masks_temporal(
        frames_dir=frames_dir,
        masks_out_dir=masks_out_dir,
        polygon=polygon,
        use_homography=use_homography,
        src_pts=src_pts,
        use_roi=use_roi,
        buffer_size=5,  # Умеренный буфер по умолчанию
        brightness_thresh=220,
        saturation_thresh=120,
        flicker_weight=0.3,
        min_flicker_frames=3,
        use_color_filter=True
    )


def save_detailed_statistics(output_dir, flame_percentages, confidences, statuses, frame_paths):
    """Сохранение детальной статистики в файл"""
    stats_file = os.path.join(output_dir, "detection_statistics.txt")
    try:
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("ДЕТАЛЬНАЯ СТАТИСТИКА ДЕТЕКЦИИ ПЛАМЕНИ\n")
            f.write("=" * 60 + "\n\n")
            f.write("Frame | Flame% | Confidence% | Status | Filename\n")
            f.write("-" * 60 + "\n")
            for i, (flame_pct, conf, status) in enumerate(zip(flame_percentages, confidences, statuses)):
                filename = os.path.basename(frame_paths[i]) if i < len(
                    frame_paths) else f"frame_{i:06d}.jpg"
                if status == 'flame':
                    status_symbol = 'FLAME'
                elif status == 'no_flame':
                    status_symbol = 'CLEAR'
                else:
                    status_symbol = '?????'
                f.write(
                    f"{i:5d} | {flame_pct:6.2f} | {conf:11.1f} | {status:8s} {status_symbol} | {filename}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("СВОДНАЯ СТАТИСТИКА:\n")
            f.write(f"Всего кадров: {len(flame_percentages)}\n")
            f.write(
                f"С пламенем: {sum(1 for s in statuses if s == 'flame')}\n")
            f.write(
                f"Без пламени: {sum(1 for s in statuses if s == 'no_flame')}\n")
            f.write(
                f"Неопределенные: {sum(1 for s in statuses if s == 'uncertain')}\n")
            if flame_percentages:
                f.write(
                    f"Средний процент пламени: {np.mean(flame_percentages):.2f}%\n")
                f.write(f"Средняя уверенность: {np.mean(confidences):.2f}%\n")
        print(f"Детальная статистика сохранена в: {stats_file}")
    except Exception as e:
        print("Ошибка при сохранении статистики:", e)
        traceback.print_exc()


def gen_masks_with_quality_check(frames_dir, masks_out_dir, polygon=None,
                                 use_homography=False, src_pts=None,
                                 save_previews=True, preview_count=10, use_roi=None):
    """
    Обёртка: генерирует маски и (опционально) создаёт превью.
    """
    gen_masks(frames_dir, masks_out_dir, polygon,
              use_homography=use_homography, src_pts=src_pts,
              use_roi=use_roi)

    if save_previews:
        print("\nСоздание превью для контроля качества...")
        create_quality_previews(frames_dir, masks_out_dir, preview_count)


def create_quality_previews(frames_dir, masks_dir, count=10):
    """Создание превью для визуальной оценки качества детекции"""
    frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    masks = sorted(glob.glob(os.path.join(masks_dir, "*.png")))

    if len(frames) == 0 or len(masks) == 0:
        print("Нет кадров или масок для создания превью")
        return

    preview_dir = os.path.join(os.path.dirname(masks_dir), "flame_preview")
    os.makedirs(preview_dir, exist_ok=True)

    # Выбираем кадры равномерно по всей последовательности
    indices = [min(i * len(frames) // count, len(frames)-1)
               for i in range(count)]

    for i, idx in enumerate(indices):
        frame = cv2.imread(frames[idx])
        mask = cv2.imread(masks[idx], cv2.IMREAD_GRAYSCALE)
        if frame is None or mask is None:
            continue

        target_height, target_width = RESIZE_TO[1], RESIZE_TO[0]
        frame_resized = cv2.resize(frame, (target_width, target_height))

        mask_colored = np.zeros_like(frame_resized)
        mask_colored[:, :, 2] = mask
        mask_colored[:, :, 1] = mask // 2

        overlay = cv2.addWeighted(frame_resized, 0.7, mask_colored, 0.3, 0)
        flame_percent = (cv2.countNonZero(mask) /
                         (mask.shape[0] * mask.shape[1])) * 100
        text = f"Frame {idx}: Flame {flame_percent:.1f}%"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(overlay, (5, 5), (text_size[0] + 10, 35), (0, 0, 0), -1)
        cv2.putText(overlay, text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        preview_path = os.path.join(
            preview_dir, f"preview_{i:03d}_frame_{idx:06d}.jpg")
        cv2.imwrite(preview_path, overlay)

    print(f"Создано {len(indices)} превью в папке: {preview_dir}")


def create_combined_previews(all_frames_dirs, all_masks_dirs, video_names,
                             output_dir="data/flame_preview", samples_per_video=10):
    """
    Создание объединенных превью для всех видео одновременно

    Args:
        all_frames_dirs: список папок с кадрами
        all_masks_dirs: список папок с масками  
        video_names: список названий видео
        output_dir: папка для сохранения превью
        samples_per_video: количество образцов на видео
    """
    os.makedirs(output_dir, exist_ok=True)

    all_stats = {
        'total_frames': 0,
        'total_masks': 0,
        'all_flame_percentages': [],
        'video_stats': {}
    }

    preview_count = 0

    for video_idx, (frames_dir, masks_dir, video_name) in enumerate(zip(all_frames_dirs, all_masks_dirs, video_names)):
        frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
        masks = sorted(glob.glob(os.path.join(masks_dir, "*.png")))

        if len(frames) == 0 or len(masks) == 0:
            print(f"Пропускаем {video_name}: нет кадров или масок")
            continue

        print(f"Создание превью для {video_name}: {len(frames)} кадров")

        # Выбираем образцы равномерно
        min_count = min(len(frames), len(masks))
        indices = [min(i * min_count // samples_per_video, min_count-1)
                   for i in range(min(samples_per_video, min_count))]

        video_flame_percentages = []

        for sample_idx, frame_idx in enumerate(indices):
            try:
                frame = cv2.imread(frames[frame_idx])
                mask = cv2.imread(masks[frame_idx], cv2.IMREAD_GRAYSCALE)
                if frame is None or mask is None:
                    continue

                # Приводим к стандартному размеру
                target_height, target_width = RESIZE_TO[1], RESIZE_TO[0]
                frame_resized = cv2.resize(
                    frame, (target_width, target_height))
                mask_resized = cv2.resize(mask, (target_width, target_height))

                # Создаем наложение
                mask_colored = np.zeros_like(frame_resized)
                mask_colored[:, :, 2] = mask_resized
                mask_colored[:, :, 1] = mask_resized // 2

                overlay = cv2.addWeighted(
                    frame_resized, 0.7, mask_colored, 0.3, 0)

                # Вычисляем процент пламени
                flame_percent = (cv2.countNonZero(mask_resized) /
                                 (mask_resized.shape[0] * mask_resized.shape[1])) * 100
                video_flame_percentages.append(flame_percent)

                # Добавляем текст с информацией о видео
                text = f"{video_name} Frame {frame_idx}: Flame {flame_percent:.1f}%"
                text_size = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(overlay, (5, 5),
                              (text_size[0] + 10, 35), (0, 0, 0), -1)
                cv2.putText(overlay, text, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Сохраняем с уникальным именем
                preview_path = os.path.join(output_dir,
                                            f"combined_preview_{video_idx:02d}_{sample_idx:03d}_frame_{frame_idx:06d}.jpg")
                cv2.imwrite(preview_path, overlay)
                preview_count += 1

            except Exception as e:
                print(
                    f"Ошибка при создании превью для {video_name}, кадр {frame_idx}: {e}")

        # Обновляем статистику
        all_stats['total_frames'] += len(frames)
        all_stats['total_masks'] += len(masks)
        all_stats['all_flame_percentages'].extend(video_flame_percentages)
        all_stats['video_stats'][video_name] = {
            'frames': len(frames),
            'masks': len(masks),
            'avg_flame_percent': np.mean(video_flame_percentages) if video_flame_percentages else 0,
            'samples_created': len(video_flame_percentages)
        }

        print(
            f"✓ Создано {len(video_flame_percentages)} превью для {video_name}")

    # Сохраняем общую статистику
    if all_stats['all_flame_percentages']:
        stats_file = os.path.join(
            output_dir, "combined_preview_statistics.txt")
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("ОБЪЕДИНЕННАЯ СТАТИСТИКА ПРЕВЬЮ ДЕТЕКЦИИ ПЛАМЕНИ\n")
            f.write("=" * 60 + "\n\n")

            f.write("ОБЩАЯ ИНФОРМАЦИЯ:\n")
            f.write(f"Всего видео: {len(video_names)}\n")
            f.write(f"Всего кадров: {all_stats['total_frames']}\n")
            f.write(f"Всего масок: {all_stats['total_masks']}\n")
            f.write(f"Всего превью: {preview_count}\n\n")

            flame_percentages = all_stats['all_flame_percentages']
            f.write("СТАТИСТИКА ПО ПЛАМЕНИ:\n")
            f.write(f"Средний процент: {np.mean(flame_percentages):.2f}%\n")
            f.write(f"Медиана: {np.median(flame_percentages):.2f}%\n")
            f.write(
                f"Стандартное отклонение: {np.std(flame_percentages):.2f}%\n")
            f.write(f"Минимум: {min(flame_percentages):.2f}%\n")
            f.write(f"Максимум: {max(flame_percentages):.2f}%\n\n")

            f.write("СТАТИСТИКА ПО ВИДЕО:\n")
            f.write("-" * 40 + "\n")
            for video_name, stats in all_stats['video_stats'].items():
                f.write(f"{video_name}:\n")
                f.write(f"  Кадров: {stats['frames']}\n")
                f.write(f"  Масок: {stats['masks']}\n")
                f.write(
                    f"  Средний % пламени: {stats['avg_flame_percent']:.2f}%\n")
                f.write(f"  Превью создано: {stats['samples_created']}\n\n")

        print(f"\nОбъединенная статистика сохранена: {stats_file}")

    print(f"Всего создано {preview_count} объединенных превью в: {output_dir}")
    return preview_count > 0


def analyze_flame_detection_results(masks_dir, frames_dir=None):
    """Анализ результатов детекции пламени"""
    masks = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
    if len(masks) == 0:
        print("Маски не найдены")
        return

    total_pixels = 0
    flame_stats = []
    for mask_path in masks:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if total_pixels == 0:
            total_pixels = mask.shape[0] * mask.shape[1]
        flame_pixels = cv2.countNonZero(mask)
        flame_percent = (flame_pixels / total_pixels) * 100
        flame_stats.append(flame_percent)

    if flame_stats:
        print(f"Средний процент пламени: {np.mean(flame_stats):.2f}%")
        print(f"Медианный процент: {np.median(flame_stats):.2f}%")
        print(f"Стандартное отклонение: {np.std(flame_stats):.2f}%")
        print(f"Минимум: {np.min(flame_stats):.2f}%")
        print(f"Максимум: {np.max(flame_stats):.2f}%")
