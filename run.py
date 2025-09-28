import os
import sys
import glob
import traceback
import io
import contextlib

# Получаем абсолютный путь к текущей папке
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
models_dir = os.path.join(current_dir, 'models')
sys.path.insert(0, models_dir)
sys.path.insert(0, current_dir)

# Добавляем src в начало пути
sys.path.insert(0, src_dir)


def setup_directories():
    """Создание необходимых папок"""
    dirs = [
        'data', 'data/videos', 'data/frames', 'data/frames/1video',
        'data/frames/2video', 'data/masks', 'data/masks/1video',
        'data/masks/2video', 'data/flame_preview', 'data/model_results'
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)

def run_and_capture_and_dedupe(func, *args, **kwargs):
    """
    Запускает func(*args, **kwargs) с перехватом stdout/stderr,
    затем удаляет подряд идущие повторяющиеся строки из вывода
    и печатает результат один раз.
    Возвращает кортеж: (результат_функции, captured_stdout_string, captured_stderr_string)
    """
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    result = None
    try:
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            result = func(*args, **kwargs)
    except Exception as e:
        # Если внутренняя функция упала, захватим вывод и пробросим исключение дальше,
        # но предварительно распечатаем отфильтрованный вывод (чтобы не терять логи).
        out_text = buf_out.getvalue()
        err_text = buf_err.getvalue()
        # Дедупликация подряд идущих повторов
        filtered_out = _dedupe_consecutive_lines(out_text)
        if filtered_out:
            print(filtered_out)
        if err_text:
            print(err_text, file=sys.stderr)
        raise

    out_text = buf_out.getvalue()
    err_text = buf_err.getvalue()
    filtered_out = _dedupe_consecutive_lines(out_text)
    if filtered_out:
        print(filtered_out)
    if err_text:
        print(err_text, file=sys.stderr)
    return result, out_text, err_text

def _dedupe_consecutive_lines(text):
    """Удаляет подряд идущие одинаковые строки, возвращает итоговую строку."""
    if not text:
        return ""
    lines = text.splitlines()
    deduped = []
    prev = None
    for ln in lines:
        if ln != prev:
            deduped.append(ln)
        prev = ln
    return "\n".join(deduped)

def step1_extract_all_frames():
    """Шаг 1: Извлечение кадров из всех видео"""
    print("=" * 50)
    print("ШАГ 1: ИЗВЛЕЧЕНИЕ КАДРОВ ИЗ ВСЕХ ВИДЕО")
    print("=" * 50)

    videos = [
        ("1video.avi", "data/videos/1video.avi", "data/frames/1video"),
        ("2video.avi", "data/videos/2video.avi", "data/frames/2video")
    ]

    extracted_videos = []

    try:
        from extract_frames import extract_frames
    except Exception as e:
        print(f"Ошибка импорта extract_frames: {e}")
        return extracted_videos

    for video_name, video_path, frames_dir in videos:
        if not os.path.exists(video_path):
            print(f"Видео не найдено: {video_name}")
            continue

        # Проверяем, есть ли уже кадры
        frames_exist = len(glob.glob(os.path.join(frames_dir, "*.jpg"))) > 0
        if frames_exist:
            print(f"✓ Кадры уже извлечены для {video_name}")
            frames_count = len(glob.glob(os.path.join(frames_dir, "*.jpg")))
            extracted_videos.append(
                (video_name, video_path, frames_dir, frames_count))
            continue

        print(f"Извлечение кадров из {video_name}")
        try:
            extract_frames(video_path, frames_dir, fps_out=5)
            frames_count = len(glob.glob(os.path.join(frames_dir, "*.jpg")))
            if frames_count > 0:
                extracted_videos.append(
                    (video_name, video_path, frames_dir, frames_count))
                print(f"✓ Извлечено {frames_count} кадров из {video_name}")
            else:
                print(f"✗ Не удалось извлечь кадры из {video_name}")
        except Exception as e:
            print(f"✗ Ошибка при извлечении кадров из {video_name}: {e}")

    print(f"\nРезультат: извлечены кадры из {len(extracted_videos)} видео")
    for video_name, _, frames_dir, frames_count in extracted_videos:
        print(f"  • {video_name}: {frames_count} кадров")

    return extracted_videos


def step2_generate_all_masks(extracted_videos):
    """Шаг 2: Генерация масок пламени для всех видео"""
    print("=" * 50)
    print("ШАГ 2: ГЕНЕРАЦИЯ МАСОК ПЛАМЕНИ ДЛЯ ВСЕХ ВИДЕО")
    print("=" * 50)

    processed_videos = []

    try:
        from gen_pseudo_masks import gen_masks_temporal
        import numpy as np
    except Exception as e:
        print(f"Ошибка импорта: {e}")
        return processed_videos

    for video_name, video_path, frames_dir, frames_count in extracted_videos:
        masks_dir = frames_dir.replace("frames", "masks")

        # Проверяем, есть ли уже маски
        masks_exist = len(glob.glob(os.path.join(masks_dir, "*.png"))) > 0
        if masks_exist:
            masks_count = len(glob.glob(os.path.join(masks_dir, "*.png")))
            print(f"✓ Маски уже созданы для {video_name}: {masks_count} масок")
            processed_videos.append((video_name, frames_dir, masks_dir))
            continue

        print(f"Генерация масок для {video_name}")

        # Проверяем наличие ROI для конкретного видео
        video_id = video_name.replace('.avi', '')
        roi_file = f"data/roi_{video_id}.npy"
        polygon = None
        use_roi = False

        if os.path.exists(roi_file):
            try:
                polygon = np.load(roi_file).tolist()
                print(f"  ROI найден для {video_id}: {len(polygon)} точек")
                use_roi = True
            except Exception as e:
                print(f"  Ошибка загрузки ROI: {e}")
        else:
            print(f"  ROI не найден для {video_id}, обрабатываем весь кадр")

        # Генерируем маски с улучшенными параметрами
        try:
            gen_masks_temporal(
                frames_dir=frames_dir,
                masks_out_dir=masks_dir,
                polygon=polygon,
                use_roi=use_roi,
                buffer_size=10,
                brightness_thresh=200,
                saturation_thresh=100,
                flicker_weight=0.3,
                min_flicker_frames=5,
                use_color_filter=True,
                confidence_threshold=15.0,
                flame_percent_threshold=0.15
            )

            masks_count = len(glob.glob(os.path.join(masks_dir, "*.png")))
            if masks_count > 0:
                processed_videos.append((video_name, frames_dir, masks_dir))
                print(f"✓ Создано {masks_count} масок для {video_name}")
            else:
                print(f"✗ Не удалось создать маски для {video_name}")

        except Exception as e:
            print(f"✗ Ошибка при создании масок для {video_name}: {e}")
            traceback.print_exc()

    print(f"\nРезультат: созданы маски для {len(processed_videos)} видео")
    return processed_videos


def step3_create_combined_preview(processed_videos, samples_per_video=10):
    """Шаг 3: Создание объединенного превью результатов для всех видео"""
    print("=" * 50)
    print("ШАГ 3: СОЗДАНИЕ ОБЪЕДИНЕННОГО ПРЕВЬЮ РЕЗУЛЬТАТОВ")
    print("=" * 50)

    if not processed_videos:
        print("Нет обработанных видео для создания превью")
        return False

    try:
        import cv2
        import numpy as np
        from datetime import datetime
    except Exception as e:
        print(f"Ошибка импорта библиотек: {e}")
        return False

    preview_dir = "data/flame_preview"
    os.makedirs(preview_dir, exist_ok=True)

    all_previews = []
    total_stats = {
        'total_frames': 0,
        'total_masks': 0,
        'all_flame_percentages': [],
        'video_stats': {}
    }

    # Создаем превью для каждого видео
    for video_name, frames_dir, masks_dir in processed_videos:
        frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
        masks = sorted(glob.glob(os.path.join(masks_dir, "*.png")))

        if len(frames) == 0 or len(masks) == 0:
            print(f"Пропускаем {video_name}: нет кадров или масок")
            continue

        print(
            f"Создание превью для {video_name}: {len(frames)} кадров, {len(masks)} масок")

        # Выбираем образцы равномерно по видео
        min_count = min(len(frames), len(masks))
        indices = [min(i * min_count // samples_per_video, min_count-1)
                   for i in range(min(samples_per_video, min_count))]

        video_flame_percentages = []
        video_previews = []

        for i, idx in enumerate(indices):
            try:
                frame = cv2.imread(frames[idx])
                mask = cv2.imread(masks[idx], cv2.IMREAD_GRAYSCALE)
                if frame is None or mask is None:
                    continue

                # Приводим к стандартному размеру
                target_height, target_width = 576, 928
                frame_resized = cv2.resize(
                    frame, (target_width, target_height))
                mask_resized = cv2.resize(mask, (target_width, target_height))

                # Создаем цветную маску для наложения
                mask_colored = np.zeros_like(frame_resized)
                mask_colored[:, :, 2] = mask_resized  # Красный канал
                # Зеленый для оранжевого
                mask_colored[:, :, 1] = mask_resized // 2

                # Создаем наложение
                overlay = cv2.addWeighted(
                    frame_resized, 0.7, mask_colored, 0.3, 0)

                # Вычисляем процент пламени
                flame_percent = (cv2.countNonZero(mask_resized) /
                                 (mask_resized.shape[0] * mask_resized.shape[1])) * 100
                video_flame_percentages.append(flame_percent)

                # Добавляем текст
                text = f"{video_name} Frame {idx}: Flame {flame_percent:.1f}%"
                text_size = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(overlay, (5, 5),
                              (text_size[0] + 10, 35), (0, 0, 0), -1)
                cv2.putText(overlay, text, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                preview_path = os.path.join(preview_dir,
                                            f"preview_{video_name}_{i:03d}_frame_{idx:06d}.jpg")
                cv2.imwrite(preview_path, overlay)
                video_previews.append(preview_path)

            except Exception as e:
                print(f"  Ошибка при создании превью для кадра {idx}: {e}")

        # Обновляем общую статистику
        total_stats['total_frames'] += len(frames)
        total_stats['total_masks'] += len(masks)
        total_stats['all_flame_percentages'].extend(video_flame_percentages)
        total_stats['video_stats'][video_name] = {
            'frames': len(frames),
            'masks': len(masks),
            'avg_flame_percent': np.mean(video_flame_percentages) if video_flame_percentages else 0,
            'previews_created': len(video_previews)
        }

        print(f"  ✓ Создано {len(video_previews)} превью для {video_name}")
        all_previews.extend(video_previews)

    # Создаем общую статистику
    if total_stats['all_flame_percentages']:
        avg_flame = np.mean(total_stats['all_flame_percentages'])
        median_flame = np.median(total_stats['all_flame_percentages'])
        std_flame = np.std(total_stats['all_flame_percentages'])
        min_flame = min(total_stats['all_flame_percentages'])
        max_flame = max(total_stats['all_flame_percentages'])

        # Сохраняем общую статистику
        stats_file = os.path.join(preview_dir, "combined_statistics.txt")
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("ОБЪЕДИНЕННАЯ СТАТИСТИКА ДЕТЕКЦИИ ПЛАМЕНИ\n")
            f.write("=" * 60 + "\n\n")

            f.write("ОБЩАЯ ИНФОРМАЦИЯ:\n")
            f.write(f"Всего видео обработано: {len(processed_videos)}\n")
            f.write(f"Всего кадров: {total_stats['total_frames']}\n")
            f.write(f"Всего масок: {total_stats['total_masks']}\n")
            f.write(f"Всего превью создано: {len(all_previews)}\n\n")

            f.write("СТАТИСТИКА ПО ПЛАМЕНИ:\n")
            f.write(f"Средний процент пламени: {avg_flame:.2f}%\n")
            f.write(f"Медиана: {median_flame:.2f}%\n")
            f.write(f"Стандартное отклонение: {std_flame:.2f}%\n")
            f.write(f"Минимум: {min_flame:.2f}%\n")
            f.write(f"Максимум: {max_flame:.2f}%\n\n")

            f.write("СТАТИСТИКА ПО ВИДЕО:\n")
            f.write("-" * 60 + "\n")
            for video_name, stats in total_stats['video_stats'].items():
                f.write(f"{video_name}:\n")
                f.write(f"  Кадров: {stats['frames']}\n")
                f.write(f"  Масок: {stats['masks']}\n")
                f.write(
                    f"  Средний % пламени: {stats['avg_flame_percent']:.2f}%\n")
                f.write(f"  Превью создано: {stats['previews_created']}\n\n")

        print(f"\nОБЩАЯ СТАТИСТИКА:")
        print(f"• Всего видео: {len(processed_videos)}")
        print(f"• Всего кадров: {total_stats['total_frames']}")
        print(f"• Всего масок: {total_stats['total_masks']}")
        print(f"• Средний процент пламени: {avg_flame:.2f}%")
        print(f"• Превью создано: {len(all_previews)}")
        print(f"• Статистика сохранена: {stats_file}")

    print(f"✓ Превью сохранены в папку: {preview_dir}")
    return True


def step4_train_model(processed_videos):
    """Шаг 4: Обучение модели на всех видео"""
    print("=" * 50)
    print("ШАГ 4: ОБУЧЕНИЕ МОДЕЛИ НА ВСЕХ ВИДЕО")
    print("=" * 50)

    if not processed_videos:
        print("Нет обработанных видео для обучения")
        return False

    try:
        from models.flame_segmentation_model import train_and_export
    except Exception as e:
        print("Не удалось импортировать train_and_export:", e)
        return False

    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        device = 'cpu'

    print(f"Используем устройство: {device}")

    # Собираем пути ко всем папкам
    all_frames_dirs = [frames_dir for _, frames_dir, _ in processed_videos]
    all_masks_dirs = [masks_dir for _, _, masks_dir in processed_videos]

    print("Данные для обучения:")
    total_frames = 0
    total_masks = 0
    for video_name, frames_dir, masks_dir in processed_videos:
        frames_count = len(glob.glob(os.path.join(frames_dir, "*.jpg")))
        masks_count = len(glob.glob(os.path.join(masks_dir, "*.png")))
        total_frames += frames_count
        total_masks += masks_count
        print(f"  • {video_name}: {frames_count} кадров, {masks_count} масок")

    print(f"ИТОГО: {total_frames} кадров, {total_masks} масок")
    print("Запуск обучения... (это может занять время)")

    try:
        model, history = train_and_export(
            frames_dir=all_frames_dirs,  # Передаем список папок
            masks_dir=all_masks_dirs,    # Передаем список папок
            img_size=(256, 256),
            batch_size=8,
            epochs=2,
            learning_rate=1e-3,
            val_split=0.2,
            use_temporal=False,
            checkpoint_dir='checkpoints',
            force_gpu=True,
            save_checkpoints=True,
            export_model=True,
            allow_unsafe_checkpoint_load=False
        )

        if model is not None:
            print("✓ Обучение завершено успешно")
            return True
        else:
            print("✗ Обучение завершено, но модель не экспортирована")
            return False

    except Exception as e:
        print(f"\n✗ Ошибка обучения: {e}")
        traceback.print_exc()
        return False


def print_final_summary(processed_videos, training_success):
    """Вывод итогового резюме"""
    print("\n" + "=" * 60)
    print("ИТОГОВОЕ РЕЗЮМЕ")
    print("=" * 60)

    if not processed_videos:
        print(" Ни одно видео не было успешно обработано")
        return

    print(" ОБРАБОТАННЫЕ ВИДЕО:")
    total_frames = 0
    total_masks = 0

    for video_name, frames_dir, masks_dir in processed_videos:
        frames_count = len(glob.glob(os.path.join(frames_dir, "*.jpg")))
        masks_count = len(glob.glob(os.path.join(masks_dir, "*.png")))
        total_frames += frames_count
        total_masks += masks_count
        print(f"  ✓ {video_name}: {frames_count} кадров, {masks_count} масок")

    print(f"\nОБЩАЯ СТАТИСТИКА:")
    print(f"  • Всего видео обработано: {len(processed_videos)}")
    print(f"  • Всего кадров: {total_frames}")
    print(f"  • Всего масок: {total_masks}")

    print(f"\n ОБУЧЕНИЕ МОДЕЛИ:")
    if training_success:
        print("  ✓ Модель успешно обучена")
        if os.path.exists("flame_unet.pth"):
            print("  ✓ Модель экспортирована: flame_unet.pth")
    else:
        print("   Ошибка при обучении модели")

    print(f"\n РЕЗУЛЬТАТЫ СОХРАНЕНЫ В:")
    dirs_to_check = [
        ("data/flame_preview/", "Превью детекции пламени"),
        ("data/model_results/", "Результаты модели"),
        ("checkpoints/", "Чекпоинты обучения")
    ]

    for dir_path, description in dirs_to_check:
        if os.path.exists(dir_path) and os.listdir(dir_path):
            files_count = len([f for f in os.listdir(
                dir_path) if os.path.isfile(os.path.join(dir_path, f))])
            print(f"   {dir_path} - {description} ({files_count} файлов)")

    if os.path.exists("flame_unet.pth"):
        print(f"  flame_unet.pth - Готовая модель для использования")

    print("\n" + "=" * 60)


def main():
    """Основная функция с последовательной обработкой всех видео"""
    print("ДЕТЕКЦИЯ ПЛАМЕНИ В СТЕКЛОВАРЕННОЙ ПЕЧИ")
    print("=" * 50)

    # Создаем папки
    setup_directories()

    # Шаг 1: Извлечение кадров из всех видео
    extracted_videos = step1_extract_all_frames()
    if not extracted_videos:
        print("Не удалось извлечь кадры ни из одного видео")
        return

    # Шаг 2: Генерация масок для всех видео
    processed_videos = step2_generate_all_masks(extracted_videos)
    if not processed_videos:
        print("Не удалось создать маски ни для одного видео")
        return

    # Шаг 3: Создание объединенного превью
    preview_success = step3_create_combined_preview(
        processed_videos, samples_per_video=10)
    if not preview_success:
        print("  Не удалось создать превью, но продолжаем обучение")

    # Шаг 4: Обучение модели на всех данных
    training_success = step4_train_model(processed_videos)

    # Итоговое резюме
    print_final_summary(processed_videos, training_success)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрервано пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"\nНепредвиденная ошибка: {e}")
        traceback.print_exc()
        sys.exit(1)
