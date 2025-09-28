#!/usr/bin/env python3
"""
Упрощенный запуск системы детекции пламени
Исправлены все проблемы с импортами
"""

import os
import sys
import glob
import traceback

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
        'data/masks/2video'
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)


def find_video():
    """Поиск видео файла"""
    candidates = [
        "data/videos/1video.avi",
        "data/videos/2video.avi",
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return None


def step1_extract_frames(video_path, output_dir):
    """Шаг 1: Извлечение кадров с указанием папки"""
    print(f"Извлечение кадров из {os.path.basename(video_path)}")

    try:
        from extract_frames import extract_frames
        extract_frames(video_path, output_dir, fps_out=5)

        frames = glob.glob(os.path.join(output_dir, "*.jpg"))
        print(f"Извлечено {len(frames)} кадров")
        return len(frames) > 0

    except Exception as e:
        print(f"Ошибка извлечения кадров: {e}")
        return False


def step2_generate_masks(frames_dir, masks_dir):
    """Шаг 2: Генерация масок пламени"""
    print("Генерация масок пламени")

    try:
        from gen_pseudo_masks import gen_masks

        # Проверяем наличие ROI для конкретного видео
        video_id = os.path.basename(frames_dir)
        roi_file = f"data/roi_{video_id}.npy"
        polygon = None
        use_roi = False

        if os.path.exists(roi_file):
            import numpy as np
            polygon = np.load(roi_file).tolist()
            print(f"ROI найден для {video_id}: {len(polygon)} точек")
            use_roi = True
        else:
            print(f"ROI не найден для {video_id}, обрабатываем весь кадр")

        # Генерируем маски
        gen_masks(
            frames_dir=frames_dir,
            masks_out_dir=masks_dir,
            polygon=polygon,
            use_roi=use_roi
        )

        # Проверяем результат
        masks = glob.glob(os.path.join(masks_dir, "*.png"))
        print(f"Создано {len(masks)} масок")
        return len(masks) > 0

    except Exception as e:
        print(f"Ошибка генерации масок: {e}")
        return False


def step3_create_preview(frames_dir, masks_dir, video_id):
    """Шаг 3: Создание превью результатов"""
    print("Создание превью результатов")

    try:
        from gen_pseudo_masks import create_quality_previews
        
        # Создаем превью в отдельной папке для каждого видео
        preview_dir = f"data/flame_preview_{video_id}"
        os.makedirs(preview_dir, exist_ok=True)
        
        create_quality_previews(frames_dir, masks_dir, count=15)
        print(f"Превью созданы в {preview_dir}/")
        return True

    except Exception as e:
        print(f"Ошибка создания превью: {e}")
        return False


def step4_train_model(processed_videos):
    """Шаг 4: Обучение модели на всех видео"""
    print(f"\nШаг 4: Обучение модели на {len(processed_videos)} видео")
    
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
    for video_name, frames_dir, masks_dir in processed_videos:
        frames_count = len(glob.glob(os.path.join(frames_dir, "*.jpg")))
        masks_count = len(glob.glob(os.path.join(masks_dir, "*.png")))
        print(f"  • {video_name}: {frames_count} кадров, {masks_count} масок")
    
    print("Запуск обучения... (это может занять время)")
    
    try:
        model, history = train_and_export(
            frames_dir=all_frames_dirs,  # Передаем список папок
            masks_dir=all_masks_dirs,    # Передаем список папок
            img_size=(256, 256),
            batch_size=8,
            epochs=30,
            learning_rate=1e-3,
            val_split=0.2,
            use_temporal=False,
            checkpoint_dir='checkpoints',
            force_gpu=True,
            save_checkpoints=False,
            export_model=False,
            allow_unsafe_checkpoint_load=False
        )
        
        if model is not None:
            print("Обучение завершено успешно")
        else:
            print("Обучение завершено, но модель не экспортирована")
        return True
        
    except Exception as e:
        print(f"\nОшибка обучения: {e}")
        traceback.print_exc()
        return False


def print_statistics(stats_file):
    """Печать статистики детекции, если она есть"""
    try:
        if not os.path.exists(stats_file):
            print("Статистика не найдена.")
            return
        with open(stats_file, 'r', encoding='utf-8') as f:
            lines = [ln.rstrip('\n') for ln in f.readlines()]
        if not lines:
            print("Файл статистики пуст.")
            return
    except Exception as e:
        print(f"Ошибка чтения статистики: {e}")
        traceback.print_exc()


def main():
    """Основная функция"""
    print("ДЕТЕКЦИЯ ПЛАМЕНИ В СТЕКЛОВАРЕННОЙ ПЕЧИ")
    print("=" * 50)

    # Создаем папки
    setup_directories()

    # Обрабатываем все найденные видео
    videos = [
        ("1video.avi", "data/videos/1video.avi", "data/frames/1video", "data/masks/1video"),
        ("2video.avi", "data/videos/2video.avi", "data/frames/2video", "data/masks/2video")
    ]
    
    processed_videos = []
    
    for video_name, video_path, frames_dir, masks_dir in videos:
        if not os.path.exists(video_path):
            print(f"Видео не найдено: {video_name}")
            continue
            
        print(f"\nОбработка {video_name}")
        print("-" * 30)
        
        # Проверяем, есть ли уже кадры
        frames_exist = len(glob.glob(os.path.join(frames_dir, "*.jpg"))) > 0
        if frames_exist:
            print(f"Кадры уже извлечены для {video_name}")
            extract_ok = True
        else:
            extract_ok = step1_extract_frames(video_path, frames_dir)

        if not extract_ok:
            print(f"Не удалось извлечь кадры для {video_name}")
            continue

        # Генерируем маски
        masks_ok = step2_generate_masks(frames_dir, masks_dir)
        if not masks_ok:
            print(f"Не удалось сгенерировать маски для {video_name}")
            continue

        # Создаем превью
        preview_ok = step3_create_preview(frames_dir, masks_dir, video_name.replace('.avi', ''))
        
        processed_videos.append((video_name, frames_dir, masks_dir))
        print(f"✓ {video_name} обработано успешно")

    if not processed_videos:
        print("Ни одно видео не удалось обработать")
        return

    print(f"\nУспешно обработано видео: {len(processed_videos)}")
    
    # Обучение модели на всех обработанных видео
    train_ok = step4_train_model(processed_videos)

    # Итог
    print("\nИТОГИ:")
    for video_name, frames_dir, masks_dir in processed_videos:
        frames_count = len(glob.glob(os.path.join(frames_dir, "*.jpg")))
        masks_count = len(glob.glob(os.path.join(masks_dir, "*.png")))
        print(f"• {video_name}: {frames_count} кадров, {masks_count} масок")
    
    print(f"• Обучение модели: {'Успешно' if train_ok else 'Ошибка'}")
    print(f"\nРезультаты:")
    print(f"• data/flame_preview/ - примеры детекции")
    print(f"• data/model_results/ - результаты модели")
    if os.path.exists("flame_unet.pth"):
        print(f"• flame_unet.pth - обученная модель")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n Прервано пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"\nНепредвиденная ошибка: {e}")
        traceback.print_exc()
        sys.exit(1)
