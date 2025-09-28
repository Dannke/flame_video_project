#!/usr/bin/env python3
"""
Полный пайплайн для обучения модели детекции пламени в стекловаренной печи
"""

import os
import sys
import subprocess
import numpy as np
import cv2
from pathlib import Path

# Добавляем src в путь
sys.path.append('src')


def setup_directories():
    """Создание необходимых директорий"""
    directories = [
        'data',
        'data/videos',
        'data/frames',
        'data/frames/1video',
        'data/masks',
        'data/masks/1video',
        'data/preview',
        'data/flame_preview',
        'models',
        'logs',
        'test_output'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Создана/проверена папка: {directory}")


def check_requirements():
    """Проверка наличия необходимых файлов и библиотек"""
    print("Проверка зависимостей...")

    # Проверяем наличие основных файлов
    required_files = [
        'src/extract_frames.py',
        'src/image_processing.py',
        'src/gen_pseudo_masks.py',
        'src/dataset.py',
        'src/train_unet.py'
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print("❌ Отсутствуют файлы:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False

    # Проверяем импорты
    try:
        import torch
        import cv2
        import numpy as np
        from skimage import filters, morphology
        import scipy.signal
        print("✓ Все необходимые библиотеки найдены")
        return True
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("Установите недостающие библиотеки:")
        print("pip install torch torchvision opencv-python scikit-image scipy numpy")
        return False


def step1_extract_frames(video_path, output_fps=5):
    """Шаг 1: Извлечение кадров из видео"""
    print(f"\n{'='*60}")
    print("ШАГ 1: ИЗВЛЕЧЕНИЕ КАДРОВ ИЗ ВИДЕО")
    print(f"{'='*60}")

    if not os.path.exists(video_path):
        print(f"❌ Видео файл не найден: {video_path}")
        return False

    frames_dir = "data/frames/1video"

    try:
        from extract_frames import extract_frames
        extract_frames(video_path, frames_dir, fps_out=output_fps)

        # Подсчитываем количество извлеченных кадров
        frame_count = len([f for f in os.listdir(
            frames_dir) if f.endswith('.jpg')])
        print(f"✓ Извлечено {frame_count} кадров в {frames_dir}")

        if frame_count == 0:
            print("❌ Не удалось извлечь кадры")
            return False

        return True

    except Exception as e:
        print(f"❌ Ошибка при извлечении кадров: {e}")
        return False


def step2_select_roi():
    """Шаг 2: Выбор области интереса (ROI)"""
    print(f"\n{'='*60}")
    print("ШАГ 2: ВЫБОР ОБЛАСТИ ИНТЕРЕСА (ROI)")
    print(f"{'='*60}")

    frames_dir = "data/frames/1video"
    roi_file = "data/roi_1video.npy"

    # Проверяем наличие кадров
    frames = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
    if len(frames) == 0:
        print("❌ Кадры не найдены. Сначала выполните извлечение кадров.")
        return False

    # Если ROI уже существует, спрашиваем пользователя
    if os.path.exists(roi_file):
        response = input(
            f"ROI файл уже существует ({roi_file}). Перезаписать? (y/N): ")
        if response.lower() != 'y':
            print("✓ Используется существующий ROI файл")
            return True

    print("Запуск интерфейса выбора ROI...")
    print("Инструкции:")
    print("- Кликайте мышью для выбора точек полигона")
    print("- Нажмите ESC для завершения выбора")

    try:
        # Импортируем и запускаем select_roi
        sys.path.append('.')  # Добавляем текущую директорию
        subprocess.run(
            [sys.executable, 'src/select_roi.py', frames[0]], check=True)

        if os.path.exists(roi_file):
            roi_points = np.load(roi_file)
            print(
                f"✓ ROI сохранен: {len(roi_points)} точек в файле {roi_file}")
            return True
        else:
            print("❌ ROI не был сохранен")
            return False

    except Exception as e:
        print(f"❌ Ошибка при выборе ROI: {e}")
        return False


def step3_generate_flame_masks():
    """Шаг 3: Генерация псевдо-масок пламени"""
    print(f"\n{'='*60}")
    print("ШАГ 3: ГЕНЕРАЦИЯ ПСЕВДО-МАСОК ПЛАМЕНИ")
    print(f"{'='*60}")

    frames_dir = "data/frames/1video"
    masks_dir = "data/masks/1video"
    roi_file = "data/roi_1video.npy"

    # Проверяем наличие кадров
    frames = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
    if len(frames) == 0:
        print("❌ Кадры не найдены")
        return False

    # Загружаем ROI если есть
    polygon = None
    if os.path.exists(roi_file):
        polygon = np.load(roi_file).tolist()
        print(f"✓ Загружен ROI: {len(polygon)} точек")
    else:
        print("⚠️  ROI не найден, будет обрабатываться весь кадр")

    try:
        from gen_pseudo_masks import gen_masks_with_quality_check

        print("Генерация масок пламени...")
        print("Это может занять некоторое время...")

        gen_masks_with_quality_check(
            frames_dir=frames_dir,
            masks_dir=masks_dir,
            polygon=polygon,
            save_previews=True,
            preview_count=10
        )

        # Подсчитываем созданные маски
        mask_count = len([f for f in os.listdir(
            masks_dir) if f.endswith('.png')])
        print(f"✓ Создано {mask_count} масок пламени")

        if mask_count == 0:
            print("❌ Маски не были созданы")
            return False

        return True

    except Exception as e:
        print(f"❌ Ошибка при генерации масок: {e}")
        return False


def step4_inspect_masks():
    """Шаг 4: Инспекция качества масок"""
    print(f"\n{'='*60}")
    print("ШАГ 4: ИНСПЕКЦИЯ КАЧЕСТВА МАСОК")
    print(f"{'='*60}")

    frames_dir = "data/frames/1video"
    masks_dir = "data/masks/1video"

    try:
        from inspect_masks import inspect_flame_masks

        inspect_flame_masks(
            frames_dir=frames_dir,
            masks_dir=masks_dir,
            output_dir="data/flame_preview",
            max_images=30,
            styles=['fire', 'heat', 'contour'],
            save_individual=True
        )

        print("✓ Инспекция масок завершена")
        print("✓ Проверьте качество в папке data/flame_preview/")

        # Спрашиваем пользователя о качестве
        response = input(
            "Качество масок приемлемо? Продолжить обучение? (Y/n): ")
        if response.lower() == 'n':
            print("Остановлено пользователем. Настройте параметры в image_processing.py")
            return False

        return True

    except Exception as e:
        print(f"❌ Ошибка при инспекции: {e}")
        return False


def step5_train_model():
    """Шаг 5: Обучение модели U-Net"""
    print(f"\n{'='*60}")
    print("ШАГ 5: ОБУЧЕНИЕ МОДЕЛИ U-NET")
    print(f"{'='*60}")

    frames_dir = "data/frames/1video"
    masks_dir = "data/masks/1video"

    # Проверяем наличие данных
    frame_count = len([f for f in os.listdir(
        frames_dir) if f.endswith('.jpg')])
    mask_count = len([f for f in os.listdir(masks_dir) if f.endswith('.png')])

    if frame_count == 0 or mask_count == 0:
        print("❌ Нет данных для обучения")
        return False

    if frame_count != mask_count:
        print(
            f"⚠️  Количество кадров ({frame_count}) не совпадает с количеством масок ({mask_count})")
        print("Будет использовано минимальное количество")

    print(
        f"Данные для обучения: {min(frame_count, mask_count)} пар кадр-маска")

    try:
        from train_unet import train

        print("Запуск обучения нейронной сети...")
        print("Параметры обучения:")
        print("- Архитектура: Simple U-Net")
        print("- Эпохи: 8")
        print("- Размер батча: 8")
        print("- Learning rate: 1e-3")
        print("- Размер изображения: 256x256")

        train(
            images_dir=frames_dir,
            masks_dir=masks_dir,
            epochs=8,
            batch=8,
            lr=1e-3,
            size=(256, 256)
        )

        # Проверяем создание модели
        if os.path.exists("flame_unet.pth"):
            print("✓ Модель успешно обучена и сохранена: flame_unet.pth")
            return True
        else:
            print("❌ Модель не была сохранена")
            return False

    except Exception as e:
        print(f"❌ Ошибка при обучении: {e}")
        return False


def step6_evaluate_model():
    """Шаг 6: Оценка качества модели"""
    print(f"\n{'='*60}")
    print("ШАГ 6: ОЦЕНКА КАЧЕСТВА МОДЕЛИ")
    print(f"{'='*60}")

    if not os.path.exists("flame_unet.pth"):
        print("❌ Модель не найдена")
        return False

    try:
        # Создаем простую оценку модели
        print("Загрузка модели...")

        import torch
        from train_unet import SimpleUNet
        from archive.dataset import FlameDataset
        from torch.utils.data import DataLoader

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SimpleUNet().to(device)
        model.load_state_dict(torch.load(
            "flame_unet.pth", map_location=device))
        model.eval()

        # Создаем тестовый набор данных
        test_dataset = FlameDataset(
            "data/frames/1video", "data/masks/1video", size=(256, 256))
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

        print(f"Тестирование на {len(test_dataset)} изображениях...")

        total_loss = 0.0
        total_iou = 0.0
        num_batches = 0

        with torch.no_grad():
            for imgs, masks in test_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)

                preds = model(imgs)

                # BCE Loss
                bce_loss = torch.nn.BCELoss()(preds, masks)

                # IoU calculation
                pred_binary = (preds > 0.5).float()
                intersection = (pred_binary * masks).sum()
                union = pred_binary.sum() + masks.sum() - intersection
                iou = intersection / (union + 1e-8)

                total_loss += bce_loss.item()
                total_iou += iou.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_iou = total_iou / num_batches

        print(f"✓ Результаты оценки:")
        print(f"  - Средняя потеря (BCE): {avg_loss:.4f}")
        print(f"  - Средний IoU: {avg_iou:.4f}")
        print(
            f"  - Точность (IoU > 0.5): {'Хорошо' if avg_iou > 0.5 else 'Требует улучшения'}")

        # Создаем несколько тестовых предсказаний
        create_prediction_samples(model, test_dataset, device)

        return True

    except Exception as e:
        print(f"❌ Ошибка при оценке: {e}")
        return False


def create_prediction_samples(model, dataset, device, num_samples=5):
    """Создание примеров предсказаний модели"""
    import torch

    output_dir = "data/model_predictions"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Создание {num_samples} примеров предсказаний...")

    for i in range(min(num_samples, len(dataset))):
        img, true_mask = dataset[i]

        # Предсказание
        img_batch = img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred_mask = model(img_batch).squeeze(0).squeeze(0).cpu().numpy()

        # Конвертируем для визуализации
        img_vis = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

        true_mask_vis = (true_mask.squeeze(0).numpy() * 255).astype(np.uint8)
        pred_mask_vis = (pred_mask * 255).astype(np.uint8)
        pred_binary_vis = ((pred_mask > 0.5) * 255).astype(np.uint8)

        # Создаем сравнительную визуализацию
        img_resized = cv2.resize(img_vis, (256, 256))
        true_resized = cv2.resize(true_mask_vis, (256, 256))
        pred_resized = cv2.resize(pred_mask_vis, (256, 256))
        pred_binary_resized = cv2.resize(pred_binary_vis, (256, 256))

        # Цветные маски
        true_colored = cv2.cvtColor(true_resized, cv2.COLOR_GRAY2BGR)
        pred_colored = cv2.applyColorMap(pred_resized, cv2.COLORMAP_HOT)
        pred_binary_colored = cv2.cvtColor(
            pred_binary_resized, cv2.COLOR_GRAY2BGR)

        # Создаем overlay
        true_overlay = cv2.addWeighted(img_resized, 0.7, true_colored, 0.3, 0)
        pred_overlay = cv2.addWeighted(img_resized, 0.7, pred_colored, 0.3, 0)

        # Добавляем подписи
        cv2.putText(img_resized, "Original", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(true_overlay, "Ground Truth", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(pred_overlay, "Prediction", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(pred_binary_colored, "Binary", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Компонуем в сетку 2x2
        top_row = np.hstack([img_resized, true_overlay])
        bottom_row = np.hstack([pred_overlay, pred_binary_colored])
        combined = np.vstack([top_row, bottom_row])

        # Сохраняем
        output_path = os.path.join(
            output_dir, f"prediction_sample_{i:03d}.jpg")
        cv2.imwrite(output_path, combined)

    print(f"✓ Примеры предсказаний сохранены в: {output_dir}")


def generate_final_report():
    """Генерация финального отчета"""
    print(f"\n{'='*60}")
    print("ГЕНЕРАЦИЯ ФИНАЛЬНОГО ОТЧЕТА")
    print(f"{'='*60}")

    report_path = "flame_detection_training_report.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("ОТЧЕТ ОБ ОБУЧЕНИИ МОДЕЛИ ДЕТЕКЦИИ ПЛАМЕНИ\n")
        f.write("=" * 60 + "\n\n")

        f.write("ПАРАМЕТРЫ ПРОЕКТА:\n")
        f.write(f"- Задача: Детекция пламени в стекловаренной печи\n")
        f.write(f"- Архитектура: Simple U-Net\n")
        f.write(f"- Размер входных данных: 256x256\n")
        f.write(
            f"- Метод генерации масок: Псевдо-разметка на основе цветовых признаков\n\n")

        f.write("СТРУКТУРА ДАННЫХ:\n")

        # Статистика по кадрам
        frames_dir = "data/frames/1video"
        if os.path.exists(frames_dir):
            frame_count = len([f for f in os.listdir(
                frames_dir) if f.endswith('.jpg')])
            f.write(f"- Количество кадров: {frame_count}\n")

        # Статистика по маскам
        masks_dir = "data/masks/1video"
        if os.path.exists(masks_dir):
            mask_count = len([f for f in os.listdir(
                masks_dir) if f.endswith('.png')])
            f.write(f"- Количество масок: {mask_count}\n")

        # ROI информация
        roi_file = "data/roi_1video.npy"
        if os.path.exists(roi_file):
            roi_points = np.load(roi_file)
            f.write(f"- ROI: определен ({len(roi_points)} точек)\n")
        else:
            f.write(f"- ROI: не определен\n")

        f.write("\nСОЗДАННЫЕ ФАЙЛЫ:\n")

        # Проверяем созданные файлы и папки
        created_items = [
            ("flame_unet.pth", "Обученная модель"),
            ("data/flame_preview/", "Примеры детекции"),
            ("data/model_predictions/", "Предсказания модели"),
            ("flame_detection_training_report.txt", "Данный отчет")
        ]

        for item_path, description in created_items:
            if os.path.exists(item_path):
                f.write(f"✓ {item_path} - {description}\n")
            else:
                f.write(f"✗ {item_path} - {description} (не создан)\n")

        f.write("\nРЕКОМЕНДАЦИИ:\n")
        f.write("1. Проверьте качество детекции в папке data/flame_preview/\n")
        f.write(
            "2. При неудовлетворительных результатах настройте параметры в image_processing.py\n")
        f.write("3. Для улучшения модели увеличьте количество эпох обучения\n")
        f.write("4. Рассмотрите возможность ручной корректировки части масок\n")
        f.write("5. Протестируйте модель на новых видео из печи\n")

        f.write(f"\nДата создания отчета: {str(np.datetime64('now'))}\n")

    print(f"✓ Отчет сохранен: {report_path}")


def run_full_pipeline(video_path=None, skip_steps=None):
    """Запуск полного пайплайна обучения"""
    print("🔥 СИСТЕМА ОБУЧЕНИЯ ДЕТЕКЦИИ ПЛАМЕНИ В СТЕКЛОВАРЕННОЙ ПЕЧИ 🔥")
    print("=" * 70)

    if skip_steps is None:
        skip_steps = []

    # Проверяем системные требования
    if not check_requirements():
        print("❌ Проверьте системные требования и повторите запуск")
        return False

    # Создаем директории
    setup_directories()

    # Определяем путь к видео
    if video_path is None:
        video_candidates = [
            "data/videos/1video.avi",
            "data/videos/video.mp4",
            "data/videos/flame_video.avi"
        ]

        for candidate in video_candidates:
            if os.path.exists(candidate):
                video_path = candidate
                break

        if video_path is None:
            print("❌ Видео файл не найден")
            print("Поместите видео в одну из папок:")
            for candidate in video_candidates:
                print(f"  - {candidate}")
            return False

    print(f"📹 Используется видео: {video_path}")

    # Выполняем шаги пайплайна
    steps = [
        (1, "extract_frames", step1_extract_frames, [video_path]),
        (2, "select_roi", step2_select_roi, []),
        (3, "generate_masks", step3_generate_flame_masks, []),
        (4, "inspect_masks", step4_inspect_masks, []),
        (5, "train_model", step5_train_model, []),
        (6, "evaluate_model", step6_evaluate_model, [])
    ]

    completed_steps = []

    for step_num, step_name, step_func, step_args in steps:
        if step_name in skip_steps:
            print(f"⏭️  Пропуск шага {step_num}: {step_name}")
            continue

        print(f"\n🚀 Выполнение шага {step_num}/6...")

        try:
            success = step_func(*step_args)
            if success:
                completed_steps.append(step_name)
                print(f"✅ Шаг {step_num} завершен успешно")
            else:
                print(f"❌ Шаг {step_num} завершился с ошибкой")

                # Спрашиваем пользователя о продолжении
                response = input("Продолжить выполнение? (y/N): ")
                if response.lower() != 'y':
                    break
        except KeyboardInterrupt:
            print("\n⏸️  Выполнение прервано пользователем")
            break
        except Exception as e:
            print(f"❌ Непредвиденная ошибка на шаге {step_num}: {e}")
            break

    # Генерируем финальный отчет
    generate_final_report()

    print(f"\n🎯 РЕЗУЛЬТАТЫ ВЫПОЛНЕНИЯ:")
    print(f"Завершено шагов: {len(completed_steps)}/6")
    print("Завершенные шаги:", ", ".join(completed_steps))

    if len(completed_steps) == 6:
        print("🎉 Все шаги выполнены успешно!")
        print("🔥 Модель детекции пламени готова к использованию!")
        print("\n📁 Проверьте результаты в следующих папках:")
        print("  - data/flame_preview/ - примеры детекции")
        print("  - data/model_predictions/ - предсказания модели")
        print("  - flame_unet.pth - обученная модель")
    else:
        print("⚠️  Пайплайн завершен не полностью")
        print("Проверьте ошибки и повторите выполнение")

    return len(completed_steps) == 6


def main():
    """Основная функция"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Полный пайплайн обучения детекции пламени')
    parser.add_argument('--video', help='Путь к видео файлу')
    parser.add_argument(
        '--skip', nargs='+', help='Пропустить шаги (extract_frames, select_roi, generate_masks, inspect_masks, train_model, evaluate_model)')
    parser.add_argument('--auto', action='store_true',
                        help='Автоматический режим (минимум взаимодействия)')

    args = parser.parse_args()

    # Настройка автоматического режима
    if args.auto:
        print("🤖 Автоматический режим активирован")
        # В автоматическом режиме можно пропустить интерактивные шаги

    # Запускаем пайплайн
    success = run_full_pipeline(
        video_path=args.video,
        skip_steps=args.skip or []
    )

    if success:
        print("\n✨ Готово! Система детекции пламени обучена.")
        print("Используйте flame_unet.pth для детекции пламени в новых видео.")
    else:
        print("\n🔧 Требуется дополнительная настройка.")


if __name__ == "__main__":
    main()
