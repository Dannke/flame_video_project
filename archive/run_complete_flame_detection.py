#!/usr/bin/env python3
"""
Полная система детекции пламени в стекловаренной печи
Поддерживает работу с ROI и без ROI, обработку кадров без пламени
"""

import os
import sys
import subprocess
import numpy as np
import cv2
import glob
from pathlib import Path

# Добавляем src в путь
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)


def print_header(text):
    """Печать заголовка"""
    print(f"\n{'='*60}")
    print(f"{text:^60}")
    print(f"{'='*60}")


def print_step(step_num, total_steps, title):
    """Печать текущего шага"""
    print(f"\n🚀 ШАГ {step_num}/{total_steps}: {title.upper()}")
    print("-" * 50)


def check_python_packages():
    """Проверка установки необходимых пакетов"""
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'torch': 'torch',
        'scipy': 'scipy'
    }

    missing_packages = []

    for package, pip_name in required_packages.items():
        try:
            if package == 'cv2':
                import cv2
            elif package == 'numpy':
                import numpy
            elif package == 'torch':
                import torch
            elif package == 'scipy':
                import scipy
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(pip_name)
            print(f"✗ {package} (отсутствует)")

    if missing_packages:
        print(f"\n❌ Установите недостающие пакеты:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True


def setup_project_structure():
    """Создание структуры проекта"""
    directories = [
        'data',
        'data/videos',
        'data/frames',
        'data/frames/1video',
        'data/masks',
        'data/masks/1video',
        'data/flame_preview',
        'data/model_predictions',
        'models',
        'logs',
        'calibration_results'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ {directory}")


def extract_frames_step(video_path, fps=5):
    """Шаг 1: Извлечение кадров"""
    frames_dir = "data/frames/1video"

    if not os.path.exists(video_path):
        print(f"❌ Видео не найдено: {video_path}")
        return False

    try:
        # Импортируем модуль
        sys.path.insert(0, src_dir)
        from extract_frames import extract_frames
        extract_frames(video_path, frames_dir, fps_out=fps)

        frame_count = len(glob.glob(os.path.join(frames_dir, "*.jpg")))
        if frame_count > 0:
            print(f"✓ Извлечено {frame_count} кадров")
            return True
        else:
            print("❌ Кадры не извлечены")
            return False

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False


def calibrate_detection_step():
    """Шаг 2: Калибровка параметров детекции"""
    frames_dir = "data/frames/1video"
    frames = glob.glob(os.path.join(frames_dir, "*.jpg"))

    if len(frames) == 0:
        print("❌ Кадры не найдены для калибровки")
        return False

    # Выбираем несколько кадров для калибровки
    sample_frames = frames[::len(frames)//5][:5]  # 5 кадров равномерно

    print("Доступные варианты калибровки:")
    print("1. Интерактивная калибровка (рекомендуется)")
    print("2. Использовать параметры по умолчанию")
    print("3. Пропустить калибровку")

    choice = input("Выберите вариант (1-3): ").strip()

    if choice == "1":
        try:
            print("Запуск интерактивной калибровки...")
            print("Используйте трекбары для настройки параметров")
            print("Нажмите 'S' для сохранения, ESC для выхода")

            # Используем правильный путь к калибровщику
            calibrator_path = os.path.join(
                current_dir, "calibrate_flame_detection.py")
            if os.path.exists(calibrator_path):
                cmd = [sys.executable, calibrator_path,
                       "--frame", sample_frames[0]]
                # Не прерываем выполнение при ошибке
                subprocess.run(cmd, check=False)
            else:
                print("❌ Файл calibrate_flame_detection.py не найден")
            return True
        except Exception as e:
            print(
                f"❌ Ошибка калибровки: {e}, используем параметры по умолчанию")
            return True
    elif choice == "2":
        print("✓ Используем параметры по умолчанию")
        return True
    else:
        print("⏭️ Калибровка пропущена")
        return True


def select_roi_step():
    """Шаг 3: Выбор ROI (опционально)"""
    print("Выбор области интереса (ROI):")
    print("1. Выбрать ROI вручную (для ограничения области поиска)")
    print("2. Обрабатывать весь кадр (рекомендуется для печей с двумя отверстиями)")

    choice = input("Выберите вариант (1-2): ").strip()

    if choice == "1":
        frames_dir = "data/frames/1video"
        frames = glob.glob(os.path.join(frames_dir, "*.jpg"))

        if len(frames) == 0:
            print("❌ Кадры не найдены")
            return False

        try:
            print("Запуск выбора ROI...")
            print("Кликайте мышью для выбора точек полигона, ESC - завершить")

            select_roi_path = os.path.join(src_dir, "select_roi.py")
            cmd = [sys.executable, select_roi_path, frames[0]]
            subprocess.run(cmd, check=False)

            if os.path.exists("data/roi_1video.npy"):
                roi = np.load("data/roi_1video.npy")
                print(f"✓ ROI сохранен: {len(roi)} точек")
                return True
            else:
                print("⚠️ ROI не сохранен, будет обрабатываться весь кадр")
                return True
        except Exception as e:
            print(f"❌ Ошибка выбора ROI: {e}")
            return False
    else:
        print("✓ Будет обрабатываться весь кадр")
        return True


def generate_masks_step():
    """Шаг 4: Генерация масок"""
    frames_dir = "data/frames/1video"
    masks_dir = "data/masks/1video"
    roi_file = "data/roi_1video.npy"

    # Проверяем наличие кадров
    frames = glob.glob(os.path.join(frames_dir, "*.jpg"))
    if len(frames) == 0:
        print("❌ Кадры не найдены")
        return False

    # Загружаем ROI если есть
    polygon = None
    if os.path.exists(roi_file):
        polygon = np.load(roi_file).tolist()
        print(f"ROI найден: {len(polygon)} точек")
        use_roi = True
    else:
        print("ROI не найден - обрабатываем весь кадр")
        use_roi = False

    try:
        # Импортируем функцию генерации масок
        sys.path.insert(0, src_dir)
        from gen_pseudo_masks import gen_masks_with_quality_check

        print(f"Генерация масок для {len(frames)} кадров...")
        print("Это может занять несколько минут...")

        gen_masks_with_quality_check(
            frames_dir=frames_dir,
            masks_dir=masks_dir,
            polygon=polygon,
            use_roi=use_roi,  # Явно указываем использование ROI
            save_previews=True,
            preview_count=15
        )

        # Проверяем результат
        masks = glob.glob(os.path.join(masks_dir, "*.png"))
        if len(masks) > 0:
            print(f"✓ Создано {len(masks)} масок")
            return True
        else:
            print("❌ Маски не созданы")
            return False

    except Exception as e:
        print(f"❌ Ошибка генерации масок: {e}")
        import traceback
        traceback.print_exc()
        return False


def inspect_results_step():
    """Шаг 5: Инспекция результатов"""
    masks_dir = "data/masks/1video"
    frames_dir = "data/frames/1video"

    try:
        from inspect_masks import inspect_flame_masks, create_flame_analysis_report

        # Создаем визуализации
        inspect_flame_masks(
            frames_dir=frames_dir,
            masks_dir=masks_dir,
            output_dir="data/flame_preview",
            max_images=20,
            styles=['fire', 'heat', 'contour'],
            save_individual=True
        )

        # Создаем отчет
        create_flame_analysis_report(
            masks_dir=masks_dir,
            output_file="data/flame_preview/detailed_analysis_report.txt"
        )

        print("✓ Создана визуализация результатов")
        print("✓ Создан детальный отчет")

        # Показываем статистику
        show_detection_statistics(masks_dir)

        return True

    except Exception as e:
        print(f"❌ Ошибка инспекции: {e}")
        return False


def show_detection_statistics(masks_dir):
    """Показ статистики детекции"""
    stats_file = os.path.join(masks_dir, "detection_statistics.txt")

    if os.path.exists(stats_file):
        print("\n📊 СТАТИСТИКА ДЕТЕКЦИИ:")
        with open(stats_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Ищем сводную статистику
        in_summary = False
        for line in lines:
            if "СВОДНАЯ СТАТИСТИКА" in line:
                in_summary = True
                continue
            if in_summary and line.strip():
                print(f"  {line.strip()}")


def train_model_step():
    """Шаг 6: Обучение модели"""
    frames_dir = "data/frames/1video"
    masks_dir = "data/masks/1video"

    # Проверяем данные
    frames = glob.glob(os.path.join(frames_dir, "*.jpg"))
    masks = glob.glob(os.path.join(masks_dir, "*.png"))

    if len(frames) == 0 or len(masks) == 0:
        print("❌ Недостаточно данных для обучения")
        return False

    print(f"Данные для обучения: {min(len(frames), len(masks))} пар")

    # Спрашиваем пользователя
    print("Параметры обучения:")
    print("1. Быстрое обучение (4 эпохи, для тестирования)")
    print("2. Стандартное обучение (8 эпох)")
    print("3. Длительное обучение (16 эпох, лучшее качество)")
    print("4. Пропустить обучение")

    choice = input("Выберите вариант (1-4): ").strip()

    if choice == "4":
        print("⏭️ Обучение пропущено")
        return True

    epochs_map = {"1": 4, "2": 8, "3": 16}
    epochs = epochs_map.get(choice, 8)

    try:
        from train_unet import train

        print(f"Запуск обучения на {epochs} эпох...")
        print("Это может занять от 10 минут до нескольких часов в зависимости от вашего оборудования")

        train(
            images_dir=frames_dir,
            masks_dir=masks_dir,
            epochs=epochs,
            batch=4 if torch.cuda.is_available() else 2,  # Меньший батч для экономии памяти
            lr=1e-3,
            size=(256, 256)
        )

        if os.path.exists("flame_unet.pth"):
            print("✓ Модель успешно обучена")
            return True
        else:
            print("❌ Модель не сохранена")
            return False

    except Exception as e:
        print(f"❌ Ошибка обучения: {e}")
        return False


def create_final_report():
    """Создание финального отчета"""
    report_path = "FLAME_DETECTION_REPORT.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Отчет о детекции пламени в стекловаренной печи\n\n")

        # Информация о проекте
        f.write("## Описание проекта\n")
        f.write(
            "Система автоматической детекции пламени в стекловаренной печи с двумя отверстиями.\n\n")

        # Характеристики данных
        frames_dir = "data/frames/1video"
        masks_dir = "data/masks/1video"

        frames = glob.glob(os.path.join(frames_dir, "*.jpg"))
        masks = glob.glob(os.path.join(masks_dir, "*.png"))

        f.write("## Данные\n")
        f.write(f"- Обработано кадров: {len(frames)}\n")
        f.write(f"- Создано масок: {len(masks)}\n")
        f.write(
            f"- ROI: {'Используется' if os.path.exists('data/roi_1video.npy') else 'Весь кадр'}\n\n")

        # Статистика детекции
        stats_file = os.path.join(masks_dir, "detection_statistics.txt")
        if os.path.exists(stats_file):
            f.write("## Статистика детекции\n")
            with open(stats_file, 'r', encoding='utf-8') as stats:
                in_summary = False
                for line in stats:
                    if "СВОДНАЯ СТАТИСТИКА" in line:
                        in_summary = True
                        continue
                    if in_summary and line.strip():
                        f.write(f"- {line.strip()}\n")

        f.write("\n## Созданные файлы\n")

        # Проверяем созданные файлы
        files_to_check = [
            ("data/flame_preview/", "Визуализация результатов"),
            ("flame_unet.pth", "Обученная модель"),
            ("data/masks/1video/", "Маски пламени"),
            ("data/flame_preview/detailed_analysis_report.txt", "Детальный анализ"),
            ("flame_params.json", "Калибровочные параметры")
        ]

        for file_path, description in files_to_check:
            if os.path.exists(file_path):
                f.write(f"✅ {file_path} - {description}\n")
            else:
                f.write(f"❌ {file_path} - {description} (не создан)\n")

        f.write("\n## Рекомендации по использованию\n")
        f.write("1. Проверьте качество детекции в папке `data/flame_preview/`\n")
        f.write("2. При неудовлетворительных результатах запустите калибровку: `python calibrate_flame_detection.py`\n")
        f.write("3. Для новых видео используйте обученную модель `flame_unet.pth`\n")
        f.write(
            "4. Система адаптирована для печей с двумя отверстиями и кадрами без пламени\n")

        f.write(f"\nОтчет создан: {str(np.datetime64('now'))}\n")

    print(f"📄 Финальный отчет создан: {report_path}")


def main():
    """Основная функция"""
    print_header("🔥 ДЕТЕКЦИЯ ПЛАМЕНИ В СТЕКЛОВАРЕННОЙ ПЕЧИ 🔥")

    print("Особенности данной системы:")
    print("• Поддержка печей с двумя отверстиями для пламени")
    print("• Обработка кадров без пламени")
    print("• Возможность работы с ROI или без него")
    print("• Интерактивная калибровка параметров")
    print("• Автоматическая классификация кадров")

    # Проверяем системные требования
    print_step(0, 6, "Проверка системных требований")

    if not check_python_packages():
        print("❌ Установите недостающие пакеты и повторите запуск")
        return

    # Создаем структуру проекта
    print("\n📁 Создание структуры проекта:")
    setup_project_structure()

    # Поиск видео файла
    video_path = None
    video_candidates = [
        "data/videos/1video.avi",
        "data/videos/flame_video.avi",
        "data/videos/video.mp4"
    ]

    for candidate in video_candidates:
        if os.path.exists(candidate):
            video_path = candidate
            break

    if not video_path:
        print("\n📹 Видео файл не найден.")
        print("Поместите видео файл в одну из папок:")
        for candidate in video_candidates:
            print(f"  • {candidate}")

        custom_path = input("\nИли введите путь к видео файлу: ").strip()
        if custom_path and os.path.exists(custom_path):
            video_path = custom_path
        else:
            print("❌ Видео файл не найден. Завершение.")
            return

    print(f"✓ Найден видео файл: {video_path}")

    # Выполняем шаги пайплайна
    steps = [
        ("Извлечение кадров", lambda: extract_frames_step(video_path)),
        ("Калибровка параметров", calibrate_detection_step),
        ("Выбор ROI", select_roi_step),
        ("Генерация масок", generate_masks_step),
        ("Инспекция результатов", inspect_results_step),
        ("Обучение модели", train_model_step)
    ]

    completed_steps = 0

    for i, (step_name, step_func) in enumerate(steps, 1):
        print_step(i, len(steps), step_name)

        try:
            success = step_func()
            if success:
                completed_steps += 1
                print(f"✅ Шаг {i} завершен успешно")
            else:
                print(f"❌ Шаг {i} завершился с ошибкой")

                # Спрашиваем о продолжении
                if i < len(steps):
                    cont = input(
                        "Продолжить выполнение следующих шагов? (y/N): ").strip().lower()
                    if cont != 'y':
                        break

        except KeyboardInterrupt:
            print("\n⏸️ Выполнение прервано пользователем")
            break
        except Exception as e:
            print(f"❌ Непредвиденная ошибка на шаге {i}: {e}")
            cont = input("Продолжить выполнение? (y/N): ").strip().lower()
            if cont != 'y':
                break

    # Создаем финальный отчет
    print_step(7, 7, "Создание финального отчета")
    create_final_report()

    # Итоговое резюме
    print_header("🎯 ИТОГИ ВЫПОЛНЕНИЯ")

    print(f"Завершено шагов: {completed_steps}/{len(steps)}")

    if completed_steps >= 4:  # Минимум до генерации масок
        print("🎉 Основная функциональность готова!")
        print("\n📂 Проверьте результаты:")
        print("  • data/flame_preview/ - примеры детекции")
        print("  • data/masks/1video/ - сгенерированные маски")
        print("  • FLAME_DETECTION_REPORT.md - подробный отчет")

        if os.path.exists("flame_unet.pth"):
            print("  • flame_unet.pth - обученная модель")

    elif completed_steps >= 2:
        print("⚠️ Выполнена частичная настройка")
        print("Для завершения запустите скрипт повторно")
    else:
        print("❌ Базовая настройка не завершена")
        print("Проверьте ошибки и повторите запуск")

    print(f"\n🔥 Система детекции пламени настроена для работы:")
    print("  • С печами, имеющими два отверстия для пламени")
    print("  • С кадрами, где может отсутствовать пламя")
    print("  • С возможностью работы как с ROI, так и без него")

    # Дополнительные команды
    print(f"\n🛠️ Дополнительные команды:")
    print("  • Калибровка: python calibrate_flame_detection.py")
    print("  • Тестирование: python test_flame_detection.py --mode sequence --input data/frames/1video")
    print("  • Обработка нового видео: python src/gen_pseudo_masks.py <frames_dir> <masks_dir>")


if __name__ == "__main__":
    try:
        import torch
        main()
    except ImportError:
        print("PyTorch не установлен. Установите: pip install torch torchvision")

        # Предлагаем вариант без обучения
        print("\nВозможен запуск без обучения модели (только генерация масок):")
        choice = input("Продолжить без PyTorch? (y/N): ").strip().lower()

        if choice == 'y':
            # Убираем шаг обучения
            print("⚠️ Обучение модели будет пропущено")
            main()
        else:
            print("Установите PyTorch и повторите запуск")
