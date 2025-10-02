import os
import sys
import glob
import traceback
from pathlib import Path
import cv2
import numpy as np

# Корень проекта
ROOT = os.path.abspath(os.path.dirname(__file__))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
src_dir = os.path.join(ROOT, "src")
models_dir = os.path.join(ROOT, "models")
for p in (src_dir, models_dir):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# Пути данных и результатов
DATA_DIR = os.path.join(ROOT, "data")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
FRAMES_ROOT = os.path.join(DATA_DIR, "frames")
MASKS_ROOT = os.path.join(DATA_DIR, "masks")
SHIKHTA_RESULTS_ROOT = os.path.join(DATA_DIR, "shikhta_results")
FLAME_PREVIEW_DIR = os.path.join(DATA_DIR, "flame_preview")
RESULTS_DIR = os.path.join(ROOT, "results")
RESULTS_MODELS_DIR = os.path.join(RESULTS_DIR, "models")
RESULTS_HISTORY_DIR = os.path.join(RESULTS_DIR, "training_history")
RESULTS_SHIKHTA_DIR = os.path.join(RESULTS_DIR, "shikhta_metrics")


def setup_directories():
    """Создание всех необходимых директорий"""
    for d in (VIDEOS_DIR, FRAMES_ROOT, MASKS_ROOT, SHIKHTA_RESULTS_ROOT,
              FLAME_PREVIEW_DIR, RESULTS_DIR, RESULTS_MODELS_DIR,
              RESULTS_HISTORY_DIR, RESULTS_SHIKHTA_DIR):
        os.makedirs(d, exist_ok=True)

# -------------- Step 0: optional auto_add_videos ----------------


def step0_auto_add_videos():
    try:
        from auto_add_videos import process_all as auto_process_all
        print("auto_add_videos найден — запускаю process_all() для data/videos/ ...")
        auto_process_all(force=False)
    except Exception as e:
        print("auto_add_videos не найден или завершился с ошибкой (пропускаем).", e)

# -------------- Step 1: извлечение кадров (если ещё нет) -----------


def step1_extract_all_frames():
    try:
        from extract_frames import extract_frames
    except Exception as e:
        print("Не удалось импортировать extract_frames:", e)
        return []

    video_files = sorted([p for p in Path(VIDEOS_DIR).glob(
        "*") if p.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv')])
    extracted = []

    if not video_files:
        print("Не найдено видео в", VIDEOS_DIR)
        return []

    for vf in video_files:
        vname = vf.name
        basename = Path(vf).stem
        frames_dir = os.path.join(FRAMES_ROOT, basename)
        os.makedirs(frames_dir, exist_ok=True)

        # Считаем существующие кадры
        existing_frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
        if existing_frames:
            frames_count = len(existing_frames)
            print(
                f"✓ Кадры уже извлечены для {vname}: {frames_count} штук (папка {frames_dir})")
            extracted.append((vname, str(vf), frames_dir, frames_count))
            continue

        print(f"Извлечение кадров из {vname} -> {frames_dir}")
        try:
            # fps_out по умолчанию 5
            extract_frames(str(vf), frames_dir, fps_out=5)
            frames_count = len(
                sorted(glob.glob(os.path.join(frames_dir, "*.jpg"))))
            if frames_count > 0:
                print(f"✓ Извлечено {frames_count} кадров из {vname}")
                extracted.append((vname, str(vf), frames_dir, frames_count))
            else:
                print(f"✗ Не удалось извлечь кадры из {vname}")
        except Exception as e:
            print(f"✗ Ошибка при извлечении кадров из {vname}:", e)

    print(f"Результат: извлечены кадры из {len(extracted)} видео")
    return extracted

# -------------- Step 2: генерация масок ---------------------------


def step2_generate_all_masks(extracted_videos, force_regenerate=False):
    processed = []
    try:
        from gen_pseudo_masks import gen_masks_temporal, create_combined_previews
    except Exception as e:
        print("Не удалось импортировать gen_pseudo_masks:", e)
        return processed

    for video_name, video_path, frames_dir, frames_count in extracted_videos:
        basename = Path(frames_dir).stem
        masks_dir = os.path.join(MASKS_ROOT, basename)
        os.makedirs(masks_dir, exist_ok=True)

        masks_exist = len(glob.glob(os.path.join(masks_dir, "*.png"))) > 0
        if masks_exist and not force_regenerate:  # Добавили условие
            cnt = len(glob.glob(os.path.join(masks_dir, "*.png")))
            print(f"✓ Маски уже созданы для {video_name}: {cnt} масок")
            processed.append((video_name, frames_dir, masks_dir))
            continue

        # Удаляем старые маски при force_regenerate
        if force_regenerate and masks_exist:
            import shutil
            shutil.rmtree(masks_dir)
            os.makedirs(masks_dir, exist_ok=True)

        print(f"Генерация масок для {video_name} (кадры {frames_count})")
        try:
            from mask_config import MASK_PARAMS
            gen_masks_temporal(
                frames_dir=frames_dir,
                masks_out_dir=masks_dir,
                polygon=None,
                use_homography=False,
                src_pts=None,
                use_roi=None,
                **MASK_PARAMS,
            )
            masks_count = len(glob.glob(os.path.join(masks_dir, "*.png")))
            if masks_count > 0:
                print(f"✓ Создано {masks_count} масок для {video_name}")
                processed.append((video_name, frames_dir, masks_dir))
            else:
                print(
                    f"✗ Маски не создались для {video_name} — проверьте логи gen_pseudo_masks")
        except Exception as e:
            print(f"✗ Ошибка при генерации масок для {video_name}:", e)
            traceback.print_exc()

    return processed

# -------------- Step 3: создаём превью и комбинированные превью ----


def step3_create_previews(processed_videos, samples_per_video=10):
    try:
        from gen_pseudo_masks import create_combined_previews
    except Exception:
        print("create_combined_previews недоступен — пропускаем шаг 3")
        return False

    all_frames_dirs = [frames for _, frames, _ in processed_videos]
    all_masks_dirs = [masks for _, _, masks in processed_videos]
    video_names = [name for name, _, _ in processed_videos]

    if not all_frames_dirs:
        print("Нет данных для создания превью.")
        return False

    print("Создаём объединённые превью для всех видео...")
    try:
        create_combined_previews(all_frames_dirs, all_masks_dirs, video_names,
                                 output_dir=FLAME_PREVIEW_DIR, samples_per_video=samples_per_video)
        print("✓ Превью созданы")
        return True
    except Exception as e:
        print("✗ Ошибка при создании превью:", e)
        return False

# -------------- Step 4: обучение модели --------------------------


def step4_train_model(processed_videos):
    if not processed_videos:
        print("Нет обработанных видео для обучения")
        return False, None

    all_frames_dirs = [frames for _, frames, _ in processed_videos]
    all_masks_dirs = [masks for _, _, masks in processed_videos]

    try:
        import models.flame_segmentation_model as fmod
    except Exception as e:
        print("Не удалось импортировать модуль модели:", e)
        return False, None

    print("Запуск обучения на всех данных...")
    model, history, final_metrics = fmod.train_and_export(  # Получаем метрики
        frames_dir=all_frames_dirs,
        masks_dir=all_masks_dirs,
        img_size=(256, 256), batch_size=16, epochs=10,
        learning_rate=1e-4, val_split=0.2,
        use_temporal=False, force_gpu=True, save_checkpoints=True
    )

    return (model is not None), final_metrics

# -------------- Step 5: анализ шихты (NEW) ---------------------------


def step5_analyze_shikhta(extracted_videos, force_reanalyze=False):
    """Анализ шихты для всех видео"""
    try:
        from shikhta_analysis import analyze_video_shikhta
    except Exception as e:
        print("Не удалось импортировать shikhta_analysis:", e)
        print("Убедитесь, что shikhta_analysis.py находится в корне проекта или src/")
        return []

    shikhta_results = []

    for video_name, video_path, frames_dir, frames_count in extracted_videos:
        basename = Path(frames_dir).stem
        shikhta_output = os.path.join(SHIKHTA_RESULTS_ROOT, basename)
        metrics_file = os.path.join(shikhta_output, f"{basename}_metrics.json")

        # Проверка существующих результатов
        if os.path.exists(metrics_file) and not force_reanalyze:
            print(f"✓ Анализ шихты уже выполнен для {video_name}")
            shikhta_results.append((video_name, shikhta_output, metrics_file))
            continue

        print(f"\n{'='*60}")
        print(f"АНАЛИЗ ШИХТЫ: {video_name}")
        print(f"{'='*60}")

        try:
            from polygon_gui import prompt_polygon_on_image, load_saved_polygon

            # Попытка загрузить ранее сохранённый полигон для этого видео
            polygons_save_dir = os.path.join(shikhta_output, "polygons")
            saved_poly = load_saved_polygon(polygons_save_dir, basename)
            polygon = None
            if saved_poly:
                # Попытаемся определить — сохранённый полигон уже в target-координатах или в исходном разрешении?
                try:
                    coords = np.array(saved_poly, dtype=np.int64)
                    target_w, target_h = 928, 576
                    if coords[:, 0].max() > target_w or coords[:, 1].max() > target_h:
                        # Требуется масштабирование — читаем первый кадр
                        imgs = sorted([os.path.join(frames_dir, p) for p in os.listdir(frames_dir)
                                       if p.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))])
                        first_frame_path = imgs[0] if imgs else None
                        if first_frame_path:
                            img = cv2.imread(first_frame_path)
                            orig_h, orig_w = img.shape[:2]
                            sx = target_w / orig_w
                            sy = target_h / orig_h
                            polygon = [(int(round(x * sx)), int(round(y * sy)))
                                       for (x, y) in saved_poly]
                        else:
                            polygon = saved_poly
                    else:
                        polygon = saved_poly
                except Exception:
                    polygon = saved_poly
            else:
                # Если нет сохранённого полигона — спрашиваем GUI и масштабируем результат
                imgs = sorted([os.path.join(frames_dir, p) for p in os.listdir(frames_dir)
                               if p.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))])
                first_frame_path = imgs[0] if imgs else None
                if first_frame_path:
                    img = cv2.imread(first_frame_path)
                    polygon_raw = prompt_polygon_on_image(img, default_polygon=None,
                                                          video_name=basename, save_dir=polygons_save_dir)
                    if polygon_raw:
                        orig_h, orig_w = img.shape[:2]
                        sx = 928 / orig_w
                        sy = 576 / orig_h
                        polygon = [(int(round(x * sx)), int(round(y * sy)))
                                   for (x, y) in polygon_raw]
                    else:
                        polygon = None
                else:
                    polygon = None

            summary = analyze_video_shikhta(
                frames_dir=frames_dir,
                output_dir=shikhta_output,
                polygon=polygon,
                save_visualizations=True,
                save_every_n=20
            )

            if summary:
                shikhta_results.append(
                    (video_name, shikhta_output, metrics_file))

                # Копируем сводную статистику в общую папку результатов
                import json
                summary_copy = os.path.join(
                    RESULTS_SHIKHTA_DIR, f"{basename}_summary.json")
                with open(summary_copy, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
            else:
                print(f"✗ Не удалось получить статистику для {video_name}")

        except Exception as e:
            print(f"✗ Ошибка при анализе шихты для {video_name}:", e)
            traceback.print_exc()

    return shikhta_results

# -------------- Финальное резюме и helper ------------------------


def print_final_summary(extracted_videos, processed_videos, shikhta_results, training_success, final_metrics=None):
    """
    Печать итоговой сводки.

    Параметры:
      - extracted_videos : список/итерация извлечённых элементов (может быть список имён или кортежей)
      - processed_videos : список кортежей (name, frames_dir, masks_dir) — как раньше
      - shikhta_results   : структура с результатами анализа шихты (может быть списком или словарём)
      - training_success : bool — статус обучения
      - final_metrics     : dict или None — финальные метрики модели
    """
    print("\n" + "="*60)
    print("ИТОГОВОЕ РЕЗЮМЕ")
    print("="*60)

    # --- Извлечённые видео / артефакты ---
    print("\nИЗВЛЕЧЁННЫЕ ВИДЕО / АРТЕФАКТЫ:")
    if not extracted_videos:
        print("  • Ничего не извлечено.")
    else:
        try:
            print(f"  • Всего извлечено: {len(extracted_videos)}")
            # Выводим имена (или первое поле кортежа)
            for item in extracted_videos:
                if isinstance(item, (list, tuple)) and item:
                    name = item[0]
                else:
                    name = str(item)
                print(f"    - {name}")
        except Exception:
            print(f"  • {extracted_videos}")

    # --- Результаты обработки видео (кадры/маски) ---
    print("\nОБРАБОТАННЫЕ ВИДЕО:")
    if not processed_videos:
        print("  Ни одно видео не было успешно обработано.")
    else:
        total_frames = 0
        total_masks = 0
        for entry in processed_videos:
            # ожидаем (name, frames_dir, masks_dir) — но допускаем гибкость
            try:
                name, frames_dir, masks_dir = entry
            except Exception:
                # если неподходящий формат — печатаем содержимое
                print(f"  • {entry}")
                continue
            fcount = len(glob.glob(os.path.join(frames_dir, "*.jpg")))
            mcount = len(glob.glob(os.path.join(masks_dir, "*.png")))
            total_frames += fcount
            total_masks += mcount
            print(f"  • {name}: {fcount} кадров, {mcount} масок")

        print("\nОБЩАЯ СТАТИСТИКА:")
        print(f"  • Всего видео обработано: {len(processed_videos)}")
        print(f"  • Всего кадров: {total_frames}")
        print(f"  • Всего масок: {total_masks}")

    # --- Результаты анализа шихты (кратко) ---
    print("\nРЕЗУЛЬТАТЫ АНАЛИЗА ШИХТЫ:")
    if not shikhta_results:
        print("  • Нет результатов анализа шихты.")
    else:
        try:
            # Если это список словарей/кортежей — выведем количество
            print(f"  • Всего результатов шихты: {len(shikhta_results)}")
        except Exception:
            print(f"  • {shikhta_results}")

    # --- Обучение модели ---
    print("\nОБУЧЕНИЕ МОДЕЛИ:")
    if training_success:
        print("  ✓ Модель успешно обучена/экспортирована.")
        if final_metrics:
            print("\n  ФИНАЛЬНЫЕ МЕТРИКИ:")
            try:
                print(f"    • Validation IoU: {final_metrics['final_val_iou']:.4f}")
                print(f"    • Validation Dice: {final_metrics['final_val_dice']:.4f}")
                print(f"    • Лучший IoU: {final_metrics['best_val_iou']:.4f}")
                print(f"    • Лучший Dice: {final_metrics['best_val_dice']:.4f}")
            except Exception:
                # Если структура иная — печатаем словарь целиком
                print(f"    • {final_metrics}")
    else:
        print("  ✗ Обучение не выполнено или завершилось ошибкой.")

    # --- Папки с результатами ---
    print("\nРЕЗУЛЬТАТЫ СОХРАНЕНЫ В:")
    if os.path.isdir(RESULTS_MODELS_DIR):
        items = [p for p in os.listdir(RESULTS_MODELS_DIR) if os.path.isfile(os.path.join(RESULTS_MODELS_DIR, p))]
        if items:
            print(f"  • {RESULTS_MODELS_DIR} ({len(items)} файлов)")
    if os.path.isdir(RESULTS_HISTORY_DIR):
        items = [p for p in os.listdir(RESULTS_HISTORY_DIR) if os.path.isfile(os.path.join(RESULTS_HISTORY_DIR, p))]
        if items:
            print(f"  • {RESULTS_HISTORY_DIR} ({len(items)} файлов)")

    print("="*60)

# -------------- main pipeline ------------------------------------


def main():
    """Главный пайплайн интегрированного проекта"""
    print("\n" + "="*70)
    print("ИНТЕГРИРОВАННЫЙ ПАЙПЛАЙН: ПЛАМЯ + ШИХТА")
    print("="*70)
    print("Этапы:")
    print("  1. Извлечение кадров из видео")
    print("  2. Генерация масок пламени")
    print("  3. Анализ шихты на кадрах")
    print("  4. Создание превью пламени")
    print("  5. Обучение модели сегментации пламени")
    print("="*70 + "\n")

    setup_directories()

    # Шаг 0 - автоматический препроцессинг (если есть)
    step0_auto_add_videos()

    # Шаг 1 - извлечение кадров
    print("\n[ШАГ 1] Извлечение кадров из видео...")
    extracted = step1_extract_all_frames()
    if not extracted:
        print("✗ Нет извлечённых кадров — останавливаем pipeline.")
        return

    # Шаг 2 - генерация масок пламени
    print("\n[ШАГ 2] Генерация масок пламени...")
    processed_flame = step2_generate_all_masks(
        extracted, force_regenerate=False)

    # Шаг 3 - превью пламени
    if processed_flame:
        print("\n[ШАГ 3] Создание превью пламени...")
        step3_create_previews(processed_flame, samples_per_video=10)

    # Шаг 4 - обучение модели пламени
    print("\n[ШАГ 4] Обучение модели сегментации пламени...")
    train_ok, final_metrics = step4_train_model(processed_flame)

    # Шаг 5 - анализ шихты
    print("\n[ШАГ 5] Анализ шихты...")
    shikhta_results = step5_analyze_shikhta(extracted, force_reanalyze=False)

    # Итоговая сводка
    print_final_summary(extracted, processed_flame, shikhta_results,
                        train_ok, final_metrics)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрервано пользователем")
        sys.exit(0)
    except Exception as e:
        print("\nНепредвиденная ошибка в run.py:", e)
        traceback.print_exc()
        sys.exit(1)
