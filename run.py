import os
import sys
import glob
import traceback
from pathlib import Path

# Корень проекта (папка, где лежит run.py)
ROOT = os.path.abspath(os.path.dirname(__file__))

# Добавляем в sys.path корень проекта и стандартные подпапки,
# чтобы импорты типа "from extract_frames import ..." работали
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
FLAME_PREVIEW_DIR = os.path.join(DATA_DIR, "flame_preview")
RESULTS_DIR = os.path.join(ROOT, "results")
RESULTS_MODELS_DIR = os.path.join(RESULTS_DIR, "models")
RESULTS_HISTORY_DIR = os.path.join(RESULTS_DIR, "training_history")

def setup_directories():
    for d in (VIDEOS_DIR, FRAMES_ROOT, MASKS_ROOT, FLAME_PREVIEW_DIR, RESULTS_DIR, RESULTS_MODELS_DIR, RESULTS_HISTORY_DIR):
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

    video_files = sorted([p for p in Path(VIDEOS_DIR).glob("*") if p.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv')])
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
            print(f"✓ Кадры уже извлечены для {vname}: {frames_count} штук (папка {frames_dir})")
            extracted.append((vname, str(vf), frames_dir, frames_count))
            continue

        print(f"Извлечение кадров из {vname} -> {frames_dir}")
        try:
            extract_frames(str(vf), frames_dir, fps_out=5)  # fps_out по умолчанию 5
            frames_count = len(sorted(glob.glob(os.path.join(frames_dir, "*.jpg"))))
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
                print(f"✗ Маски не создались для {video_name} — проверьте логи gen_pseudo_masks")
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
        img_size=(256,256), batch_size=16, epochs=70,
        learning_rate=1e-4, val_split=0.2,
        use_temporal=False, force_gpu=True, save_checkpoints=True
    )
    
    return (model is not None), final_metrics

# -------------- Финальное резюме и helper ------------------------
def print_final_summary(processed_videos, training_success, final_metrics=None):
    print("\n" + "="*60)
    print("ИТОГОВОЕ РЕЗЮМЕ")
    print("="*60)

    if not processed_videos:
        print("Ни одно видео не было успешно обработано.")
    else:
        total_frames = 0
        total_masks = 0
        for name, frames_dir, masks_dir in processed_videos:
            fcount = len(glob.glob(os.path.join(frames_dir, "*.jpg")))
            mcount = len(glob.glob(os.path.join(masks_dir, "*.png")))
            total_frames += fcount
            total_masks += mcount
            print(f"  • {name}: {fcount} кадров, {mcount} масок")

        print("\nОБЩАЯ СТАТИСТИКА:")
        print(f"  • Всего видео обработано: {len(processed_videos)}")
        print(f"  • Всего кадров: {total_frames}")
        print(f"  • Всего масок: {total_masks}")

    print("\nОБУЧЕНИЕ МОДЕЛИ:")
    if training_success:
        print("  ✓ Модель успешно обучена/экспортирована.")
        
        if final_metrics:
            print("\n  ФИНАЛЬНЫЕ МЕТРИКИ:")
            print(f"    • Validation IoU: {final_metrics['final_val_iou']:.4f}")
            print(f"    • Validation Dice: {final_metrics['final_val_dice']:.4f}")
            print(f"    • Лучший IoU: {final_metrics['best_val_iou']:.4f}")
            print(f"    • Лучший Dice: {final_metrics['best_val_dice']:.4f}")
    else:
        print("  ✗ Обучение не выполнено или завершилось ошибкой.")

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
    print("\n=== PIPELINE: Video -> Frames -> Masks -> Previews -> Train ===\n")
    setup_directories()

    # шаг 0 - опциональный автоматический предобработчик (если есть)
    step0_auto_add_videos()

    # шаг 1 - извлечение кадров (если ещё нет)
    extracted = step1_extract_all_frames()
    if not extracted:
        print("Нет извлечённых кадров — останавливаем pipeline.")
        return

    # шаг 2 - генерация масок
    processed = step2_generate_all_masks(extracted, force_regenerate=True)
    if not processed:
        print("Маски не получены ни для одного видео — можно проверить логи.")
        # продолжаем — возможно уже есть маски/кадры, попытка продолжить
    else:
        # шаг 3 - превью
        step3_create_previews(processed, samples_per_video=10)

    # шаг 4 - обучение
    train_ok, final_metrics = step4_train_model(processed)

    # итог
    print_final_summary(processed, train_ok, final_metrics)

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