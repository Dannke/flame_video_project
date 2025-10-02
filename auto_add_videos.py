import os
import glob
from pathlib import Path

# Параметры (настройте при необходимости)
RAW_VIDEOS_DIR = "data/videos"       # куда вы кладёте nvideo.avi
FRAMES_ROOT = "data/frames"
MASKS_ROOT = "data/masks"
EXTRACT_FPS = 7                       # fps при извлечении кадров
FORCE_REPROCESS = True               # если True — перезаписывает существующие папки
SUPPORTED_EXTS = (".mp4", ".avi", ".mov", ".mkv")

# Параметры генерации масок
from mask_config import MASK_PARAMS as GEN_PARAMS

# Импорты функций проекта (предполагается, что они доступны в PYTHONPATH)
try:
    from extract_frames import extract_frames
    from gen_pseudo_masks import gen_masks_temporal
except Exception:
    import sys
    sys.path.append(os.getcwd())
    from extract_frames import extract_frames
    from gen_pseudo_masks import gen_masks_temporal

def sanitize_name(path):
    base = os.path.splitext(os.path.basename(path))[0]
    # Оставляем цифры, буквы, подчеркивания и дефисы
    return "".join(c for c in base if c.isalnum() or c in ("_", "-")).strip()

def already_processed(video_name):
    frames_dir = os.path.join(FRAMES_ROOT, video_name)
    masks_dir = os.path.join(MASKS_ROOT, video_name)
    # считаем обработанным, если есть хотя бы 1 маска и 1 кадр
    frames_exist = os.path.isdir(frames_dir) and len(list(Path(frames_dir).glob("*.jpg"))) > 0
    masks_exist = os.path.isdir(masks_dir) and len(list(Path(masks_dir).glob("*.png"))) > 0
    return frames_exist and masks_exist

def process_video(video_path, extract_fps=EXTRACT_FPS, force=False):
    video_path = str(video_path)
    vname = sanitize_name(video_path)
    frames_dir = os.path.join(FRAMES_ROOT, vname)
    masks_dir = os.path.join(MASKS_ROOT, vname)

    if already_processed(vname) and not force:
        print(f"Пропускаю {video_path} — уже обработано (use FORCE to reprocess).")
        return True

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    print(f"\n=== Обработка видео: {video_path} -> {vname} ===")

    # 1) извлекаем кадры
    print("Извлечение кадров...")
    try:
        extract_frames(video_path, frames_dir, fps_out=extract_fps)
    except Exception as e:
        print(f"Ошибка при извлечении кадров: {e}")
        return False

    # 2) Генерация масок
    print("Генерация масок (gen_masks_temporal)...")
    try:
        gen_masks_temporal(
            frames_dir=frames_dir,
            masks_out_dir=masks_dir,
            polygon=None,
            use_homography=False,
            src_pts=None,
            use_roi=None,
            **GEN_PARAMS
        )
    except Exception as e:
        print(f"Ошибка при генерации масок: {e}")
        return False

    print(f"Готово: фреймы -> {frames_dir}, маски -> {masks_dir}")
    return True

def process_all(raw_dir=RAW_VIDEOS_DIR, force=False):
    raw_dir = os.path.abspath(raw_dir)
    if not os.path.isdir(raw_dir):
        print(f"Папка с исходными видео не найдена: {raw_dir}")
        return

    video_files = sorted([p for p in Path(raw_dir).glob("*") if p.suffix.lower() in SUPPORTED_EXTS])
    if not video_files:
        print(f"Не найдено видео в {raw_dir} (расширения: {SUPPORTED_EXTS})")
        return

    summary = {"processed": [], "skipped": [], "failed": []}
    for v in video_files:
        ok = process_video(v, extract_fps=EXTRACT_FPS, force=force)
        if ok:
            summary["processed"].append(str(v))
        else:
            summary["failed"].append(str(v))

    print("\n=== Итог ===")
    print(f"Обработано: {len(summary['processed'])}")
    print(f"Не удалось: {len(summary['failed'])}")
    if summary['processed']:
        for p in summary['processed']:
            print("  ✓", p)
    if summary['failed']:
        for p in summary['failed']:
            print("  ✗", p)

if __name__ == '__main__':
    # Запускаем обработку всех видео в папке data/videos
    process_all(force=FORCE_REPROCESS)