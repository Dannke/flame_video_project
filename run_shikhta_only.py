"""
Упрощенный запуск только анализа шихты
Без обучения модели пламени
"""

from pathlib import Path
import os
import sys
import glob
from pathlib import Path
import cv2
import json

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

src_dir = os.path.join(ROOT, "src")
if os.path.isdir(src_dir) and src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Импорт модуля анализа (с детальной диагностикой)
analyze_video_shikhta = None

# Попытка 1: импорт из src/
try:
    sys.path.insert(0, src_dir)
    from shikhta_analysis import analyze_video_shikhta
    print("✓ Модуль загружен из src/")
except ImportError:
    pass

# Если ничего не сработало
if analyze_video_shikhta is None:
    print("Ошибка: не найден модуль shikhta_analysis.py")
    print(f"\nПроверьте:")
    print(
        f"  1. Файл существует в: {os.path.join(src_dir, 'shikhta_analysis.py')}")
    print(
        f"     Результат: {os.path.exists(os.path.join(src_dir, 'shikhta_analysis.py'))}")
    print(f"  2. Или в корне: {os.path.join(ROOT, 'shikhta_analysis.py')}")
    print(
        f"     Результат: {os.path.exists(os.path.join(ROOT, 'shikhta_analysis.py'))}")
    print(f"\n  Текущие пути поиска Python:")
    for i, p in enumerate(sys.path[:5], 1):
        print(f"     {i}. {p}")
    sys.exit(1)

# Пути
DATA_DIR = os.path.join(ROOT, "data")
FRAMES_ROOT = os.path.join(DATA_DIR, "frames")
SHIKHTA_RESULTS_ROOT = os.path.join(DATA_DIR, "shikhta_results")
RESULTS_SHIKHTA_DIR = os.path.join(ROOT, "results", "shikhta_metrics")


def setup_directories():
    os.makedirs(SHIKHTA_RESULTS_ROOT, exist_ok=True)
    os.makedirs(RESULTS_SHIKHTA_DIR, exist_ok=True)


def find_frame_directories():
    """Находит все директории с кадрами"""
    if not os.path.exists(FRAMES_ROOT):
        print(f"Директория {FRAMES_ROOT} не существует")
        return []

    frame_dirs = []
    for item in os.listdir(FRAMES_ROOT):
        full_path = os.path.join(FRAMES_ROOT, item)
        if os.path.isdir(full_path):
            # Проверяем наличие кадров
            jpg_files = list(Path(full_path).glob("*.jpg"))
            if jpg_files:
                frame_dirs.append((item, full_path, len(jpg_files)))

    return frame_dirs


def analyze_all_videos(force_reanalyze=False, save_every_n=20,
                       use_parallel=True, max_workers=8):
    """Анализ шихты для всех найденных видео"""
    frame_dirs = find_frame_directories()

    if not frame_dirs:
        print("Не найдено директорий с кадрами в", FRAMES_ROOT)
        print("\nСначала запустите:")
        print("  python run.py  (для полного пайплайна)")
        print("или извлеките кадры вручную в data/frames/")
        return

    print(f"Найдено {len(frame_dirs)} видео для анализа:\n")
    for name, path, count in frame_dirs:
        print(f"  • {name}: {count} кадров")

    print("\n" + "="*70)
    print("ЗАПУСК АНАЛИЗА ШИХТЫ")
    if use_parallel:
        print(f"Режим: параллельная обработка ({max_workers} потоков)")
    else:
        print("Режим: последовательная обработка")
    print("="*70 + "\n")

    results = []

    for video_name, frames_dir, frames_count in frame_dirs:
        shikhta_output = os.path.join(SHIKHTA_RESULTS_ROOT, video_name)
        metrics_file = os.path.join(
            shikhta_output, f"{video_name}_metrics.json")

        # Проверка существующих результатов
        if os.path.exists(metrics_file) and not force_reanalyze:
            print(
                f"✓ Анализ уже выполнен для {video_name} (используйте force_reanalyze=True для повтора)")
            results.append((video_name, shikhta_output, metrics_file))
            continue

        print(f"\n{'='*70}")
        print(f"ОБРАБОТКА: {video_name}")
        print(f"{'='*70}")

        try:
            # --- POLYGON: попытка загрузить сохранённый полигон, иначе предложить GUI ---
            from polygon_gui import prompt_polygon_on_image, load_saved_polygon

            polygons_save_dir = os.path.join(shikhta_output, "polygons")
            saved_poly = load_saved_polygon(polygons_save_dir, video_name)
            if saved_poly:
                polygon = saved_poly
            else:
                # Попробуем найти первый кадр в frames_dir
                first_frame_path = None
                for ext in (".jpg", ".png", ".jpeg", ".bmp"):
                    candidate = os.path.join(frames_dir, f"frame_0001{ext}")
                    if os.path.exists(candidate):
                        first_frame_path = candidate
                        break
                if not first_frame_path:
                    imgs = sorted([os.path.join(frames_dir, p) for p in os.listdir(frames_dir)
                                   if p.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))])
                    first_frame_path = imgs[0] if imgs else None

                if first_frame_path:
                    img = cv2.imread(first_frame_path)
                    polygon_raw = prompt_polygon_on_image(img, default_polygon=None,
                                                         video_name=video_name, save_dir=polygons_save_dir)
                    # Если пользователь вернул полигон — масштабируем его к размерам анализатора
                    if polygon_raw:
                        orig_h, orig_w = img.shape[:2]
                        target_w, target_h = 928, 576  # должно совпадать с ShikhtaAnalyzer.target_size по умолчанию
                        sx = target_w / orig_w
                        sy = target_h / orig_h
                        polygon = [(int(round(x * sx)), int(round(y * sy))) for (x, y) in polygon_raw]
                    else:
                        polygon = None
                else:
                    print(f"Не найден первый кадр в {frames_dir} — используем дефолтный полигон.")
                    polygon = None

            # Если загрузили ранее сохранённый полигон — убедимся, что он в координатах target_size.
            if polygon is None and saved_poly:
                # saved_poly возможно было сохранено в оригинальном разрешении — делаем попытку адаптировать.
                # Попытка определения: если координаты превышают target — считаем, что это оригинал и масштабируем.
                try:
                    coords = np.array(saved_poly, dtype=np.int64)
                    target_w, target_h = 928, 576
                    max_x = int(coords[:, 0].max())
                    max_y = int(coords[:, 1].max())
                    # Если сохранённый полигон явно больше target, считаем его в разрешении оригинала.
                    if max_x > target_w or max_y > target_h:
                        # Для масштабирования нам нужен оригинальный кадр; попытаемся прочитать первый кадр
                        if 'img' not in locals() or img is None:
                            # пробуем снова найти первый кадр и загрузить
                            if first_frame_path and os.path.exists(first_frame_path):
                                img = cv2.imread(first_frame_path)
                        if img is not None:
                            orig_h, orig_w = img.shape[:2]
                            sx = target_w / orig_w
                            sy = target_h / orig_h
                            polygon = [(int(round(x * sx)), int(round(y * sy))) for (x, y) in saved_poly]
                        else:
                            # без кадра — используем saved_poly как есть (может быть уже в target_coords)
                            polygon = saved_poly
                    else:
                        # Считаем, что polygon уже в target-координатах
                        polygon = saved_poly
                except Exception:
                    polygon = saved_poly

            summary = analyze_video_shikhta(
                frames_dir=frames_dir,
                output_dir=shikhta_output,
                polygon=polygon,
                save_visualizations=True,
                save_every_n=save_every_n,
                use_parallel=use_parallel,
                max_workers=max_workers
            )

            if summary:
                results.append((video_name, shikhta_output, metrics_file))

                # Копируем в общую папку результатов
                import json
                summary_copy = os.path.join(
                    RESULTS_SHIKHTA_DIR, f"{video_name}_summary.json")
                with open(summary_copy, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"✗ Ошибка при анализе {video_name}:", e)
            import traceback
            traceback.print_exc()

    # Итоговая сводка
    print_summary(results)


def print_summary(results):
    """Вывод итоговой статистики"""
    import json

    print("\n" + "="*70)
    print("ИТОГОВАЯ СТАТИСТИКА ПО ВСЕМ ВИДЕО")
    print("="*70)

    if not results:
        print("Нет результатов анализа")
        return

    print(f"\nВсего проанализировано видео: {len(results)}\n")

    # Агрегированная статистика
    all_left_means = []
    all_right_means = []

    for video_name, output_dir, metrics_file in results:
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                summary = data.get('summary', {})

            total_frames = summary.get('total_frames', 0)
            left = summary.get('left', {})
            right = summary.get('right', {})

            all_left_means.append(left.get('mean', 0))
            all_right_means.append(right.get('mean', 0))

            print(f"{video_name}:")
            print(f"  Кадров: {total_frames}")
            print(
                f"  Левая часть:  {left.get('mean', 0):6.2f}% (мин: {left.get('min', 0):5.2f}%, макс: {left.get('max', 0):5.2f}%)")
            print(
                f"  Правая часть: {right.get('mean', 0):6.2f}% (мин: {right.get('min', 0):5.2f}%, макс: {right.get('max', 0):5.2f}%)")
            print()

        except Exception as e:
            print(f"✗ Не удалось прочитать метрики для {video_name}: {e}\n")

    # Общая статистика по всем видео
    if all_left_means and all_right_means:
        import numpy as np
        print("="*70)
        print("АГРЕГИРОВАННАЯ СТАТИСТИКА ПО ВСЕМ ВИДЕО:")
        print("="*70)
        print(
            f"  Средняя левая часть:  {np.mean(all_left_means):.2f}% (σ={np.std(all_left_means):.2f}%)")
        print(
            f"  Средняя правая часть: {np.mean(all_right_means):.2f}% (σ={np.std(all_right_means):.2f}%)")

    print("\nРезультаты сохранены в:")
    print(f"  • Детальные результаты: {SHIKHTA_RESULTS_ROOT}")
    print(f"  • Сводные метрики: {RESULTS_SHIKHTA_DIR}")
    print("="*70 + "\n")


def main():
    print("\n" + "="*70)
    print("АНАЛИЗ ШИХТЫ (БЕЗ ОБУЧЕНИЯ МОДЕЛИ)")
    print("="*70 + "\n")

    setup_directories()

    # Параметры
    force_reanalyze = False  # Установите True для повторного анализа
    save_every_n = 20        # Сохранять каждый N-й кадр

    analyze_all_videos(force_reanalyze=force_reanalyze,
                       save_every_n=save_every_n)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрервано пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"\nОшибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
