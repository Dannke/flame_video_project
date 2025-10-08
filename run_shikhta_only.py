# run_shikhta_only.py
"""
run_shikhta_only.py — анализ шихты с GUI выбора полигона (сначала основной), затем настройкой перспективы.
Порядок:
 1) выбор/загрузка основного полигона (GUI или файл)
 2) настройка/загрузка перспективы (модуль или GUI), сохранение persp-полигона
 3) анализ шихты
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Добавляем src при наличии
src_dir = os.path.join(ROOT, "src")
if os.path.isdir(src_dir) and src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Импорт модулей
try:
    from shikhta_analysis import analyze_video_shikhta
except ImportError:
    print("Ошибка: не найден модуль анализа шихты (shikhta_analysis.py)")
    sys.exit(1)

# Импорт стандартной перспективы (4 точки)
try:
    from perspective_transform import load_or_setup_perspective, PerspectiveTransformer
    PERSPECTIVE_AVAILABLE = True
except ImportError:
    PERSPECTIVE_AVAILABLE = False
    load_or_setup_perspective = None
    PerspectiveTransformer = None
    print("ℹ Модуль perspective_transform не найден (4-точечная перспектива недоступна)")

# Импорт шестиугольной перспективы (6 точек)
try:
    from perspective_transform_hexagon import (
        HexagonPerspectiveTransformer,
        setup_hexagon_perspective_gui
    )
    HEXAGON_PERSPECTIVE_AVAILABLE = True
except ImportError:
    HEXAGON_PERSPECTIVE_AVAILABLE = False
    HexagonPerspectiveTransformer = None
    setup_hexagon_perspective_gui = None
    print("ℹ Модуль perspective_transform_hexagon не найден (6-точечная перспектива недоступна)")

# Импорт GUI полигона
try:
    from polygon_gui import prompt_polygon_on_image, load_saved_polygon
    POLYGON_GUI_AVAILABLE = True
except ImportError:
    POLYGON_GUI_AVAILABLE = False
    prompt_polygon_on_image = None
    load_saved_polygon = None
    print("ℹ Модуль polygon_gui не найден")

# Пути по умолчанию
DATA_DIR = os.path.join(ROOT, "data")
FRAMES_ROOT = os.path.join(DATA_DIR, "frames")
SHIKHTA_RESULTS_ROOT = os.path.join(DATA_DIR, "shikhta_results")
RESULTS_SHIKHTA_DIR = os.path.join(ROOT, "results", "shikhta_metrics")


def setup_directories():
    os.makedirs(SHIKHTA_RESULTS_ROOT, exist_ok=True)
    os.makedirs(RESULTS_SHIKHTA_DIR, exist_ok=True)


def find_frame_directories():
    """Находит директории с извлечёнными кадрами"""
    if not os.path.exists(FRAMES_ROOT):
        print(f"Директория {FRAMES_ROOT} не существует")
        return []

    frame_dirs = []
    for item in sorted(os.listdir(FRAMES_ROOT)):
        full_path = os.path.join(FRAMES_ROOT, item)
        if os.path.isdir(full_path):
            jpg_files = list(Path(full_path).glob("*.jpg"))
            if jpg_files:
                frame_dirs.append((item, full_path, len(jpg_files)))
    return frame_dirs


def _save_polygon_to_file(polygon, save_path):
    """Сохранение полигона в JSON (список [ [x,y], ... ])"""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump({'polygon': [list(map(int, p))
                      for p in polygon]}, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print("Ошибка при сохранении полигона:", e)
        return False


def _load_polygon_from_file(load_path):
    try:
        if not os.path.exists(load_path):
            return None
        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            poly = data.get('polygon')
            if poly:
                return np.array(poly, dtype=np.int32)
    except Exception as e:
        print("Ошибка при загрузке полигона:", e)
    return None


def prepare_main_polygon_via_gui(first_frame_path, polygons_save_dir, video_name):
    """
    Шаг 1: загрузка/выбор основного полигона (GUI или файл)
    Возвращает main_polygon (np.ndarray или None)
    """
    main_polygon_path = os.path.join(
        polygons_save_dir, f"{video_name}_polygon.json")

    # Попытка загрузить средствами polygon_gui.load_saved_polygon (если реализовано)
    main_polygon = None
    if POLYGON_GUI_AVAILABLE and load_saved_polygon:
        try:
            loaded = load_saved_polygon(polygons_save_dir, video_name)
            if loaded is not None:
                main_polygon = np.array(loaded, dtype=np.int32)
        except Exception:
            pass

    # fallback — из файла
    if main_polygon is None:
        main_polygon = _load_polygon_from_file(main_polygon_path)

    # Если нет и есть GUI — попросим пользователя нарисовать
    if main_polygon is None and POLYGON_GUI_AVAILABLE and prompt_polygon_on_image:
        print("Откроется GUI для выбора основного полигона (шестиугольника).")
        img = cv2.imread(first_frame_path)
        try:
            polygon_raw = prompt_polygon_on_image(
                img, default_polygon=None, video_name=video_name, save_dir=polygons_save_dir)
        except Exception as e:
            print("Ошибка в prompt_polygon_on_image (main):", e)
            polygon_raw = None

        if polygon_raw:
            main_polygon = np.array(polygon_raw, dtype=np.int32)
            if _save_polygon_to_file(main_polygon, main_polygon_path):
                print(f"✓ Основной полигон сохранён: {main_polygon_path}")

    return main_polygon


def setup_perspective_after_main_polygon(video_name, first_frame_path, polygons_save_dir,
                                         main_polygon=None, use_perspective=True, 
                                         auto_setup_perspective=True, perspective_method='hexagon'):
    """
    Шаг 2: настройка перспективы — выполняется ПОСЛЕ выбора основного полигона.
    Возвращает (perspective_transformer or None, persp_polygon or None)
    
    Args:
        perspective_method: 'standard' (4 точки) или 'hexagon' (6 точек)
    """
    if not use_perspective:
        return None, None
    
    perspective_transformer = None
    persp_polygon = None
    
    # ========== МЕТОД: HEXAGON (6 ТОЧЕК) ==========
    if perspective_method == 'hexagon':
        if not HEXAGON_PERSPECTIVE_AVAILABLE:
            print("⚠ Модуль perspective_transform_hexagon недоступен")
            print("  Убедитесь, что файл perspective_transform_hexagon.py находится в директории проекта")
            print("  Попытка использовать стандартный метод...")
            perspective_method = 'standard'
        else:
            print(f"✓ Модуль HexagonPerspectiveTransformer доступен")
            
            # Путь к конфигу шестиугольной перспективы
            hexagon_config_path = os.path.join(
                polygons_save_dir, f"{video_name}_hexagon_perspective.json"
            )
            
            # Попытка загрузить существующий конфиг
            if os.path.exists(hexagon_config_path):
                try:
                    perspective_transformer = HexagonPerspectiveTransformer.load_config(hexagon_config_path)
                    print(f"✓ Загружена 6-точечная конфигурация: {hexagon_config_path}")
                    return perspective_transformer, None
                except Exception as e:
                    print(f"⚠ Ошибка загрузки 6-точечной конфигурации: {e}")
            
            # Если нет сохраненного конфига и нужна авто-настройка
            if auto_setup_perspective:
                print("Откроется GUI для настройки 6-точечной перспективы")
                img = cv2.imread(first_frame_path)
                if img is None:
                    print(f"✗ Не удалось загрузить кадр: {first_frame_path}")
                    return None, None
                
                # Показываем основной полигон для ориентира (если есть)
                if main_polygon is not None:
                    try:
                        vis = img.copy()
                        cv2.polylines(vis, [main_polygon], True, (0, 255, 0), 2)
                        cv2.putText(vis, "Main polygon (reference)", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        img = vis
                    except Exception as e:
                        print(f"⚠ Не удалось отобразить основной полигон: {e}")
                
                try:
                    points = setup_hexagon_perspective_gui(img)
                    
                    if points:
                        perspective_transformer = HexagonPerspectiveTransformer(
                            points,
                            dst_width=800,
                            dst_height=600
                        )
                        # Сохраняем конфиг
                        os.makedirs(polygons_save_dir, exist_ok=True)
                        perspective_transformer.save_config(hexagon_config_path)
                        print(f"✓ 6-точечная конфигурация сохранена: {hexagon_config_path}")
                        return perspective_transformer, None
                    else:
                        print("⚠ Настройка 6-точечной перспективы отменена")
                        return None, None
                        
                except Exception as e:
                    print(f"✗ Ошибка при настройке 6-точечной перспективы: {e}")
                    import traceback
                    traceback.print_exc()
                    return None, None
            else:
                print("ℹ Авто-настройка перспективы отключена")
                return None, None
    
    # ========== МЕТОД: STANDARD (4 ТОЧКИ) ==========
    if perspective_method == 'standard':
        if not PERSPECTIVE_AVAILABLE:
            print("⚠ Модуль perspective_transform недоступен")
            print("  Убедитесь, что файл perspective_transform.py находится в директории проекта")
            return None, None
        
        print(f"✓ Модуль PerspectiveTransformer доступен")
        
        # Путь к стандартному конфигу
        standard_config_path = os.path.join(
            polygons_save_dir, f"{video_name}_perspective.json"
        )
        
        # Попытка загрузить существующий конфиг
        if os.path.exists(standard_config_path):
            try:
                perspective_transformer = PerspectiveTransformer.load_config(standard_config_path)
                print(f"✓ Загружена 4-точечная конфигурация: {standard_config_path}")
                return perspective_transformer, None
            except Exception as e:
                print(f"⚠ Ошибка загрузки 4-точечной конфигурации: {e}")
        
        # Если нет сохраненного конфига и есть load_or_setup_perspective
        if auto_setup_perspective and load_or_setup_perspective:
            try:
                perspective_transformer = load_or_setup_perspective(
                    video_name, 
                    first_frame_path,
                    config_dir=polygons_save_dir
                )
                if perspective_transformer:
                    print("✓ 4-точечная перспектива настроена")
                    return perspective_transformer, None
                else:
                    print("⚠ Настройка 4-точечной перспективы отменена")
                    return None, None
            except Exception as e:
                print(f"✗ Ошибка при настройке 4-точечной перспективы: {e}")
                import traceback
                traceback.print_exc()
                return None, None
    
    return None, None


def analyze_all_videos(force_reanalyze=False, save_every_n=20,
                       use_parallel=True, max_workers=8,
                       use_perspective=True, auto_setup_perspective=True, perspective_method='hexagon'):
    """
    Анализ шихты для всех видео — первым шагом всегда основной полигон.
    """
    frame_dirs = find_frame_directories()
    if not frame_dirs:
        print("Не найдено директорий с кадрами в", FRAMES_ROOT)
        return []

    print(f"Найдено {len(frame_dirs)} видео для анализа:")
    for name, path, count in frame_dirs:
        print(f"  • {name}: {count} кадров")

    print("\nЗапуск анализа (перспектива: {})".format(
        "вкл" if use_perspective and PERSPECTIVE_AVAILABLE else "выкл"))

    results = []

    for video_name, frames_dir, frames_count in frame_dirs:
        shikhta_output = os.path.join(SHIKHTA_RESULTS_ROOT, video_name)
        metrics_file = os.path.join(
            shikhta_output, f"{video_name}_metrics.json")
        polygons_save_dir = os.path.join(shikhta_output, "polygons")

        if os.path.exists(metrics_file) and not force_reanalyze:
            print(
                f"✓ Анализ уже есть для {video_name}, пропускаю (use --force для повтора)")
            results.append((video_name, shikhta_output, metrics_file))
            continue

        print("\n" + "="*50)
        print(f"Обработка видео: {video_name}")
        print("="*50)

        # Первый кадр для GUI
        imgs = sorted([os.path.join(frames_dir, p) for p in os.listdir(frames_dir)
                       if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
        first_frame_path = imgs[0] if imgs else None
        if first_frame_path is None:
            print("  ✗ Нет кадров — пропускаю")
            continue

        os.makedirs(polygons_save_dir, exist_ok=True)

        # === ШАГ 1: основной полигон ===
        main_polygon = prepare_main_polygon_via_gui(
            first_frame_path, polygons_save_dir, video_name)
        if main_polygon is None:
            print(
                "  ⚠ Основной полигон не задан — будет использован дефолт (если есть) в анализаторе")

        # === ШАГ 2: настройка перспективы (после выбора main polygon) ===
        perspective_transformer = None
        persp_polygon = None
        if use_perspective:
            perspective_transformer, persp_polygon = setup_perspective_after_main_polygon(
                video_name, first_frame_path, polygons_save_dir,
                main_polygon=main_polygon,
                use_perspective=use_perspective,
                auto_setup_perspective=auto_setup_perspective,
                perspective_method=perspective_method  # <-- ПЕРЕДАЁМ МЕТОД
            )

            if perspective_transformer:
                print("✓ Перспектива настроена и будет применяться")
            elif persp_polygon is not None:
                print("✓ Перспективный полигон найден и сохранён")
            else:
                print("ℹ Перспектива не настроена — анализ будет выполнен без коррекции перспективы")

        # === ШАГ 3: запуск анализа ===
        try:
            summary = analyze_video_shikhta(
                frames_dir=frames_dir,
                output_dir=shikhta_output,
                polygon=main_polygon,
                save_visualizations=True,
                save_every_n=save_every_n,
                use_parallel=use_parallel,
                max_workers=max_workers,
                use_perspective=(perspective_transformer is not None),
                perspective_config=(
                    perspective_transformer if perspective_transformer is not None else persp_polygon),
                perspective_method=perspective_method
            )

            if summary:
                results.append((video_name, shikhta_output, metrics_file))
                # Копия сводки в общую папку
                try:
                    summary_copy = os.path.join(
                        RESULTS_SHIKHTA_DIR, f"{video_name}_summary.json")
                    os.makedirs(os.path.dirname(summary_copy), exist_ok=True)
                    with open(summary_copy, 'w', encoding='utf-8') as f:
                        json.dump(summary, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print("  ⚠ Не удалось сохранить краткую сводку:", e)

        except Exception as e:
            print(f"✗ Ошибка при анализе {video_name}: {e}")
            import traceback
            traceback.print_exc()

    # Печать итогов
    print_summary(results)
    return results


def print_summary(results):
    print("\n" + "="*60)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("="*60)
    if not results:
        print("Нет результатов анализа")
        return

    all_left_means = []
    all_right_means = []
    perspective_count = 0
    per_video = []

    for video_name, output_dir, metrics_file in results:
        if not os.path.exists(metrics_file):
            print(f"  ⚠ Файл метрик не найден: {metrics_file}")
            continue
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            summary = data.get('summary', {})
            left = summary.get('left', {})
            right = summary.get('right', {})
            pused = summary.get('perspective_corrected', False)
            if pused:
                perspective_count += 1
            lm = left.get('mean', 0)
            rm = right.get('mean', 0)
            all_left_means.append(lm)
            all_right_means.append(rm)
            per_video.append({
                'video': video_name,
                'frames': summary.get('total_frames', 0),
                'left_mean': lm,
                'right_mean': rm,
                'perspective': bool(pused)
            })
            print(
                f"{video_name}: frames={summary.get('total_frames', 0)}; left={lm:.2f}%; right={rm:.2f}%; persp={pused}")
        except Exception as e:
            print("  ✗ Ошибка чтения метрик:", e)

    agg_left = float(np.mean(all_left_means)) if all_left_means else 0.0
    agg_right = float(np.mean(all_right_means)) if all_right_means else 0.0

    print("\n--- Aggregated ---")
    print(f"Videos: {len(per_video)}, perspective used: {perspective_count}")
    print(f"Avg left mean: {agg_left:.2f}%, Avg right mean: {agg_right:.2f}%")

    aggregate_path = os.path.join(
        RESULTS_SHIKHTA_DIR, "aggregate_summary.json")
    try:
        os.makedirs(os.path.dirname(aggregate_path), exist_ok=True)
        with open(aggregate_path, 'w', encoding='utf-8') as f:
            json.dump({
                'generated_at': datetime.now().isoformat(),
                'videos': per_video,
                'avg_left': agg_left,
                'avg_right': agg_right,
                'perspective_count': perspective_count
            }, f, indent=2, ensure_ascii=False)
        print("✓ Агрегированная сводка сохранена:", aggregate_path)
    except Exception as e:
        print("⚠ Не удалось сохранить агрегированную сводку:", e)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="run_shikhta_only — анализ шихты с GUI полигона")
    p.add_argument("--force", action="store_true",
                   help="принудительно перезапустить анализ")
    p.add_argument("--no-perspective", action="store_true",
                   help="выключить попытки настройки перспективы/GUI для неё")
    p.add_argument("--no-parallel", action="store_true",
                   help="не использовать многопоточность")
    p.add_argument("--workers", "-w", type=int,
                   default=8, help="количество потоков")
    p.add_argument("--save-every", "-s", type=int, default=20,
                   help="сохранять визуализацию каждого N кадра")
    p.add_argument("--no-auto-persp", action="store_true",
                   help="не выполнять авто-настройку перспективы (если модуль доступен)")
    p.add_argument("--persp-method",
                   choices=['standard', 'hexagon'],
                   default='hexagon',
                   help="метод перспективной коррекции: standard (4 точки) или hexagon (6 точек)")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    setup_directories()
    analyze_all_videos(
        force_reanalyze=args.force,
        save_every_n=args.save_every,
        use_parallel=(not args.no_parallel),
        max_workers=max(1, args.workers),
        use_perspective=(not args.no_perspective),
        auto_setup_perspective=(not args.no_auto_persp),
        perspective_method=args.persp_method
    )
    print("\nГотово.")
