from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Подготовка окружения
# ---------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Поддержка src/ если есть
src_dir = os.path.join(ROOT, "src")
if os.path.isdir(src_dir) and src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Импорты модулей проекта
# ---------------------------------------------------------------------------
try:
    from shikhta_analysis import analyze_video_shikhta
except Exception as exc:  # pragma: no cover - требуется в рантайме
    logger.error(
        "Не найден модуль анализа шихты (shikhta_analysis.py): %s", exc)
    raise

try:
    from perspective_transform_hexagon import (
        HexagonPerspectiveTransformer,
        setup_hexagon_perspective_gui,
    )
    HEXAGON_PERSPECTIVE_AVAILABLE: bool = True
except Exception:
    HEXAGON_PERSPECTIVE_AVAILABLE = False
    HexagonPerspectiveTransformer = None  # type: ignore
    setup_hexagon_perspective_gui = None  # type: ignore
    logger.info(
        "Модуль perspective_transform_hexagon.py не найден — перспективная коррекция недоступна")

# GUI для рисования полигона (опционально)
try:
    from polygon_gui import prompt_polygon_on_image, load_saved_polygon
    POLYGON_GUI_AVAILABLE = True
except Exception:
    POLYGON_GUI_AVAILABLE = False
    prompt_polygon_on_image = None  # type: ignore
    load_saved_polygon = None  # type: ignore
    logger.info(
        "Модуль polygon_gui не найден — выбор полигона через GUI недоступен")

# ---------------------------------------------------------------------------
# Константы — пути
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(ROOT, "data")
FRAMES_ROOT = os.path.join(DATA_DIR, "frames")
SHIKHTA_RESULTS_ROOT = os.path.join(DATA_DIR, "shikhta_results")
RESULTS_SHIKHTA_DIR = os.path.join(ROOT, "results", "shikhta_metrics")

# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------


def setup_directories() -> None:
    """Создать необходимые директории, если их нет."""
    os.makedirs(SHIKHTA_RESULTS_ROOT, exist_ok=True)
    os.makedirs(RESULTS_SHIKHTA_DIR, exist_ok=True)


def find_frame_directories() -> List[Tuple[str, str, int]]:
    """Найти поддиректории с кадрами (jpg/png и т.д.).

    Возвращает список кортежей (имя_видео, путь, число_кадров).
    """
    if not os.path.exists(FRAMES_ROOT):
        logger.warning("Директория с кадрами не найдена: %s", FRAMES_ROOT)
        return []

    frame_dirs: List[Tuple[str, str, int]] = []
    for item in sorted(os.listdir(FRAMES_ROOT)):
        full_path = os.path.join(FRAMES_ROOT, item)
        if os.path.isdir(full_path):
            jpg_files = list(Path(full_path).glob("*.jpg"))
            if jpg_files:
                frame_dirs.append((item, full_path, len(jpg_files)))
    return frame_dirs


def _save_polygon_to_file(polygon: np.ndarray, save_path: str) -> bool:
    """Сохранить полигон в JSON формате — список [ [x,y], ... ].

    Возвращает True при успешном сохранении.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"polygon": [list(map(int, p))
                      for p in polygon]}, f, indent=2, ensure_ascii=False)
        return True
    except Exception as exc:
        logger.exception("Ошибка при сохранении полигона: %s", exc)
        return False


def _load_polygon_from_file(load_path: str) -> Optional[np.ndarray]:
    """Загрузить полигон из файла. Возвращает np.ndarray или None."""
    try:
        if not os.path.exists(load_path):
            return None
        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        poly = data.get("polygon")
        if poly:
            return np.array(poly, dtype=np.int32)
    except Exception:
        logger.exception("Ошибка при загрузке полигона из %s", load_path)
    return None


# ---------------------------------------------------------------------------
# GUI / подготовка полигона
# ---------------------------------------------------------------------------

def prepare_main_polygon_via_gui(first_frame_path: str, polygons_save_dir: str, video_name: str) -> Optional[np.ndarray]:
    """Шаг 1: загрузка/выбор основного полигона.

    Алгоритм:
      1) пробуем загрузить через polygon_gui.load_saved_polygon (если есть)
      2) пробуем загрузить JSON файл {video_name}_polygon.json
      3) если есть GUI — предлагаем пользователю нарисовать полигон

    Возвращает основный полигон (np.ndarray) или None.
    """
    main_polygon_path = os.path.join(
        polygons_save_dir, f"{video_name}_polygon.json")

    # 1) Попытка получить из polygon_gui
    main_polygon = None
    if POLYGON_GUI_AVAILABLE and load_saved_polygon:
        try:
            loaded = load_saved_polygon(polygons_save_dir, video_name)
            if loaded is not None:
                main_polygon = np.array(loaded, dtype=np.int32)
                logger.info(
                    "Загружен полигон через polygon_gui для %s", video_name)
        except Exception:
            logger.exception("load_saved_polygon вернул исключение")

    # 2) Пытаться загрузить из файла
    if main_polygon is None:
        main_polygon = _load_polygon_from_file(main_polygon_path)
        if main_polygon is not None:
            logger.info("Основной полигон загружен из %s", main_polygon_path)

    # 3) Рисуем через GUI если всё ещё нет
    if main_polygon is None and POLYGON_GUI_AVAILABLE and prompt_polygon_on_image:
        logger.info(
            "Откроется GUI для выбора основного полигона (шестиугольник)")
        img = cv2.imread(first_frame_path)
        try:
            polygon_raw = prompt_polygon_on_image(
                img, default_polygon=None, video_name=video_name, save_dir=polygons_save_dir)
        except Exception:
            logger.exception("prompt_polygon_on_image вернула исключение")
            polygon_raw = None

        if polygon_raw:
            main_polygon = np.array(polygon_raw, dtype=np.int32)
            if _save_polygon_to_file(main_polygon, main_polygon_path):
                logger.info("Основной полигон сохранён: %s", main_polygon_path)

    return main_polygon


# ---------------------------------------------------------------------------
# Перспектива — ТОЛЬКО hexagon (6 точек)
# ---------------------------------------------------------------------------

def setup_perspective_after_main_polygon(video_name: str,
                                         first_frame_path: str,
                                         polygons_save_dir: str,
                                         main_polygon: Optional[np.ndarray] = None,
                                         use_perspective: bool = True,
                                         auto_setup_perspective: bool = True) -> Tuple[Optional[object], Optional[np.ndarray]]:
    """Настроить 6-точечную перспективу (если разрешено и доступен модуль).

    Возвращает кортеж (perspective_transformer | None, persp_polygon | None).
    """
    if not use_perspective:
        return None, None

    if not HEXAGON_PERSPECTIVE_AVAILABLE:
        logger.warning("Hexagon perspective unavailable — skipping perspective setup")
        return None, None

    hexagon_config_path = os.path.join(polygons_save_dir, f"{video_name}_hexagon_perspective.json")

    # === ИНТЕГРАЦИЯ: Автоматическая генерация с проверкой существующего ===
    if auto_setup_perspective and main_polygon is not None:
        logger.info("Запуск генерации hexagon полигона (с проверкой существующего)")
        
        img = cv2.imread(first_frame_path)
        if img is None:
            logger.error("Не удалось загрузить кадр: %s", first_frame_path)
            return None, None
        
        try:
            # Импортируем модуль автоматической генерации
            from auto_hexagon_perspective import integrate_with_existing_gui
            
            # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Передаём изображение С base_polygon
            # Сначала рисуем base_polygon на изображении для контекста
            img_with_base = img.copy()
            try:
                cv2.polylines(img_with_base, [main_polygon], True, (0, 255, 0), 2)
                cv2.putText(
                    img_with_base, 
                    "Base polygon (furnace zone)", 
                    (10, img.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 255, 0), 
                    2
                )
            except Exception as e:
                logger.warning("Не удалось нарисовать base_polygon: %s", e)
                img_with_base = img
            
            # Запускаем с предпросмотром (включая проверку существующего конфига)
            hexagon_points = integrate_with_existing_gui(
                image=img_with_base,  # ПЕРЕДАЁМ ИЗОБРАЖЕНИЕ С БАЗОВЫМ ПОЛИГОНОМ
                base_polygon=main_polygon,  # И САМ ПОЛИГОН ДЛЯ ФУНКЦИЙ
                video_name=video_name,
                save_dir=polygons_save_dir,
                existing_config_path=hexagon_config_path
            )
            
            if hexagon_points:
                # Пользователь принял полигон (существующий или новый)
                logger.info("✓ Hexagon полигон принят пользователем")
                
                transformer = HexagonPerspectiveTransformer(
                    hexagon_points, dst_width=1920, dst_height=1080)
                os.makedirs(polygons_save_dir, exist_ok=True)
                transformer.save_config(hexagon_config_path)
                logger.info("6-точечная конфигурация сохранена: %s", hexagon_config_path)
                return transformer, None
            
            else:
                # Пользователь отменил (ESC)
                logger.warning("Генерация hexagon отменена пользователем")
                return None, None
        
        except ImportError:
            logger.warning("Модуль auto_hexagon_perspective не найден, используется стандартный GUI")
            # Откатываемся на стандартный ручной GUI
            pass
        except Exception:
            logger.exception("Ошибка при автоматической генерации hexagon полигона")
            # Откатываемся на стандартный ручной GUI
            pass

    # Запустить GUI для настройки, если разрешено (стандартный путь - fallback)
    if auto_setup_perspective:
        logger.info("Открывается GUI для настройки 6-точечной перспективы")
        img = cv2.imread(first_frame_path)
        if img is None:
            logger.error("Не удалось загрузить кадр: %s", first_frame_path)
            return None, None

        # Наложим основной полигон для ориентира
        vis = img.copy()
        if main_polygon is not None:
            try:
                cv2.polylines(vis, [main_polygon], True, (0, 255, 0), 2)
                cv2.putText(vis, "Main polygon (reference)", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            except Exception:
                logger.exception("Не удалось отобразить основной полигон для GUI")

        try:
            points = setup_hexagon_perspective_gui(vis)
            if not points:
                logger.warning("Настройка 6-точечной перспективы отменена пользователем")
                return None, None

            transformer = HexagonPerspectiveTransformer(
                points, dst_width=1920, dst_height=1080)
            os.makedirs(polygons_save_dir, exist_ok=True)
            transformer.save_config(hexagon_config_path)
            logger.info("6-точечная конфигурация сохранена: %s", hexagon_config_path)
            return transformer, None

        except Exception:
            logger.exception("Ошибка при настройке 6-точечной перспективы")
            return None, None

    logger.info("Авто-настройка перспективы отключена")
    return None, None


# ---------------------------------------------------------------------------
# Основная логика анализа
# ---------------------------------------------------------------------------

def analyze_all_videos(force_reanalyze: bool = False,
                       save_every_n: int = 20,
                       use_parallel: bool = True,
                       max_workers: int = 8,
                       use_perspective: bool = True,
                       auto_setup_perspective: bool = True,
                       min_contour_area: int = 100,
                       near_zone_ratio: float = 0.5,
                       near_zone_c_offset: int = -5,
                       far_zone_c_offset: int = 5,
                       near_zone_area_multiplier: float = 3.0,
                       use_adaptive_flame_detection: bool = True,
                       far_c_boost_no_flame: int = 5,
                       flame_detection_threshold: float = 10.0,
                       perspective_profile: str = 'perspective') -> List[Tuple[str, str, str]]:
    """Проход по всем директориям с кадрами и запуск анализа для каждого видео.

    Возвращает список кортежей (video_name, output_dir, metrics_file)
    """
    frame_dirs = find_frame_directories()
    if not frame_dirs:
        logger.warning("Не найдено директорий с кадрами в %s", FRAMES_ROOT)
        return []

    logger.info("Найдено %d видео для анализа", len(frame_dirs))

    # Проверяем доступность hexagon-перспективы
    if use_perspective and HEXAGON_PERSPECTIVE_AVAILABLE:
        logger.info(
            "6-точечная перспектива доступна и будет использована при запросе")
    elif use_perspective:
        logger.warning(
            "Перспектива запрошена, но модуль hexagon-perspective недоступен")

    results: List[Tuple[str, str, str]] = []

    for video_name, frames_dir, frames_count in frame_dirs:
        shikhta_output = os.path.join(SHIKHTA_RESULTS_ROOT, video_name)
        metrics_file = os.path.join(
            shikhta_output, f"{video_name}_metrics.json")
        polygons_save_dir = os.path.join(shikhta_output, "polygons")

        if os.path.exists(metrics_file) and not force_reanalyze:
            logger.info(
                "Анализ уже есть для %s — пропускаю (use --force для повтора)", video_name)
            results.append((video_name, shikhta_output, metrics_file))
            continue

        logger.info("\nProcessing video: %s — frames: %d",
                    video_name, frames_count)

        # Первый кадр
        imgs = sorted([os.path.join(frames_dir, p) for p in os.listdir(frames_dir)
                       if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
        first_frame_path = imgs[0] if imgs else None
        if first_frame_path is None:
            logger.warning("Нет кадров для %s — пропускаю", video_name)
            continue

        os.makedirs(polygons_save_dir, exist_ok=True)

        # Шаг 1: основной полигон
        main_polygon = prepare_main_polygon_via_gui(
            first_frame_path, polygons_save_dir, video_name)
        if main_polygon is None:
            logger.info(
                "Основной полигон не задан — будет использован дефолт в анализаторе (если есть)")

        # Шаг 2: перспектива (после выбора основного полигона)
        perspective_transformer = None
        persp_polygon = None
        if use_perspective:
            perspective_transformer, persp_polygon = setup_perspective_after_main_polygon(
                video_name, first_frame_path, polygons_save_dir, main_polygon=main_polygon,
                use_perspective=use_perspective, auto_setup_perspective=auto_setup_perspective)

            if perspective_transformer:
                logger.info("Перспектива настроена и будет применяться")
            else:
                logger.info(
                    "Перспектива не настроена — анализ будет выполнен без коррекции перспективы")

        # Шаг 3: собственно запуск анализа
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
                perspective_method="hexagon",
                min_contour_area=min_contour_area,
                near_zone_ratio=near_zone_ratio,
                near_zone_c_offset=near_zone_c_offset,
                far_zone_c_offset=far_zone_c_offset,
                near_zone_area_multiplier=near_zone_area_multiplier,
                use_adaptive_flame_detection=use_adaptive_flame_detection,
                far_c_boost_no_flame=far_c_boost_no_flame,
                flame_detection_threshold=flame_detection_threshold,
            )

            if summary:
                results.append((video_name, shikhta_output, metrics_file))
                # сохраняем краткую сводку
                try:
                    summary_copy = os.path.join(
                        RESULTS_SHIKHTA_DIR, f"{video_name}_summary.json")
                    os.makedirs(os.path.dirname(summary_copy), exist_ok=True)
                    with open(summary_copy, "w", encoding="utf-8") as f:
                        json.dump(summary, f, indent=2, ensure_ascii=False)
                except Exception:
                    logger.exception(
                        "Не удалось сохранить краткую сводку для %s", video_name)

        except Exception:
            logger.exception("Ошибка при анализе %s", video_name)

    print_summary(results)
    return results


# ---------------------------------------------------------------------------
# Печать итоговой сводки
# ---------------------------------------------------------------------------

def print_summary(results: List[Tuple[str, str, str]]) -> None:
    logger.info("\n=== ИТОГОВАЯ СТАТИСТИКА ===")
    if not results:
        logger.info("Нет результатов анализа")
        return

    all_left_means: List[float] = []
    all_right_means: List[float] = []
    perspective_count = 0
    per_video = []

    for video_name, output_dir, metrics_file in results:
        if not os.path.exists(metrics_file):
            logger.warning("Файл метрик не найден: %s", metrics_file)
            continue
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            summary = data.get("summary", {})
            left = summary.get("left", {})
            right = summary.get("right", {})
            pused = summary.get("perspective_corrected", False)
            if pused:
                perspective_count += 1
            lm = left.get("mean", 0)
            rm = right.get("mean", 0)
            all_left_means.append(lm)
            all_right_means.append(rm)
            per_video.append({
                "video": video_name,
                "frames": summary.get("total_frames", 0),
                "left_mean": lm,
                "right_mean": rm,
                "perspective": bool(pused),
            })
            logger.info("%s: frames=%d; left=%.2f%%; right=%.2f%%; persp=%s",
                        video_name, summary.get("total_frames", 0), lm, rm, pused)
        except Exception:
            logger.exception("Ошибка чтения метрик из %s", metrics_file)

    agg_left = float(np.mean(all_left_means)) if all_left_means else 0.0
    agg_right = float(np.mean(all_right_means)) if all_right_means else 0.0

    logger.info("--- Aggregated ---")
    logger.info("Videos: %d, perspective used: %d",
                len(per_video), perspective_count)
    logger.info("Avg left mean: %.2f%%, Avg right mean: %.2f%%",
                agg_left, agg_right)

    aggregate_path = os.path.join(
        RESULTS_SHIKHTA_DIR, "aggregate_summary.json")
    try:
        os.makedirs(os.path.dirname(aggregate_path), exist_ok=True)
        with open(aggregate_path, "w", encoding="utf-8") as f:
            json.dump({
                "generated_at": datetime.now().isoformat(),
                "videos": per_video,
                "avg_left": agg_left,
                "avg_right": agg_right,
                "perspective_count": perspective_count,
            }, f, indent=2, ensure_ascii=False)
        logger.info("Агрегированная сводка сохранена: %s", aggregate_path)
    except Exception:
        logger.exception("Не удалось сохранить агрегированную сводку")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="run_shikhta_only — анализ шихты")

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

    # === НОВЫЙ ПАРАМЕТР ===
    p.add_argument("--persp-profile", type=str, default='perspective',
                   choices=['uniform', 'perspective', 'wide', 'conservative'],
                   help="профиль расширения для автоматического hexagon полигона")

    # Параметры детекции
    detection_group = p.add_argument_group("Параметры детекции шихты")
    detection_group.add_argument(
        "--min-area", type=int, default=100, help="минимальная площадь контура (пикс²)")
    detection_group.add_argument(
        "--near-zone", type=float, default=0.5, help="доля ближней зоны (0.0-1.0)")
    detection_group.add_argument(
        "--near-c", type=int, default=-5, help="порог C для adaptiveThreshold в ближней зоне")
    detection_group.add_argument(
        "--far-c", type=int, default=5, help="порог C для adaptiveThreshold в далёкой зоне")
    detection_group.add_argument("--near-multiplier", type=float, default=2.0,
                                 help="множитель для минимальной площади в ближней зоне")

    flame_group = p.add_argument_group("Адаптивная детекция пламени")
    flame_group.add_argument("--no-adaptive-flame", action="store_true",
                             help="отключить адаптивную детекцию пламени")
    flame_group.add_argument("--far-c-boost", type=int, default=6,
                             help="насколько повысить far_c в кадрах без пламени")
    flame_group.add_argument("--flame-threshold", type=float, default=10.0,
                             help="минимальный процент площади пламени для отметки кадра как \"с пламенем\"")

    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    setup_directories()
    
    # === ПЕРЕДАЁМ ПРОФИЛЬ В analyze_all_videos ===
    analyze_all_videos(
        force_reanalyze=args.force,
        save_every_n=args.save_every,
        use_parallel=(not args.no_parallel),
        max_workers=max(1, args.workers),
        use_perspective=(not args.no_perspective),
        auto_setup_perspective=(not args.no_auto_persp),
        min_contour_area=args.min_area,
        near_zone_ratio=args.near_zone,
        near_zone_c_offset=args.near_c,
        far_zone_c_offset=args.far_c,
        near_zone_area_multiplier=args.near_multiplier,
        flame_detection_threshold=args.flame_threshold,
        perspective_profile=args.persp_profile  # НОВЫЙ ПАРАМЕТР
    )
    logger.info("Готово.")

# python run_shikhta_only.py --force --min-area 120  --near-c 9 --far-c 7 --near-multiplier 4.0 --near-zone 0.7
