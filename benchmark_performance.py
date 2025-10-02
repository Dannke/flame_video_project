"""
Сравнение производительности последовательной и параллельной обработки
"""
import os
import sys
import time
from pathlib import Path

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

try:
    from src.shikhta_analysis import ShikhtaAnalyzer
except ImportError:
    print("Модуль shikhta_analysis.py не найден")
    sys.exit(1)

def benchmark_processing(frames_dir, num_frames=100):
    """Тест производительности"""
    
    frames_dir = Path(frames_dir)
    frame_files = sorted(frames_dir.glob("*.jpg"))[:num_frames]
    
    if not frame_files:
        print(f"Нет кадров в {frames_dir}")
        return
    
    print("="*70)
    print(f"ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("="*70)
    print(f"Видео: {frames_dir.name}")
    print(f"Кадров для теста: {len(frame_files)}")
    print()
    
    # Тест 1: Последовательная обработка
    print("[1/3] Последовательная обработка...")
    analyzer_seq = ShikhtaAnalyzer()
    start = time.time()
    analyzer_seq.process_video_frames(
        frames_dir=frames_dir,
        output_dir=None,
        save_every_n=999999,  # не сохраняем визуализации
        max_frames=num_frames,
        use_parallel=False
    )
    time_seq = time.time() - start
    fps_seq = len(frame_files) / time_seq
    
    print(f"✓ Завершено за {time_seq:.2f} сек ({fps_seq:.2f} кадров/сек)\n")
    
    # Тест 2: Параллельная обработка (4 потока)
    print("[2/3] Параллельная обработка (4 потока)...")
    analyzer_par4 = ShikhtaAnalyzer()
    start = time.time()
    analyzer_par4.process_video_frames(
        frames_dir=frames_dir,
        output_dir=None,
        save_every_n=999999,
        max_frames=num_frames,
        use_parallel=True,
        max_workers=4
    )
    time_par4 = time.time() - start
    fps_par4 = len(frame_files) / time_par4
    
    print(f"✓ Завершено за {time_par4:.2f} сек ({fps_par4:.2f} кадров/сек)\n")
    
    # Тест 3: Параллельная обработка (8 потоков)
    print("[3/3] Параллельная обработка (8 потоков)...")
    analyzer_par8 = ShikhtaAnalyzer()
    start = time.time()
    analyzer_par8.process_video_frames(
        frames_dir=frames_dir,
        output_dir=None,
        save_every_n=999999,
        max_frames=num_frames,
        use_parallel=True,
        max_workers=8
    )
    time_par8 = time.time() - start
    fps_par8 = len(frame_files) / time_par8
    
    print(f"✓ Завершено за {time_par8:.2f} сек ({fps_par8:.2f} кадров/сек)\n")
    
    # Результаты
    print("="*70)
    print("РЕЗУЛЬТАТЫ ТЕСТА")
    print("="*70)
    print(f"{'Режим':<30} {'Время, сек':<15} {'Кадров/сек':<15} {'Ускорение'}")
    print("-"*70)
    print(f"{'Последовательный':<30} {time_seq:<15.2f} {fps_seq:<15.2f} {'1.0x'}")
    print(f"{'Параллельный (4 потока)':<30} {time_par4:<15.2f} {fps_par4:<15.2f} {time_seq/time_par4:.1f}x")
    print(f"{'Параллельный (8 потоков)':<30} {time_par8:<15.2f} {fps_par8:<15.2f} {time_seq/time_par8:.1f}x")
    print("="*70)
    
    # Рекомендация
    best_time = min(time_seq, time_par4, time_par8)
    if best_time == time_par8:
        print("\n💡 Рекомендация: используйте max_workers=8 для максимальной скорости")
    elif best_time == time_par4:
        print("\n💡 Рекомендация: используйте max_workers=4 для оптимального баланса")
    else:
        print("\n💡 Рекомендация: параллелизация не дает преимущества на вашей системе")
    
    # Прогноз для полного видео
    print(f"\nПрогноз для полного видео ({frames_dir.name}):")
    total_frames = len(list(frames_dir.glob("*.jpg")))
    if total_frames > num_frames:
        est_time_seq = (total_frames / num_frames) * time_seq
        est_time_best = (total_frames / num_frames) * best_time
        print(f"  Последовательно: ~{est_time_seq/60:.1f} минут")
        print(f"  Оптимально:      ~{est_time_best/60:.1f} минут")
        print(f"  Экономия:        ~{(est_time_seq - est_time_best)/60:.1f} минут")

def main():
    # Автоматический поиск первого видео
    frames_root = Path("data/frames")
    
    if not frames_root.exists():
        print("Директория data/frames не найдена")
        return
    
    video_dirs = [d for d in frames_root.iterdir() if d.is_dir()]
    
    if not video_dirs:
        print("Нет видео в data/frames/")
        print("Сначала запустите: python run.py")
        return
    
    # Выбор видео для теста
    test_video = video_dirs[0]
    print(f"Тестирование на видео: {test_video.name}\n")
    
    # Запуск теста на 100 кадрах
    benchmark_processing(test_video, num_frames=100)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрервано пользователем")
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()