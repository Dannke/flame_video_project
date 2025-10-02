"""
Высокоуровневый процессор для анализа шихты
Предоставляет удобный API для батчевой обработки и экспорта данных
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt

try:
    from shikhta_analysis import ShikhtaAnalyzer, analyze_video_shikhta
except ImportError:
    from src.shikhta_analysis import ShikhtaAnalyzer, analyze_video_shikhta


class ShikhtaProcessor:
    """
    Высокоуровневый процессор для пакетного анализа шихты
    и экспорта результатов в различные форматы
    """
    
    def __init__(self, frames_root='data/frames', output_root='data/shikhta_results'):
        self.frames_root = Path(frames_root)
        self.output_root = Path(output_root)
        self.results = {}
        
    def find_videos(self) -> List[Tuple[str, Path, int]]:
        """Найти все директории с кадрами"""
        videos = []
        
        if not self.frames_root.exists():
            print(f"Директория {self.frames_root} не существует")
            return videos
        
        for item in self.frames_root.iterdir():
            if item.is_dir():
                frames = list(item.glob("*.jpg"))
                if frames:
                    videos.append((item.name, item, len(frames)))
        
        return sorted(videos)
    
    def process_single_video(self, video_name: str, frames_dir: Path, 
                            polygon=None, save_every_n=20, 
                            force_reprocess=False) -> Optional[Dict]:
        """Обработка одного видео"""
        output_dir = self.output_root / video_name
        metrics_file = output_dir / f"{video_name}_metrics.json"
        
        # Проверка существующих результатов
        if metrics_file.exists() and not force_reprocess:
            print(f"✓ {video_name}: уже обработано (загружаем из кэша)")
            with open(metrics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.results[video_name] = data
            return data
        
        print(f"\n{'='*60}")
        print(f"Обработка: {video_name}")
        print(f"{'='*60}")
        
        try:
            summary = analyze_video_shikhta(
                frames_dir=str(frames_dir),
                output_dir=str(output_dir),
                polygon=polygon,
                save_visualizations=True,
                save_every_n=save_every_n
            )
            
            # Загружаем полные данные
            with open(metrics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.results[video_name] = data
            return data
            
        except Exception as e:
            print(f"✗ Ошибка при обработке {video_name}: {e}")
            return None
    
    def process_all_videos(self, polygon=None, save_every_n=20, 
                          force_reprocess=False) -> Dict[str, Dict]:
        """Обработка всех найденных видео"""
        videos = self.find_videos()
        
        if not videos:
            print("Не найдено видео для обработки")
            return {}
        
        print(f"Найдено {len(videos)} видео для обработки\n")
        
        for video_name, frames_dir, frames_count in videos:
            print(f"{video_name}: {frames_count} кадров")
            self.process_single_video(
                video_name, frames_dir, 
                polygon=polygon,
                save_every_n=save_every_n,
                force_reprocess=force_reprocess
            )
        
        return self.results
    
    def get_summary(self, video_name: str) -> Optional[Dict]:
        """Получить сводную статистику для видео"""
        if video_name not in self.results:
            # Попытка загрузить из файла
            metrics_file = self.output_root / video_name / f"{video_name}_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.results[video_name] = data
            else:
                return None
        
        return self.results[video_name].get('summary')
    
    def get_frame_data(self, video_name: str) -> Optional[pd.DataFrame]:
        """Получить покадровые данные в виде DataFrame"""
        if video_name not in self.results:
            summary = self.get_summary(video_name)
            if not summary:
                return None
        
        frames_data = self.results[video_name].get('frames', [])
        if not frames_data:
            return None
        
        df = pd.DataFrame(frames_data)
        return df
    
    def export_to_csv(self, video_name: str, output_path: str) -> bool:
        """Экспорт покадровых данных в CSV"""
        df = self.get_frame_data(video_name)
        
        if df is None:
            print(f"Нет данных для экспорта: {video_name}")
            return False
        
        try:
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"✓ Экспортировано в {output_path}")
            return True
        except Exception as e:
            print(f"✗ Ошибка экспорта: {e}")
            return False
    
    def export_all_to_csv(self, output_dir='results/csv_exports'):
        """Экспорт всех видео в отдельные CSV файлы"""
        os.makedirs(output_dir, exist_ok=True)
        
        for video_name in self.results.keys():
            output_path = os.path.join(output_dir, f"{video_name}.csv")
            self.export_to_csv(video_name, output_path)
    
    def export_combined_csv(self, output_path='results/all_videos_combined.csv'):
        """Экспорт всех видео в один CSV с колонкой video_name"""
        dfs = []
        
        for video_name in self.results.keys():
            df = self.get_frame_data(video_name)
            if df is not None:
                df['video_name'] = video_name
                dfs.append(df)
        
        if not dfs:
            print("Нет данных для экспорта")
            return False
        
        combined = pd.concat(dfs, ignore_index=True)
        
        try:
            combined.to_csv(output_path, index=False, encoding='utf-8')
            print(f"✓ Объединенный CSV сохранен: {output_path}")
            print(f"  Всего строк: {len(combined)}")
            print(f"  Видео: {combined['video_name'].nunique()}")
            return True
        except Exception as e:
            print(f"✗ Ошибка экспорта: {e}")
            return False
    
    def plot_video_timeline(self, video_name: str, save_path: Optional[str] = None):
        """Построить график изменения распределения шихты во времени"""
        df = self.get_frame_data(video_name)
        
        if df is None:
            print(f"Нет данных для графика: {video_name}")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # График 1: Процентное соотношение лево/право
        axes[0].plot(df['frame_idx'], df['left_percent'], 
                    label='Левая часть', color='blue', alpha=0.7)
        axes[0].plot(df['frame_idx'], df['right_percent'], 
                    label='Правая часть', color='red', alpha=0.7)
        axes[0].axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_ylabel('Процент, %')
        axes[0].set_title(f'Распределение шихты: {video_name}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # График 2: Общая площадь шихты
        axes[1].plot(df['frame_idx'], df['total_area'], 
                    color='green', alpha=0.7)
        axes[1].set_ylabel('Площадь, пикс²')
        axes[1].set_title('Общая площадь шихты')
        axes[1].grid(True, alpha=0.3)
        
        # График 3: Количество контуров
        axes[2].plot(df['frame_idx'], df['left_contours_count'], 
                    label='Левые контуры', color='blue', alpha=0.7)
        axes[2].plot(df['frame_idx'], df['right_contours_count'], 
                    label='Правые контуры', color='red', alpha=0.7)
        axes[2].set_xlabel('Номер кадра')
        axes[2].set_ylabel('Количество контуров')
        axes[2].set_title('Фрагментация шихты')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ График сохранен: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_all_videos_comparison(self, save_path='results/comparison.png'):
        """Сравнительный график для всех видео"""
        if not self.results:
            print("Нет данных для сравнения")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))
        
        for idx, (video_name, color) in enumerate(zip(self.results.keys(), colors)):
            summary = self.get_summary(video_name)
            if not summary:
                continue
            
            left_mean = summary['left']['mean']
            right_mean = summary['right']['mean']
            
            # График средних значений
            axes[0].bar(idx * 2, left_mean, color=color, alpha=0.7, label=f'{video_name} (L)')
            axes[0].bar(idx * 2 + 1, right_mean, color=color, alpha=0.4, label=f'{video_name} (R)')
        
        axes[0].axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_ylabel('Средний процент, %')
        axes[0].set_title('Среднее распределение шихты по видео')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Boxplot распределения
        all_left_data = []
        all_right_data = []
        labels = []
        
        for video_name in self.results.keys():
            df = self.get_frame_data(video_name)
            if df is not None:
                all_left_data.append(df['left_percent'].values)
                all_right_data.append(df['right_percent'].values)
                labels.append(video_name)
        
        if all_left_data:
            positions = np.arange(len(labels)) * 2
            bp1 = axes[1].boxplot(all_left_data, positions=positions, widths=0.6,
                                  patch_artist=True, label='Левая часть')
            bp2 = axes[1].boxplot(all_right_data, positions=positions + 0.7, widths=0.6,
                                  patch_artist=True, label='Правая часть')
            
            for patch in bp1['boxes']:
                patch.set_facecolor('blue')
                patch.set_alpha(0.5)
            for patch in bp2['boxes']:
                patch.set_facecolor('red')
                patch.set_alpha(0.5)
            
            axes[1].set_xticks(positions + 0.35)
            axes[1].set_xticklabels(labels, rotation=45, ha='right')
            axes[1].set_ylabel('Процент, %')
            axes[1].set_title('Распределение значений (boxplot)')
            axes[1].axhline(y=50, color='gray', linestyle='--', alpha=0.5)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Сравнительный график сохранен: {save_path}")
        plt.close()
    
    def generate_report(self, output_path='results/shikhta_report.txt'):
        """Генерация текстового отчета"""
        if not self.results:
            print("Нет данных для отчета")
            return
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("ОТЧЕТ ПО АНАЛИЗУ ШИХТЫ\n")
            f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Всего проанализировано видео: {len(self.results)}\n\n")
            
            for video_name in sorted(self.results.keys()):
                summary = self.get_summary(video_name)
                if not summary:
                    continue
                
                f.write("-"*70 + "\n")
                f.write(f"ВИДЕО: {video_name}\n")
                f.write("-"*70 + "\n")
                f.write(f"Кадров: {summary['total_frames']}\n\n")
                
                left = summary['left']
                right = summary['right']
                
                f.write("ЛЕВАЯ ЧАСТЬ:\n")
                f.write(f"  Среднее:  {left['mean']:6.2f}%\n")
                f.write(f"  Медиана:  {left['median']:6.2f}%\n")
                f.write(f"  Мин/Макс: {left['min']:6.2f}% / {left['max']:6.2f}%\n")
                f.write(f"  Ст.откл:  {left['std']:6.2f}%\n\n")
                
                f.write("ПРАВАЯ ЧАСТЬ:\n")
                f.write(f"  Среднее:  {right['mean']:6.2f}%\n")
                f.write(f"  Медиана:  {right['median']:6.2f}%\n")
                f.write(f"  Мин/Макс: {right['min']:6.2f}% / {right['max']:6.2f}%\n")
                f.write(f"  Ст.откл:  {right['std']:6.2f}%\n\n")
                
                # Анализ баланса
                balance = abs(left['mean'] - 50)
                if balance < 5:
                    status = "ОТЛИЧНО (сбалансировано)"
                elif balance < 10:
                    status = "ХОРОШО"
                elif balance < 15:
                    status = "УДОВЛЕТВОРИТЕЛЬНО"
                else:
                    status = "ТРЕБУЕТ ВНИМАНИЯ (несбалансировано)"
                
                f.write(f"БАЛАНС: {status}\n")
                f.write(f"  Отклонение от 50/50: {balance:.2f}%\n\n")
            
            # Агрегированная статистика
            f.write("="*70 + "\n")
            f.write("ОБЩАЯ СТАТИСТИКА ПО ВСЕМ ВИДЕО\n")
            f.write("="*70 + "\n")
            
            all_left_means = [self.get_summary(v)['left']['mean'] 
                             for v in self.results.keys() 
                             if self.get_summary(v)]
            all_right_means = [self.get_summary(v)['right']['mean'] 
                              for v in self.results.keys() 
                              if self.get_summary(v)]
            
            if all_left_means:
                f.write(f"Средняя левая часть:  {np.mean(all_left_means):.2f}% ")
                f.write(f"(σ={np.std(all_left_means):.2f}%)\n")
                f.write(f"Средняя правая часть: {np.mean(all_right_means):.2f}% ")
                f.write(f"(σ={np.std(all_right_means):.2f}%)\n")
        
        print(f"✓ Отчет сохранен: {output_path}")


# Пример использования
if __name__ == "__main__":
    processor = ShikhtaProcessor()
    
    # Обработка всех видео
    processor.process_all_videos(save_every_n=20, force_reprocess=False)
    
    # Экспорт в CSV
    processor.export_combined_csv('results/all_shikhta_data.csv')
    
    # Генерация графиков
    for video_name in processor.results.keys():
        processor.plot_video_timeline(
            video_name, 
            save_path=f'results/timeline_{video_name}.png'
        )
    
    processor.plot_all_videos_comparison()
    
    # Текстовый отчет
    processor.generate_report()
    
    print("\n✓ Все задачи выполнены!")