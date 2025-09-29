import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import random

MODEL_EXPORT_DIR = os.path.join('results', 'models')
os.makedirs(MODEL_EXPORT_DIR, exist_ok=True)

TRAINING_HISTORY_DIR = os.path.join('results', 'training_history')
os.makedirs(TRAINING_HISTORY_DIR, exist_ok=True)


def safe_ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5):
    try:
        import inspect as _inspect
        _sig = _inspect.signature(optim.lr_scheduler.ReduceLROnPlateau)
        if 'verbose' in _sig.parameters:
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, patience=patience, factor=factor, verbose=True)
        else:
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, patience=patience, factor=factor)
    except Exception:
        try:
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, patience=patience, factor=factor)
        except Exception as e:
            print('Warning: could not create ReduceLROnPlateau scheduler:', e)
            return None


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in features:
            self.encoder_blocks.append(ConvBlock(in_channels, feature))
            in_channels = feature
        self.bottleneck = ConvBlock(features[-1], features[-1]*2)
        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        for feature in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(
                feature*2, feature, kernel_size=2, stride=2))
            self.attention_blocks.append(AttentionBlock(
                F_g=feature, F_l=feature, F_int=feature//2))
            self.decoder_blocks.append(ConvBlock(feature*2, feature))
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.dropout = nn.Dropout2d(0.3)

    def forward(self, x):
        skip_connections = []
        for i, encoder in enumerate(self.encoder_blocks):
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        x = self.dropout(x)
        skip_connections = skip_connections[::-1]
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            skip = self.attention_blocks[i](g=x, x=skip_connections[i])
            x = torch.cat((skip, x), dim=1)
            x = self.decoder_blocks[i](x)
        return torch.sigmoid(self.final_conv(x))


class TemporalUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, temporal_frames=3):
        super().__init__()
        self.temporal_frames = temporal_frames
        self.temporal_conv = nn.Conv3d(in_channels, 16, kernel_size=(
            temporal_frames, 3, 3), padding=(0, 1, 1))
        self.unet = ImprovedUNet(in_channels=16, out_channels=out_channels)

    def forward(self, x):
        batch, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = F.relu(self.temporal_conv(x))
        x = x.squeeze(2)
        return self.unet(x)


class FlameDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, img_size=(256, 256), augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.img_size = img_size
        self.augment = augment

        # Проверяем соответствие количества изображений и масок
        if len(self.image_paths) != len(self.mask_paths):
            print(
                f"Warning: Mismatch between images ({len(self.image_paths)}) and masks ({len(self.mask_paths)})")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = cv2.imread(self.image_paths[idx])
            if image is None:
                raise FileNotFoundError(
                    f"Image not found: {self.image_paths[idx]}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(
                    f"Mask not found: {self.mask_paths[idx]}")

            image = cv2.resize(image, self.img_size)
            mask = cv2.resize(mask, self.img_size)

            # Улучшенная аугментация
            if self.augment and np.random.random() > 0.5:
                # Горизонтальное отражение
                if np.random.random() > 0.5:
                    image = cv2.flip(image, 1)
                    mask = cv2.flip(mask, 1)

                # Изменение яркости
                if np.random.random() > 0.5:
                    factor = np.random.uniform(0.8, 1.2)
                    image = np.clip(image * factor, 0, 255).astype(np.uint8)

                # Поворот
                if np.random.random() > 0.5:
                    angle = np.random.uniform(-10, 10)
                    M = cv2.getRotationMatrix2D(
                        (self.img_size[0]//2, self.img_size[1]//2), angle, 1)
                    image = cv2.warpAffine(image, M, self.img_size)
                    mask = cv2.warpAffine(mask, M, self.img_size)

                # Гауссовский шум
                if np.random.random() > 0.7:
                    noise = np.random.normal(
                        0, 5, image.shape).astype(np.uint8)
                    image = np.clip(image.astype(int) + noise,
                                    0, 255).astype(np.uint8)

            image = image.astype(np.float32) / 255.0
            mask = mask.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
            mask = torch.from_numpy(mask).unsqueeze(0)
            return image, mask

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Возвращаем пустое изображение в случае ошибки
            empty_image = torch.zeros((3, self.img_size[1], self.img_size[0]))
            empty_mask = torch.zeros((1, self.img_size[1], self.img_size[0]))
            return empty_image, empty_mask


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / \
            (pred.sum() + target.sum() + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


def calculate_metrics(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    dice = (2 * intersection + 1e-7) / \
        (pred_binary.sum() + target_binary.sum() + 1e-7)
    true_positive = intersection
    false_positive = pred_binary.sum() - intersection
    false_negative = target_binary.sum() - intersection
    precision = (true_positive + 1e-7) / \
        (true_positive + false_positive + 1e-7)
    recall = (true_positive + 1e-7) / (true_positive + false_negative + 1e-7)
    return {'iou': iou.item(), 'dice': dice.item(), 'precision': precision.item(), 'recall': recall.item()}


class FlameSegmentationTrainer:
    def __init__(self, model, device='cuda'):
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA недоступна! Принудительно переключаемся на CPU")
            device = 'cpu'
        elif device == 'cuda':
            print(f"Используем GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA версия: {torch.version.cuda}")
            print(
                f"Доступно GPU памяти: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.history = {'train_loss': [], 'val_loss': [],
                        'val_iou': [], 'val_dice': []}

    def train_epoch(self, dataloader, criterion, optimizer):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc='Training',
                            ascii=True if os.name == 'nt' else False)
        for images, masks in progress_bar:
            try:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                optimizer.zero_grad()
                loss.backward()
                # Градиентный клиппинг для стабильности
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'device': str(
                    next(self.model.parameters()).device)})
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        return total_loss / len(dataloader) if len(dataloader) > 0 else 0

    def validate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0
        metrics = {'iou': [], 'dice': [], 'precision': [], 'recall': []}
        with torch.no_grad():
            for images, masks in tqdm(dataloader, desc='Validation', ascii=True if os.name == 'nt' else False):
                try:
                    images = images.to(self.device, non_blocking=True)
                    masks = masks.to(self.device, non_blocking=True)
                    outputs = self.model(images)
                    loss = criterion(outputs, masks)
                    total_loss += loss.item()
                    for i in range(outputs.shape[0]):
                        m = calculate_metrics(outputs[i], masks[i])
                        for key in metrics:
                            metrics[key].append(m[key])
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue

        avg_metrics = {key: np.mean(values) for key, values in metrics.items()} if any(
            len(v) > 0 for v in metrics.values()) else {k: 0.0 for k in metrics}
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        return avg_loss, avg_metrics

    def fit(self, train_loader, val_loader, epochs=50, lr=1e-3, save_best=True, checkpoint_dir='checkpoints'):
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer = optim.Adam(self.model.parameters(),
                               lr=lr, weight_decay=1e-5)
        scheduler = safe_ReduceLROnPlateau(optimizer, patience=7, factor=0.5)
        criterion = CombinedLoss(bce_weight=0.5)
        best_val_iou = 0
        patience_counter = 0
        max_patience = 15

        print(f"Начинаем обучение на устройстве: {self.device}")
        print(
            f"Параметры модели находятся на: {next(self.model.parameters()).device}")

        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)

            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            val_loss, val_metrics = self.validate(val_loader, criterion)

            if scheduler is not None:
                scheduler.step(val_loss)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['val_dice'].append(val_metrics['dice'])

            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print(f'Val IoU: {val_metrics["iou"]:.4f}')
            print(f'Val Dice: {val_metrics["dice"]:.4f}')
            print(f'Val Precision: {val_metrics["precision"]:.4f}')
            print(f'Val Recall: {val_metrics["recall"]:.4f}')
            print(f'Model device: {next(self.model.parameters()).device}')

            # Early stopping
            if val_metrics['iou'] > best_val_iou:
                best_val_iou = val_metrics['iou']
                patience_counter = 0
                if save_best:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_iou': val_metrics['iou'],
                        'val_loss': val_loss
                    }
                    torch.save(checkpoint, os.path.join(
                        checkpoint_dir, 'best_model.pth'))
                    print(f'✓ Saved best model with IoU: {best_val_iou:.4f}')
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break

        return self.history

    def plot_history(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Loss
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_title('Training and Validation Loss')

        # Добавляем финальное значение
        final_val_loss = self.history['val_loss'][-1]
        axes[0].text(0.98, 0.98, f'Final Val Loss: {final_val_loss:.4f}',
                     transform=axes[0].transAxes, ha='right', va='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # IoU
        axes[1].plot(self.history['val_iou'], label='Val IoU', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('IoU')
        axes[1].legend()
        axes[1].grid(True)
        axes[1].set_title('Validation IoU')

        # Добавляем лучшее и финальное значение
        best_iou = max(self.history['val_iou'])
        final_iou = self.history['val_iou'][-1]
        axes[1].text(0.98, 0.98, f'Best: {best_iou:.4f}\nFinal: {final_iou:.4f}',
                     transform=axes[1].transAxes, ha='right', va='top',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        # Dice
        axes[2].plot(self.history['val_dice'],
                     label='Val Dice', color='orange')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Dice Score')
        axes[2].legend()
        axes[2].grid(True)
        axes[2].set_title('Validation Dice Score')

        # Добавляем лучшее и финальное значение
        best_dice = max(self.history['val_dice'])
        final_dice = self.history['val_dice'][-1]
        axes[2].text(0.98, 0.98, f'Best: {best_dice:.4f}\nFinal: {final_dice:.4f}',
                     transform=axes[2].transAxes, ha='right', va='top',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(TRAINING_HISTORY_DIR, f'training_history_{timestamp}.png'),
                    dpi=150, bbox_inches='tight')
        plt.show()

# ---------------- Safe checkpoint loader helper ----------------


def _safe_torch_load(path, map_location='cpu', allow_unsafe=False):
    """
    Robust loader for checkpoints:
      - try weights-only load (safe)
      - if fails, try allowlisting numpy objects and load full pickle inside safe_globals
      - if allow_unsafe=True, fallback to full load (weights_only=False)
    """
    import torch
    try:
        # Явно пробуем безопасную загрузку только весов
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as orig_exc:
        # Попробуем allowlist numpy-объекты, которые часто мешают (dtype, multiarray.scalar)
        try:
            import numpy as _np
            allowed = []
            if hasattr(_np, 'dtype'):
                allowed.append(_np.dtype)
            # попытка получить multiarray.scalar (если присутствует)
            try:
                core = getattr(_np, '_core', None)
                if core is not None:
                    ma = getattr(core, 'multiarray', None)
                    if ma is not None and hasattr(ma, 'scalar'):
                        allowed.append(ma.scalar)
            except Exception:
                pass

            # Если присутствуют API safe_globals / add_safe_globals — используем их
            if hasattr(torch.serialization, 'safe_globals'):
                with torch.serialization.safe_globals(allowed):
                    return torch.load(path, map_location=map_location, weights_only=False)
            elif hasattr(torch.serialization, 'add_safe_globals'):
                torch.serialization.add_safe_globals(allowed)
                return torch.load(path, map_location=map_location, weights_only=False)
        except Exception:
            # ничего — идём дальше к опасному варианту
            pass

        # Финальный fallback: если пользователь явно разрешил небезопасную загрузку
        if allow_unsafe:
            try:
                return torch.load(path, map_location=map_location, weights_only=False)
            except Exception:
                pass

        # если ничего не помогло — пробросим исходную ошибку
        raise orig_exc

# ============= ОБЪЕДИНЕНИЕ ДАННЫХ ИЗ НЕСКОЛЬКИХ ИСТОЧНИКОВ =============


def combine_multiple_datasets(frames_dirs, masks_dirs):
    """Объединение данных из нескольких папок"""
    all_image_paths = []
    all_mask_paths = []

    # Преобразуем в списки если переданы строки
    if isinstance(frames_dirs, str):
        frames_dirs = [frames_dirs]
    if isinstance(masks_dirs, str):
        masks_dirs = [masks_dirs]

    print("Объединение данных из папок:")

    for frames_dir, masks_dir in zip(frames_dirs, masks_dirs):
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        mask_extensions = ['*.png', '*.jpg', '*.jpeg']

        # Собираем изображения
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(frames_dir, ext)))
        image_paths = sorted(image_paths)

        # Собираем маски
        mask_paths = []
        for ext in mask_extensions:
            mask_paths.extend(glob.glob(os.path.join(masks_dir, ext)))
        mask_paths = sorted(mask_paths)

        print(f"  • {frames_dir}: {len(image_paths)} изображений")
        print(f"  • {masks_dir}: {len(mask_paths)} масок")

        # Проверяем соответствие
        min_samples = min(len(image_paths), len(mask_paths))
        if len(image_paths) != len(mask_paths):
            print(
                f"    Предупреждение: количество не совпадает, используем {min_samples} пар")
            image_paths = image_paths[:min_samples]
            mask_paths = mask_paths[:min_samples]

        all_image_paths.extend(image_paths)
        all_mask_paths.extend(mask_paths)

    print(f"\nИтого: {len(all_image_paths)} пар изображение-маска")
    return all_image_paths, all_mask_paths

# ============= INFERENCE AND VISUALIZATION =============


def predict_single_image(model, image_path, device='cuda', img_size=(256, 256)):
    """Предсказание для одного изображения"""
    model.eval()
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = (image.shape[1], image.shape[0])
    image_resized = cv2.resize(image_rgb, img_size)
    image_tensor = torch.from_numpy(image_resized.astype(np.float32) / 255.0)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(image_tensor)

    pred_mask = prediction.squeeze().cpu().numpy()
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    pred_mask = cv2.resize(pred_mask, original_size,
                           interpolation=cv2.INTER_NEAREST)

    return image_rgb, pred_mask


def visualize_results(model, val_dataset, device='cuda', num_samples=10, save_path=None, img_size=(256, 256)):
    """
    Визуализация результатов модели: оригинал + предсказание поверх оригинала

    Args:
        model: обученная модель
        val_dataset: валидационный датасет
        device: устройство для вычислений
        num_samples: количество образцов для визуализации
        save_path: путь для сохранения изображения (опционально)
        img_size: размер изображений
    """
    import os as _os
    import platform as _platform
    from datetime import datetime

    model.eval()

    # Случайно выбираем образцы из валидационной выборки
    total_samples = len(val_dataset)
    if num_samples > total_samples:
        num_samples = total_samples
        print(
            f"Количество образцов уменьшено до {num_samples} (размер валидационной выборки)")

    # Получаем случайные индексы
    random_indices = random.sample(range(total_samples), num_samples)

    # Создаем сетку 2 строки x num_samples столбцов
    fig, axes = plt.subplots(2, num_samples, figsize=(3*num_samples, 6))
    if num_samples == 1:
        axes = axes.reshape(-1, 1)

    with torch.no_grad():
        for i, idx in enumerate(random_indices):
            try:
                # Получаем изображение и маску из датасета
                image_tensor, true_mask = val_dataset[idx]

                # Добавляем batch dimension
                image_batch = image_tensor.unsqueeze(0).to(device)

                # Делаем предсказание
                pred_mask = model(image_batch)

                # Конвертируем в numpy для визуализации
                image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
                pred_mask_np = pred_mask.squeeze().cpu().numpy()

                # Бинаризация предсказания
                pred_mask_binary = (pred_mask_np > 0.5).astype(np.uint8)

                # Вычисляем метрики для данного образца
                iou = calculate_metrics(
                    pred_mask.cpu(), true_mask.unsqueeze(0))['iou']

                # Создаем наложение маски на оригинальное изображение
                overlay = image_np.copy()
                # Полупрозрачное красное наложение для областей пламени
                flame_mask = pred_mask_binary > 0
                overlay[flame_mask] = overlay[flame_mask] * \
                    0.6 + np.array([1.0, 0.2, 0.2]) * 0.4

                # Отображаем результаты
                axes[0, i].imshow(image_np)
                axes[0, i].set_title(f'Оригинал {i+1}', fontsize=10)
                axes[0, i].axis('off')

                axes[1, i].imshow(overlay)
                axes[1, i].set_title(
                    f'Предсказание (IoU: {iou:.3f})', fontsize=10)
                axes[1, i].axis('off')

            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                # В случае ошибки показываем пустые изображения
                for j in range(2):
                    axes[j, i].imshow(np.zeros((img_size[1], img_size[0], 3)))
                    axes[j, i].axis('off')
                    if j == 1:
                        axes[j, i].set_title(f'Ошибка {i+1}', fontsize=10)

    plt.tight_layout()

   # Создаем папку для результатов
    results_dir = 'models/results/model_results/predict_and_original'
    _os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = _os.path.join(results_dir, f'comparison_grid_{timestamp}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Comparison grid saved to: {save_path}")

    try:
        system = _platform.system()
        if system == "Windows":
            _os.startfile(save_path)
        elif system == "Darwin":  # macOS
            _os.system(f'open "{save_path}"')
        else:  # Linux
            _os.system(f'xdg-open "{save_path}"')

        print(f"Изображение открыто в системном просмотрщике: {save_path}")
    except Exception as e:
        print(f"Не удалось автоматически открыть изображение: {e}")
        print(f"Вы можете открыть файл вручную: {save_path}")

    plt.close(fig)


def create_comparison_grid(model, image_paths, mask_paths, device='cuda', num_samples=10, img_size=(256, 256)):
    """
    Создает сетку сравнения: оригинал + предсказание поверх оригинала
    """
    model.eval()

    if len(image_paths) < num_samples:
        num_samples = len(image_paths)
        print(f"Количество образцов уменьшено до {num_samples}")

    # Выбираем случайные образцы
    indices = random.sample(range(len(image_paths)), num_samples)

    # Создаем сетку 2 строки x num_samples столбцов
    fig, axes = plt.subplots(2, num_samples, figsize=(3*num_samples, 6))
    if num_samples == 1:
        axes = axes.reshape(-1, 1)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            try:
                # Загружаем и обрабатываем изображение
                image = cv2.imread(image_paths[idx])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_resized = cv2.resize(image, img_size)

                # Загружаем маску для метрик
                mask = cv2.imread(mask_paths[idx], cv2.IMREAD_GRAYSCALE)
                mask_resized = cv2.resize(mask, img_size)

                # Подготавливаем для модели
                image_tensor = torch.from_numpy(
                    image_resized.astype(np.float32) / 255.0)
                image_tensor = image_tensor.permute(
                    2, 0, 1).unsqueeze(0).to(device)

                # Предсказание
                pred_mask = model(image_tensor)
                pred_mask_np = pred_mask.squeeze().cpu().numpy()
                pred_mask_binary = (pred_mask_np > 0.5).astype(np.uint8)

                # Вычисляем IoU
                true_mask_tensor = torch.from_numpy(mask_resized.astype(
                    np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
                iou = calculate_metrics(
                    pred_mask.cpu(), true_mask_tensor)['iou']

                # Создаем наложение предсказания на оригинал
                image_normalized = image_resized.astype(np.float32) / 255.0
                overlay = image_normalized.copy()
                flame_mask = pred_mask_binary > 0
                # Полупрозрачное красное наложение
                overlay[flame_mask] = overlay[flame_mask] * \
                    0.6 + np.array([1.0, 0.2, 0.2]) * 0.4

                # Отображение
                axes[0, i].imshow(image_normalized)
                axes[0, i].set_title(f'Оригинал {i+1}', fontsize=10)
                axes[0, i].axis('off')

                axes[1, i].imshow(overlay)
                axes[1, i].set_title(
                    f'Предсказание (IoU: {iou:.3f})', fontsize=10)
                axes[1, i].axis('off')

            except Exception as e:
                print(f"Error processing image {image_paths[idx]}: {e}")
                for j in range(2):
                    axes[j, i].text(0.5, 0.5, 'Ошибка', ha='center',
                                    va='center', transform=axes[j, i].transAxes)
                    axes[j, i].axis('off')

    plt.tight_layout()

    # Создаем папку для результатов
    results_dir = 'data/model_results'
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(results_dir, f'comparison_grid_{timestamp}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Comparison grid saved to: {save_path}")

    # Открываем в системном просмотрщике
    try:
        import platform

        system = platform.system()
        if system == "Windows":
            os.startfile(save_path)
        elif system == "Darwin":  # macOS
            os.system(f'open "{save_path}"')
        else:  # Linux
            os.system(f'xdg-open "{save_path}"')

        print(f"Изображение открыто в системном просмотрщике: {save_path}")

    except Exception as e:
        print(f"Не удалось автоматически открыть изображение: {e}")
        print(f"Вы можете открыть файл вручную: {save_path}")

    plt.close(fig)


def test_model_on_folder(model, test_folder, device='cuda', img_size=(256, 256), save_results=True):
    """
    Тестирует модель на папке с изображениями и создает наложения предсказаний
    """
    model.eval()
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']

    test_images = []
    for ext in image_extensions:
        test_images.extend(glob.glob(os.path.join(test_folder, ext)))

    if len(test_images) == 0:
        print(f"Не найдено изображений в {test_folder}")
        return

    print(f"Найдено {len(test_images)} изображений для тестирования")

    results_folder = 'data/test_results'
    if save_results:
        os.makedirs(results_folder, exist_ok=True)

    for i, img_path in enumerate(tqdm(test_images, desc="Обработка изображений")):
        try:
            original_image, pred_mask = predict_single_image(
                model, img_path, device=device, img_size=img_size
            )

            if save_results:
                # Сохраняем результат
                filename = os.path.splitext(os.path.basename(img_path))[0]
                result_path = os.path.join(
                    results_folder, f"{filename}_result.png")

                # Создаем композитное изображение: оригинал + наложение
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))

                # Оригинальное изображение
                axes[0].imshow(original_image)
                axes[0].set_title('Оригинал', fontsize=12)
                axes[0].axis('off')

                # Создаем наложение предсказания на оригинал
                image_normalized = original_image.astype(np.float32) / 255.0
                overlay = image_normalized.copy()
                flame_mask = pred_mask > 127
                # Полупрозрачное красное наложение для пламени
                overlay[flame_mask] = overlay[flame_mask] * \
                    0.6 + np.array([1.0, 0.2, 0.2]) * 0.4

                axes[1].imshow(overlay)
                axes[1].set_title(
                    'Предсказание модели (наложение)', fontsize=12)
                axes[1].axis('off')

                plt.tight_layout()
                plt.savefig(result_path, dpi=200,
                            bbox_inches='tight', facecolor='white')
                plt.close()

        except Exception as e:
            print(f"Ошибка при обработке {img_path}: {e}")

    print(f"Результаты сохранены в папку: {results_folder}")

    # Открываем папку с результатами в системном проводнике
    try:
        import platform

        system = platform.system()
        if system == "Windows":
            os.startfile(results_folder)
        elif system == "Darwin":  # macOS
            os.system(f'open "{results_folder}"')
        else:  # Linux
            os.system(f'xdg-open "{results_folder}"')

        print(f"Папка с результатами открыта: {results_folder}")

    except Exception as e:
        print(f"Не удалось автоматически открыть папку: {e}")
        print(f"Вы можете открыть папку вручную: {results_folder}")


# ============= MAIN + train_and_export =============

def main(frames_dir='data/frames/1video', masks_dir='data/masks/1video',
         img_size=(256, 256), batch_size=8, epochs=30, learning_rate=1e-3,
         val_split=0.2, use_temporal=False, force_gpu=True, save_checkpoints=True):
    print('=' * 50)
    print('ОБУЧЕНИЕ МОДЕЛИ СЕГМЕНТАЦИИ ПЛАМЕНИ')
    print('=' * 50)

    # Настройка устройства
    if force_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    elif force_gpu and not torch.cuda.is_available():
        print('⚠ CUDA недоступна! Проверьте установку PyTorch с CUDA поддержкой')
        return None, None
    else:
        device = torch.device('cpu')

    # Инициализируем пути (чтобы избежать UnboundLocalError)
    image_paths = None
    mask_paths = None

    # Обработка путей к данным (поддержка списков папок)
    if isinstance(frames_dir, list) and isinstance(masks_dir, list):
        # Множественные источники данных
        image_paths, mask_paths = combine_multiple_datasets(
            frames_dir, masks_dir)
    else:
        # Одиночный источник данных
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        mask_extensions = ['*.png', '*.jpg', '*.jpeg']

        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(frames_dir, ext)))
        image_paths = sorted(image_paths)

        mask_paths = []
        for ext in mask_extensions:
            mask_paths.extend(glob.glob(os.path.join(masks_dir, ext)))
        mask_paths = sorted(mask_paths)

    # На этом моменте image_paths и mask_paths гарантированно определены
    if image_paths is None or mask_paths is None or len(image_paths) == 0 or len(mask_paths) == 0:
        print(f'Ошибка: не найдены изображения или маски.')
        print(
            f'Изображения в {frames_dir}: {len(image_paths) if image_paths is not None else 0}')
        print(
            f'Маски в {masks_dir}: {len(mask_paths) if mask_paths is not None else 0}')
        return None, None

    print(f'Найдено изображений: {len(image_paths)}')
    print(f'Найдено масок: {len(mask_paths)}')

    # Проверка соответствия количества
    min_samples = min(len(image_paths), len(mask_paths))
    if len(image_paths) != len(mask_paths):
        print(
            f'Внимание: количество изображений и масок не совпадает. Используем {min_samples} образцов.')
        image_paths = image_paths[:min_samples]
        mask_paths = mask_paths[:min_samples]

    # Разделение на train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, mask_paths, test_size=val_split, random_state=42, shuffle=True)

    print(f'Обучающая выборка: {len(X_train)} образцов')
    print(f'Валидационная выборка: {len(X_val)} образцов')

    # Создание датасетов
    train_dataset = FlameDataset(
        X_train, y_train, img_size=img_size, augment=True)
    val_dataset = FlameDataset(X_val, y_val, img_size=img_size, augment=False)

    # Оптимизация DataLoader для производительности
    num_workers = min(4, os.cpu_count()) if device.type == 'cuda' else 2
    pin_memory = device.type == 'cuda'

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )

    # Создание модели
    if use_temporal:
        model = TemporalUNet(in_channels=3, out_channels=1, temporal_frames=3)
        print('Использую TemporalUNet модель')
    else:
        model = ImprovedUNet(in_channels=3, out_channels=1)
        print('Использую ImprovedUNet модель')

    model = model.to(device)

    # Подсчет параметров модели
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f'Общее количество параметров: {total_params:,}')
    print(f'Обучаемые параметры: {trainable_params:,}')

    # Обучение
    trainer = FlameSegmentationTrainer(model, device=device.type)
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=epochs,
        lr=learning_rate,
        save_best=save_checkpoints
    )

    # Построение графиков обучения
    try:
        trainer.plot_history()
    except Exception as e:
        print(f"Не удалось построить график: {e}")

    # Загрузка лучшей модели
    ckpt_path = os.path.join('checkpoints', 'best_model.pth')
    model_for_return = trainer.model

    if save_checkpoints and os.path.exists(ckpt_path):
        try:
            checkpoint = _safe_torch_load(ckpt_path, map_location=device)
            state = checkpoint.get('model_state_dict', checkpoint)
            if isinstance(state, dict):
                model_for_return.load_state_dict(state)
                print(
                    f"\nЛучшая модель загружена (val_iou: {checkpoint.get('val_iou', 0):.4f})")
            else:
                print('Loaded checkpoint does not contain state_dict, skipping load.')
        except Exception as e:
            print('Не удалось загрузить checkpoints/best_model.pth:', e)

    # Финальная визуализация результатов
    print('\n' + '=' * 50)
    print('ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ МОДЕЛИ')
    print('=' * 50)

    try:
        # Используем валидационный датасет для визуализации
        visualize_results(
            model_for_return,
            val_dataset,
            device=device,
            num_samples=min(10, len(val_dataset)),
            img_size=img_size
        )
    except Exception as e:
        print(f"Ошибка при создании визуализации: {e}")
        # Альтернативный способ через пути к файлам
        try:
            print("Пробуем альтернативный метод визуализации...")
            create_comparison_grid(
                model_for_return,
                X_val,
                y_val,
                device=device,
                num_samples=min(10, len(X_val)),
                img_size=img_size
            )
        except Exception as e2:
            print(f"Альтернативная визуализация также не удалась: {e2}")

    print('\n' + '=' * 50)
    print('ОБУЧЕНИЕ ЗАВЕРШЕНО')
    print('=' * 50)

    return model_for_return, history


def train_and_export(frames_dir='data/frames/1video', masks_dir='data/masks/1video',
                     img_size=(256, 256), batch_size=8, epochs=30, learning_rate=1e-3,
                     val_split=0.2, use_temporal=False, checkpoint_dir='checkpoints',
                     force_gpu=True, save_checkpoints=True, export_model=True,
                     allow_unsafe_checkpoint_load=False, visualize_results_flag=True):
    """
    Полный цикл обучения и экспорта модели с визуализацией результатов

    Args:
        frames_dir: путь к директории с изображениями
        masks_dir: путь к директории с масками
        img_size: размер изображений для обучения
        batch_size: размер батча
        epochs: количество эпох
        learning_rate: скорость обучения
        val_split: доля валидационной выборки
        use_temporal: использовать TemporalUNet
        checkpoint_dir: директория для сохранения чекпоинтов
        force_gpu: принудительно использовать GPU
        save_checkpoints: сохранять чекпоинты
        export_model: экспортировать финальную модель
        allow_unsafe_checkpoint_load: разрешить небезопасную загрузку
        visualize_results_flag: показать визуализацию результатов

    Returns:
        tuple: (model, history, final_metrics)
    """

    try:
        model, history = main(
            frames_dir=frames_dir,
            masks_dir=masks_dir,
            img_size=img_size,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            val_split=val_split,
            use_temporal=use_temporal,
            force_gpu=force_gpu,
            save_checkpoints=save_checkpoints
        )
    except Exception as e:
        print('Ошибка при вызове main():', e)
        import traceback
        traceback.print_exc()
        return None, None, None

    if model is None:
        print("Модель не была обучена.")
        return None, None, None

    # Собираем финальные метрики из истории обучения
    final_metrics = None
    if history and 'val_iou' in history and len(history['val_iou']) > 0:
        final_metrics = {
            'final_val_iou': history['val_iou'][-1],
            'final_val_dice': history['val_dice'][-1],
            'best_val_iou': max(history['val_iou']),
            'best_val_dice': max(history['val_dice']),
            'final_val_loss': history['val_loss'][-1],
            'final_train_loss': history['train_loss'][-1],
            'avg_val_iou': sum(history['val_iou']) / len(history['val_iou']),
            'avg_val_dice': sum(history['val_dice']) / len(history['val_dice'])
        }

        # Выводим метрики в консоль
        print('\n' + '='*60)
        print('ФИНАЛЬНЫЕ МЕТРИКИ МОДЕЛИ:')
        print('='*60)
        print(
            f"Финальный IoU (validation): {final_metrics['final_val_iou']:.4f}")
        print(
            f"Финальный Dice (validation): {final_metrics['final_val_dice']:.4f}")
        print(f"Лучший IoU: {final_metrics['best_val_iou']:.4f}")
        print(f"Лучший Dice: {final_metrics['best_val_dice']:.4f}")
        print(f"Средний IoU: {final_metrics['avg_val_iou']:.4f}")
        print(f"Средний Dice: {final_metrics['avg_val_dice']:.4f}")
        print(
            f"Финальный validation loss: {final_metrics['final_val_loss']:.4f}")
        print(
            f"Финальный training loss: {final_metrics['final_train_loss']:.4f}")
        print('='*60)

    # Экспорт модели
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(MODEL_EXPORT_DIR, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, 'best_model.pth')

    try:
        if save_checkpoints and os.path.exists(ckpt_path):
            try:
                ckpt = _safe_torch_load(
                    ckpt_path,
                    map_location='cpu',
                    allow_unsafe=allow_unsafe_checkpoint_load
                )
                state = ckpt.get('model_state_dict', ckpt)
                if isinstance(state, dict):
                    # Сохраняем только веса
                    torch.save(state, os.path.join(
                        MODEL_EXPORT_DIR, 'flame_unet.pth'))
                    print(
                        'Экспортирован state_dict -> flame_unet.pth (из checkpoints/best_model.pth)')

                    # Сохраняем также полную информацию о модели
                    model_info = {
                        'model_state_dict': state,
                        'model_architecture': 'ImprovedUNet' if not use_temporal else 'TemporalUNet',
                        'img_size': img_size,
                        'training_epochs': epochs,
                        'final_metrics': final_metrics,
                        'checkpoint_val_iou': ckpt.get('val_iou', 0),
                        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                    }
                    torch.save(model_info, os.path.join(
                        MODEL_EXPORT_DIR, 'flame_unet_full.pth'))
                    print('Сохранена полная информация о модели -> flame_unet_full.pth')

                    # Сохраняем метрики в отдельный текстовый файл для удобства
                    metrics_file = os.path.join(
                        MODEL_EXPORT_DIR, 'model_metrics.txt')
                    with open(metrics_file, 'w', encoding='utf-8') as f:
                        f.write('МЕТРИКИ ОБУЧЕННОЙ МОДЕЛИ СЕГМЕНТАЦИИ ПЛАМЕНИ\n')
                        f.write('='*60 + '\n\n')
                        f.write('ИНФОРМАЦИЯ О МОДЕЛИ:\n')
                        f.write('-'*60 + '\n')
                        f.write(
                            f"Архитектура: {model_info['model_architecture']}\n")
                        f.write(f"Размер изображений: {img_size}\n")
                        f.write(f"Эпох обучения: {epochs}\n")
                        f.write(f"Размер батча: {batch_size}\n")
                        f.write(f"Скорость обучения: {learning_rate}\n")
                        f.write(f"Validation split: {val_split}\n")
                        f.write(
                            f"Дата обучения: {model_info['timestamp']}\n\n")

                        if final_metrics:
                            f.write('ФИНАЛЬНЫЕ МЕТРИКИ:\n')
                            f.write('-'*60 + '\n')
                            f.write(
                                f"Validation IoU (финальный): {final_metrics['final_val_iou']:.4f}\n")
                            f.write(
                                f"Validation Dice (финальный): {final_metrics['final_val_dice']:.4f}\n")
                            f.write(
                                f"Validation IoU (лучший): {final_metrics['best_val_iou']:.4f}\n")
                            f.write(
                                f"Validation Dice (лучший): {final_metrics['best_val_dice']:.4f}\n")
                            f.write(
                                f"Validation IoU (средний): {final_metrics['avg_val_iou']:.4f}\n")
                            f.write(
                                f"Validation Dice (средний): {final_metrics['avg_val_dice']:.4f}\n")
                            f.write(
                                f"Validation Loss (финальный): {final_metrics['final_val_loss']:.4f}\n")
                            f.write(
                                f"Training Loss (финальный): {final_metrics['final_train_loss']:.4f}\n\n")

                            f.write('ИНТЕРПРЕТАЦИЯ МЕТРИК:\n')
                            f.write('-'*60 + '\n')
                            f.write("IoU (Intersection over Union):\n")
                            f.write(
                                "  - Измеряет пересечение предсказания и истинной маски\n")
                            f.write(
                                "  - Диапазон: 0.0 (плохо) - 1.0 (идеально)\n")
                            f.write("  - >0.7 считается хорошим результатом\n\n")
                            f.write("Dice Score:\n")
                            f.write(
                                "  - Схож с IoU, но больше веса на перекрытие\n")
                            f.write(
                                "  - Диапазон: 0.0 (плохо) - 1.0 (идеально)\n")
                            f.write("  - >0.8 считается хорошим результатом\n\n")

                    print(f'Метрики сохранены в: {metrics_file}')

                    return model, history, final_metrics

            except Exception as e:
                print(
                    'Failed to export from checkpoint (will try to export current model state):', e)

        # Если не удалось загрузить из чекпоинта, сохраняем текущее состояние модели
        if model is not None and isinstance(model, torch.nn.Module) and export_model:
            torch.save(model.state_dict(), os.path.join(
                MODEL_EXPORT_DIR, 'flame_unet.pth'))
            print('Сохранены веса модели в flame_unet.pth')

            # Дополнительная информация
            model_info = {
                'model_state_dict': model.state_dict(),
                'model_architecture': 'ImprovedUNet' if not use_temporal else 'TemporalUNet',
                'img_size': img_size,
                'training_epochs': epochs,
                'final_metrics': final_metrics,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            torch.save(model_info, os.path.join(
                MODEL_EXPORT_DIR, 'flame_unet_full.pth'))
            print('Сохранена полная информация о модели -> flame_unet_full.pth')

            # Сохраняем метрики
            if final_metrics:
                metrics_file = os.path.join(
                    MODEL_EXPORT_DIR, 'model_metrics.txt')
                with open(metrics_file, 'w', encoding='utf-8') as f:
                    f.write('МЕТРИКИ ОБУЧЕННОЙ МОДЕЛИ СЕГМЕНТАЦИИ ПЛАМЕНИ\n')
                    f.write('='*60 + '\n\n')
                    f.write('ИНФОРМАЦИЯ О МОДЕЛИ:\n')
                    f.write('-'*60 + '\n')
                    f.write(
                        f"Архитектура: {model_info['model_architecture']}\n")
                    f.write(f"Размер изображений: {img_size}\n")
                    f.write(f"Эпох обучения: {epochs}\n")
                    f.write(f"Дата обучения: {model_info['timestamp']}\n\n")
                    f.write('ФИНАЛЬНЫЕ МЕТРИКИ:\n')
                    f.write('-'*60 + '\n')
                    for key, value in final_metrics.items():
                        f.write(f"{key}: {value:.4f}\n")

                print(f'Метрики сохранены в: {metrics_file}')

            return model, history, final_metrics

    except Exception as e:
        print('Не удалось экспортировать модель:', e)
        import traceback
        traceback.print_exc()

    return model, history, final_metrics


def load_trained_model(model_path='flame_unet.pth', device='cuda', use_temporal=False, img_size=(256, 256)):
    """
    Загружает обученную модель из файла

    Args:
        model_path: путь к файлу модели
        device: устройство для загрузки
        use_temporal: использовать TemporalUNet архитектуру
        img_size: размер изображений

    Returns:
        Загруженная модель
    """
    try:
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
            print("CUDA недоступна, используем CPU")

        device = torch.device(device)

        # Создаем архитектуру модели
        if use_temporal:
            model = TemporalUNet(
                in_channels=3, out_channels=1, temporal_frames=3)
        else:
            model = ImprovedUNet(in_channels=3, out_channels=1)

        # Загружаем веса
        if model_path.endswith('_full.pth'):
            # Полная информация о модели
            checkpoint = _safe_torch_load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(
                f"Загружена модель: {checkpoint.get('model_architecture', 'Unknown')}")
            print(
                f"Обучена эпох: {checkpoint.get('training_epochs', 'Unknown')}")
            print(
                f"Финальные метрики: {checkpoint.get('final_metrics', 'Unknown')}")
        else:
            # Только веса
            state_dict = _safe_torch_load(model_path, map_location=device)
            model.load_state_dict(state_dict)

        model = model.to(device)
        model.eval()
        print(f"Модель успешно загружена на {device}")
        return model

    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return None


if __name__ == "__main__":
    # Пример использования
    model, history = train_and_export(
        frames_dir='data/frames/1video',
        masks_dir='data/masks/1video',
        img_size=(256, 256),
        batch_size=8,
        epochs=30,
        learning_rate=1e-3,
        force_gpu=True,
        visualize_results_flag=True
    )
