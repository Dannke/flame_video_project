# src/train_unet.py
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from archive.dataset import FlameDataset

# очень простая U-Net-like модель
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.Conv2d(32,32,3,padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.Conv2d(64,64,3,padding=1), nn.ReLU())
        self.up = nn.ConvTranspose2d(64,32,2,stride=2)
        self.dec = nn.Sequential(nn.Conv2d(64,32,3,padding=1), nn.ReLU(), nn.Conv2d(32,16,3,padding=1), nn.ReLU())
        self.out = nn.Conv2d(16,1,1)
    def forward(self,x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        u = self.up(e2)
        cat = torch.cat([u, e1], dim=1)
        d = self.dec(cat)
        return torch.sigmoid(self.out(d))

def loss_fn(pred, target):
    bce = nn.BCELoss()(pred, target)
    # dice
    smooth=1.0
    pred_flat = pred.view(-1)
    targ_flat = target.view(-1)
    intersection = (pred_flat * targ_flat).sum()
    dice = 1 - (2.*intersection + smooth) / (pred_flat.sum() + targ_flat.sum() + smooth)
    return bce + dice

def train(images_dir, masks_dir, epochs=8, batch=8, lr=1e-3, size=(256,256)):
    ds = FlameDataset(images_dir, masks_dir, size=size)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleUNet().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train(); tot=0.0
        for imgs, masks in dl:
            imgs = imgs.to(device); masks = masks.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, masks)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"Epoch {ep+1}/{epochs} loss={tot/len(dl):.4f}")
    torch.save(model.state_dict(), "flame_unet.pth")
    print("Saved flame_unet.pth")

if __name__ == "__main__":
    import sys
    train(sys.argv[1], sys.argv[2], epochs=8)
