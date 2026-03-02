import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR  = r"C:\Users\goray\Desktop\FYP\Data-kaggle\New Plant Diseases Dataset(Augmented)"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "valid")  # change to "val" if needed

BATCH_SIZE    = 32
EPOCHS        = 15
LR            = 1e-3
IMG_SIZE      = 224
SAVE_PATH     = "plant_disease_model.pth"
UNFREEZE_EPOCH = 5


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for i, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)
        if (i + 1) % 50 == 0:
            print(f"  Step {i+1}/{len(loader)} | "
                  f"Loss: {running_loss/total:.4f} | Acc: {100.*correct/total:.2f}%")
    return running_loss / total, 100. * correct / total


def val_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total   += labels.size(0)
    return running_loss / total, 100. * correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print("Loading datasets...")
    train_ds = ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds   = ImageFolder(VAL_DIR,   transform=val_tf)
    num_classes = len(train_ds.classes)
    print(f"Classes: {num_classes} | Train: {len(train_ds)} | Val: {len(val_ds)}")

    # num_workers=0 avoids Windows multiprocessing crash
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=False)

    print("Building model (ResNet50, pretrained)...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, num_classes)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    print("\n" + "="*60)
    print("Training started!")
    print("="*60)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        if epoch == UNFREEZE_EPOCH + 1:
            print("\n>>> Unfreezing full backbone for fine-tuning <<<\n")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=LR * 0.1, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=EPOCHS - UNFREEZE_EPOCH)

        print(f"\nEpoch [{epoch}/{EPOCHS}]")
        print("-" * 40)
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}   | Val   Acc: {val_acc:.2f}%")
        print(f"  Time: {time.time()-t0:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_to_idx": train_ds.class_to_idx,
            }, SAVE_PATH)
            print(f"  ✓ Best model saved! (val_acc={val_acc:.2f}%)")

    print(f"\nDone! Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {SAVE_PATH}")


# REQUIRED on Windows — all code must be inside this guard
if __name__ == "__main__":
    main()
