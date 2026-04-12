# EfficientNet-B4 학습 코드
# 과적합 방지 전략: Label Smoothing(0.05) + AdamW + Early Stopping(patience=7)
# Warm-up: 사전학습 가중치 보호를 위해 3 epoch 선형 증가 후 CosineAnnealing 적용
# AMP: GPU 메모리 절약 및 학습 속도 개선, GradScaler로 수치 불안정 방지

import time
from pathlib import Path
import pandas as pd

import torch
import timm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# EarlyStopping

class EarlyStopping:
    """
    mode='min'  : score가 작을수록 좋은 경우 (예: val_loss)
    mode='max'  : score가 클수록 좋은 경우 (예: val_acc)
    """
    def __init__(self, patience=7, mode="min", delta=1e-4):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (
            score < self.best_score - self.delta
            if self.mode == "min"
            else score > self.best_score + self.delta
        )

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop



# Train one epoch (AMP)

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    preds_all, labels_all = [], []

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)

        preds_all.extend(preds.detach().cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)
    acc = accuracy_score(labels_all, preds_all)
    return avg_loss, acc



# Validation (FP32)

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    preds_all, labels_all = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Valid", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            preds = outputs.argmax(dim=1)

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    avg_loss = val_loss / len(loader)
    acc = accuracy_score(labels_all, preds_all)
    return avg_loss, acc



# Main
def main():
    history = []

    # -------- Config --------
    MODEL_NAME = "efficientnet_b4"
    TRAIN_DIR = Path("E:/classification_dataset_split_5to1_filtered_copied/train")
    VAL_DIR = Path("E:/classification_dataset_split_5to1_filtered_copied/val")
    CHECKPOINT_DIR = Path("checkpoints_b4_68class")
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    IMG_SIZE = 352
    BATCH_SIZE = 16
    EPOCHS = 30

    BASE_LR = 1e-4
    WARMUP_START_LR = 1e-6
    WARMUP_EPOCHS = 3

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"▶ Device: {DEVICE}")

    # -------- Transform --------
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # YOLO crop + padding 이미지 기준
    transform_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    #Dataset / Loader
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=transform_train)
    val_ds = datasets.ImageFolder(VAL_DIR, transform=transform_val)

    num_classes = len(train_ds.classes)
    print(f"▶ Classes: {num_classes}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 모델
    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=num_classes,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=WARMUP_START_LR,
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
    early_stopping = EarlyStopping(patience=7, mode="min")

    best_val_acc = 0.0
    best_model_path = CHECKPOINT_DIR / "efficientnet_b4_68class_best.pth"

    # -------- 학습 루프 --------
    for epoch in range(1, EPOCHS + 1):
        print(f"\n Epoch {epoch}/{EPOCHS}")
        start_time = time.time()

        # 웝 업
        if epoch <= WARMUP_EPOCHS:
            lr = WARMUP_START_LR + (BASE_LR - WARMUP_START_LR) * (epoch / WARMUP_EPOCHS)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            warmup_flag = True
        else:
            scheduler.step()
            warmup_flag = False

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, DEVICE
        )

        epoch_time_min = (time.time() - start_time) / 60.0
        current_lr = optimizer.param_groups[0]["lr"]

        print(f" Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f" Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        print(f" LR: {current_lr:.6e}")
        print(f" Epoch Time: {epoch_time_min:.2f} min")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
            "epoch_time_min": epoch_time_min,
            "warmup": warmup_flag,
        })

        # Best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f" Best model updated! (Val Acc: {best_val_acc:.4f})")

        # EarlyStopping (after warm-up)
        if not warmup_flag:
            #마지막 2 epoch에서는 early stopping 무시
            if epoch <= EPOCHS - 2:
                if early_stopping(val_loss):
                    print(" Early stopping triggered.")
                    break

    # -------- Save logs --------
    df = pd.DataFrame(history)
    excel_path = CHECKPOINT_DIR / "training_log_efficientnet_b4_68class_final.xlsx"
    df.to_excel(excel_path, index=False)

    print("\n Training finished")
    print(f" Best model: {best_model_path}")
    print(f" Log saved: {excel_path}")


if __name__ == "__main__":
    main()