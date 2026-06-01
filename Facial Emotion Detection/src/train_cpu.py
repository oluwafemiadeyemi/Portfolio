"""
CPU-friendly FER2013 emotion training.
MobileNetV3-Small, 112x112, 7 epochs — ~30-40 min on CPU.
Saves to models/emotion_best.pth (picked up by API + dashboard automatically).
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm
from pathlib import Path
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_PROC  = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE   = torch.device("cpu")
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
IMG_SIZE = 112
EPOCHS   = 7
BATCH    = 32


class FERDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, augment: bool = False):
        self.df = manifest.reset_index(drop=True)
        ops = [T.Resize((IMG_SIZE, IMG_SIZE)), T.Grayscale(num_output_channels=3)]
        if augment:
            ops += [T.RandomHorizontalFlip(),
                    T.RandomRotation(15),
                    T.ColorJitter(brightness=0.3, contrast=0.3)]
        ops += [T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        self.transform = T.Compose(ops)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row["path"]).convert("RGB")
        except Exception:
            img = Image.fromarray(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
        return self.transform(img), int(row["label"])


def train():
    train_df = pd.read_csv(DATA_PROC / "train_manifest.csv")
    val_df   = pd.read_csv(DATA_PROC / "val_manifest.csv")
    test_df  = pd.read_csv(DATA_PROC / "test_manifest.csv")

    # Handle class imbalance — FER2013 has very few Disgust samples
    class_counts = train_df["label"].value_counts().sort_index()
    weights = 1.0 / class_counts.values.astype(float)
    class_weights = torch.tensor(weights / weights.sum() * len(EMOTIONS),
                                 dtype=torch.float32)

    train_loader = DataLoader(FERDataset(train_df, augment=True),
                              batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader   = DataLoader(FERDataset(val_df),  batch_size=BATCH, num_workers=0)
    test_loader  = DataLoader(FERDataset(test_df), batch_size=BATCH, num_workers=0)

    model     = timm.create_model("mobilenetv3_small_100", pretrained=True,
                                   num_classes=len(EMOTIONS), drop_rate=0.3)
    model     = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3,
        steps_per_epoch=len(train_loader), epochs=EPOCHS
    )

    best_acc, best_state = 0, None
    print(f"Training MobileNetV3-Small on FER2013 | {len(train_df):,} train images | device={DEVICE}")
    print(f"Class distribution: {class_counts.to_dict()}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(imgs)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += len(imgs)

        # Validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                val_preds.extend(model(imgs).argmax(1).numpy())
                val_true.extend(labels.numpy())
        val_acc = accuracy_score(val_true, val_preds)
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {total_loss/total:.4f} | "
              f"Train Acc: {correct/total*100:.1f}% | Val Acc: {val_acc*100:.1f}%")

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Final test evaluation
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            all_preds.extend(model(imgs).argmax(1).numpy())
            all_true.extend(labels.numpy())

    test_acc = accuracy_score(all_true, all_preds)
    print(f"\nTest Accuracy: {test_acc*100:.1f}%")
    print(classification_report(all_true, all_preds, target_names=EMOTIONS))

    # Save model + metadata
    torch.save(best_state, MODELS_DIR / "emotion_best.pth")
    (MODELS_DIR / "model_info.json").write_text(
        json.dumps({"arch": "mobilenetv3_small_100", "img_size": IMG_SIZE,
                    "num_classes": len(EMOTIONS), "dataset": "FER2013",
                    "n_train": len(train_df), "test_accuracy": round(test_acc, 4)})
    )
    print(f"Model saved → {MODELS_DIR / 'emotion_best.pth'}")
    return model


if __name__ == "__main__":
    train()
