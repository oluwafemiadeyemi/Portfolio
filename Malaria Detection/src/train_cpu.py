"""
CPU-friendly training: MobileNetV3-Small, 112x112, 5 epochs.
Completes in ~10 minutes on CPU. Saves to models/efficientnetv2_best.pth
so the existing API and dashboard pick it up automatically.
"""

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
from sklearn.metrics import classification_report, roc_auc_score

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_PROC  = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE    = torch.device("cpu")
IMG_SIZE  = 112
EPOCHS    = 5
BATCH     = 16


class CellDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, augment: bool = False):
        self.df = manifest.reset_index(drop=True)
        ops = [T.Resize((IMG_SIZE, IMG_SIZE))]
        if augment:
            ops += [T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                    T.ColorJitter(0.2, 0.2, 0.1)]
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


def build_model():
    # MobileNetV3-Small: fast on CPU, ImageNet pretrained
    model = timm.create_model("mobilenetv3_small_100", pretrained=True,
                               num_classes=2, drop_rate=0.2)
    return model.to(DEVICE)


def train():
    train_df = pd.read_csv(DATA_PROC / "train_manifest.csv")
    val_df   = pd.read_csv(DATA_PROC / "val_manifest.csv")
    test_df  = pd.read_csv(DATA_PROC / "test_manifest.csv")

    train_loader = DataLoader(CellDataset(train_df, augment=True),
                              batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(CellDataset(val_df),  batch_size=BATCH)
    test_loader  = DataLoader(CellDataset(test_df), batch_size=BATCH)

    model     = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_auc, best_state = 0, None
    print(f"Training MobileNetV3-Small on {DEVICE} | {len(train_df)} train images")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(imgs)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += len(imgs)
        scheduler.step()

        # Validation
        model.eval()
        probs, truths = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                p = torch.softmax(model(imgs), 1)[:, 1]
                probs.extend(p.numpy()); truths.extend(labels.numpy())
        auc = roc_auc_score(truths, probs)
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {total_loss/total:.4f} | "
              f"Acc: {correct/total*100:.1f}% | Val AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc   = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Final test evaluation
    model.load_state_dict(best_state)
    model.eval()
    probs, truths = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            p = torch.softmax(model(imgs), 1)[:, 1]
            probs.extend(p.numpy()); truths.extend(labels.numpy())

    preds = [1 if p >= 0.5 else 0 for p in probs]
    auc   = roc_auc_score(truths, probs)
    print(f"\nTest AUC: {auc:.4f}")
    print(classification_report(truths, preds, target_names=["Uninfected", "Parasitized"]))

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(truths, preds)
    tn, fp, fn, tp = cm.ravel()
    print(f"Sensitivity: {tp/(tp+fn)*100:.1f}% | Specificity: {tn/(tn+fp)*100:.1f}%")

    # Save under the name the API expects
    torch.save(best_state, MODELS_DIR / "efficientnetv2_best.pth")

    # Also save model class info so api.py can load it
    import json
    (MODELS_DIR / "model_info.json").write_text(
        json.dumps({"arch": "mobilenetv3_small_100", "img_size": IMG_SIZE})
    )
    print(f"\nModel saved to {MODELS_DIR / 'efficientnetv2_best.pth'}")
    return model


if __name__ == "__main__":
    train()
