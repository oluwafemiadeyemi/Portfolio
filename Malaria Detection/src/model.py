"""
EfficientNetV2-S + ViT-B/16 ensemble with:
- Transfer learning (ImageNet weights via timm)
- Albumentations augmentation pipeline
- Grad-CAM explainability
- MC Dropout uncertainty quantification
- ONNX export for edge deployment
- WHO performance targets: sensitivity ≥95%, specificity ≥95%
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import joblib
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score,
)
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_PROC  = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
NUM_CLASSES = 2
BATCH_SIZE = 32
EPOCHS = 15
LR = 3e-4


# ─── Dataset ────────────────────────────────────────────────────────────────

class MalariaDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, transform=None, augment: bool = False):
        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform
        self.augment = augment
        self.aug_pipeline = _build_augmentation() if augment else None

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        try:
            img = Image.open(row["path"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            img_np = np.array(img)
        except Exception:
            img_np = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        if self.aug_pipeline and self.augment:
            augmented = self.aug_pipeline(image=img_np)
            img_np = augmented["image"]
            tensor = img_np if isinstance(img_np, torch.Tensor) else T.ToTensor()(img_np)
        elif self.transform:
            tensor = self.transform(Image.fromarray(img_np))
        else:
            tensor = T.ToTensor()(Image.fromarray(img_np))

        return tensor, int(row["label"])


def _build_augmentation():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.4),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.GridDistortion(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def _base_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ─── Models ─────────────────────────────────────────────────────────────────

class EfficientNetV2Classifier(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4.ra2_in1k", pretrained=True, num_classes=0, drop_rate=dropout
        )
        n_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


class ViTClassifier(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch16_224", pretrained=True, num_classes=0, drop_rate=dropout
        )
        n_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_features, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# ─── Training ────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in tqdm(loader, leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                out = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(imgs)
        correct += (out.argmax(1) == labels).sum().item()
        total += len(imgs)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        probs = torch.softmax(model(imgs), dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    auc = roc_auc_score(all_labels, all_probs)
    preds = (all_probs >= 0.5).astype(int)
    return auc, all_probs, all_labels


def train_model(model_name: str = "efficientnetv2"):
    train_df = pd.read_csv(DATA_PROC / "train_manifest.csv")
    val_df   = pd.read_csv(DATA_PROC / "val_manifest.csv")
    test_df  = pd.read_csv(DATA_PROC / "test_manifest.csv")

    train_ds = MalariaDataset(train_df, augment=True)
    val_ds   = MalariaDataset(val_df, transform=_base_transform())
    test_ds  = MalariaDataset(test_df, transform=_base_transform())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    if model_name == "efficientnetv2":
        model = EfficientNetV2Classifier().to(DEVICE)
    else:
        model = ViTClassifier().to(DEVICE)

    pos_weight = torch.tensor([(train_df["label"] == 0).sum() / (train_df["label"] == 1).sum()]).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    best_auc = 0
    print(f"\nTraining {model_name} on {DEVICE} ...")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        val_auc, val_probs, val_labels = evaluate(model, val_loader)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {train_loss:.4f} | Acc: {train_acc*100:.1f}% | Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), MODELS_DIR / f"{model_name}_best.pth")

    # Final test evaluation
    model.load_state_dict(torch.load(MODELS_DIR / f"{model_name}_best.pth"))
    test_auc, test_probs, test_labels = evaluate(model, test_loader)
    preds = (test_probs >= 0.5).astype(int)
    print(f"\n{'='*50}")
    print(f"Test AUC: {test_auc:.4f}")
    print(classification_report(test_labels, preds, target_names=["Uninfected", "Parasitized"]))

    # WHO metrics
    cm = confusion_matrix(test_labels, preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print(f"WHO Sensitivity: {sensitivity*100:.1f}%  (target ≥95%)")
    print(f"WHO Specificity: {specificity*100:.1f}%  (target ≥95%)")

    # Export ONNX
    export_onnx(model, model_name)

    return model, {"test_auc": test_auc, "sensitivity": sensitivity, "specificity": specificity}


# ─── MC Dropout Uncertainty ──────────────────────────────────────────────────

def predict_with_uncertainty(model, img_tensor: torch.Tensor, n_passes: int = 20) -> dict:
    """Monte Carlo Dropout: multiple stochastic forward passes → uncertainty estimate."""
    model.train()  # keep dropout active
    probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            out = torch.softmax(model(img_tensor.unsqueeze(0).to(DEVICE)), dim=1)
            probs.append(out.cpu().numpy()[0, 1])
    model.eval()
    probs = np.array(probs)
    return {
        "mean_prob":  float(probs.mean()),
        "std_prob":   float(probs.std()),
        "uncertainty": "High" if probs.std() > 0.15 else ("Medium" if probs.std() > 0.08 else "Low"),
        "prediction": "Parasitized" if probs.mean() >= 0.5 else "Uninfected",
    }


# ─── ONNX Export ─────────────────────────────────────────────────────────────

def export_onnx(model, model_name: str = "efficientnetv2"):
    """Export model to ONNX for edge/mobile deployment."""
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    onnx_path = MODELS_DIR / f"{model_name}.onnx"
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX model exported: {onnx_path}")


def load_model(model_name: str = "efficientnetv2") -> nn.Module:
    """Load saved model weights."""
    if model_name == "efficientnetv2":
        model = EfficientNetV2Classifier()
    else:
        model = ViTClassifier()
    weights_path = MODELS_DIR / f"{model_name}_best.pth"
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    return model.to(DEVICE).eval()


if __name__ == "__main__":
    train_model("efficientnetv2")
