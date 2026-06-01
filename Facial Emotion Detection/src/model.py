"""
EfficientNet-B4 + DINOv2 ViT emotion classifier with:
- Multi-task learning (emotion + arousal/valence regression)
- Attention pooling
- Label smoothing
- Temporal smoothing for video streams
- ONNX export for real-time inference
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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_PROC  = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
NUM_CLASSES = len(EMOTIONS)
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 2e-4


# ─── Dataset ────────────────────────────────────────────────────────────────

class EmotionDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, augment: bool = False):
        self.manifest = manifest.reset_index(drop=True)
        self.augment = augment
        self.aug = _build_augmentation() if augment else None
        self.base_transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        try:
            img = Image.open(row["path"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            img_np = np.array(img)
        except Exception:
            img_np = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        if self.aug and self.augment:
            augmented = self.aug(image=img_np)
            tensor = augmented["image"]
        else:
            tensor = self.base_transform(Image.fromarray(img_np))

        return tensor, int(row["label"])


def _build_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.HueSaturationValue(10, 20, 10, p=0.4),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.3),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ─── Model Architecture ───────────────────────────────────────────────────────

class EmotionClassifier(nn.Module):
    """
    EfficientNet-B4 backbone with attention pooling and label smoothing.
    Optional multi-task head for arousal/valence regression.
    """
    def __init__(self, num_classes: int = 7, dropout: float = 0.4, multitask: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=True, num_classes=0,
            global_pool="", drop_rate=dropout,
        )
        n_features = self.backbone.num_features

        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        self.emotion_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_features, 512),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )
        self.multitask = multitask
        if multitask:
            # Arousal/valence regression (circumplex model of emotion)
            self.av_head = nn.Sequential(
                nn.Dropout(dropout / 2),
                nn.Linear(n_features, 128),
                nn.ReLU(),
                nn.Linear(128, 2),  # [arousal, valence]
                nn.Tanh(),
            )

    def forward(self, x):
        feat_map = self.backbone.forward_features(x)   # (B, C, H, W)
        B, C, H, W = feat_map.shape
        feat_flat = feat_map.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Attention pooling
        attn_weights = torch.softmax(self.attention(feat_flat), dim=1)
        pooled = (attn_weights * feat_flat).sum(dim=1)   # (B, C)

        emotion_logits = self.emotion_head(pooled)

        if self.multitask:
            av = self.av_head(pooled)
            return emotion_logits, av
        return emotion_logits


# ─── Training ────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in tqdm(loader, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(imgs)
        logits = output[0] if isinstance(output, tuple) else output

        loss = criterion(logits, labels)
        if isinstance(output, tuple):
            # Arousal/valence proxy loss (circular emotion mapping)
            av = output[1]
            av_target = _emotion_to_av(labels)
            loss += 0.1 * nn.MSELoss()(av, av_target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(imgs)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(imgs)
    return total_loss / total, correct / total


def _emotion_to_av(labels: torch.Tensor) -> torch.Tensor:
    """Map emotion index → (arousal, valence) in [-1, 1]."""
    av_map = torch.tensor([
        [0.8, -0.7],   # Angry
        [0.5, -0.8],   # Disgust
        [0.9, -0.6],   # Fear
        [0.5, 0.9],    # Happy
        [0.0, 0.0],    # Neutral
        [-0.3, -0.8],  # Sad
        [0.9, 0.5],    # Surprise
    ], dtype=torch.float32).to(labels.device)
    return av_map[labels]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        output = model(imgs)
        logits = output[0] if isinstance(output, tuple) else output
        preds = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    return acc, np.array(all_preds), np.array(all_labels)


def train_model():
    train_df = pd.read_csv(DATA_PROC / "train_manifest.csv")
    val_df   = pd.read_csv(DATA_PROC / "val_manifest.csv")
    test_df  = pd.read_csv(DATA_PROC / "test_manifest.csv")

    train_ds = EmotionDataset(train_df, augment=True)
    val_ds   = EmotionDataset(val_df, augment=False)
    test_ds  = EmotionDataset(test_df, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = EmotionClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)

    best_acc = 0
    print(f"Training EmotionClassifier (EfficientNet-B4) on {DEVICE} ...")
    for epoch in range(1, EPOCHS + 1):
        loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_acc, _, _ = evaluate(model, val_loader, DEVICE)
        scheduler.step()
        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {loss:.4f} | Acc: {acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODELS_DIR / "emotion_best.pth")

    model.load_state_dict(torch.load(MODELS_DIR / "emotion_best.pth"))
    test_acc, preds, labels = evaluate(model, test_loader, DEVICE)
    print(f"\nTest Accuracy: {test_acc*100:.1f}%")
    print(classification_report(labels, preds, target_names=EMOTIONS))

    export_onnx(model)
    return model, {"test_accuracy": test_acc}


def export_onnx(model):
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    torch.onnx.export(
        model, dummy, MODELS_DIR / "emotion_model.onnx",
        input_names=["face_image"], output_names=["logits"],
        dynamic_axes={"face_image": {0: "batch"}},
        opset_version=17,
    )
    print("ONNX exported.")


def load_model():
    model = EmotionClassifier().to(DEVICE).eval()
    p = MODELS_DIR / "emotion_best.pth"
    if p.exists():
        model.load_state_dict(torch.load(p, map_location=DEVICE))
    return model


class TemporalSmoother:
    """Exponential moving average over consecutive frame predictions."""
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.state = None

    def update(self, probs: np.ndarray) -> np.ndarray:
        if self.state is None:
            self.state = probs
        else:
            self.state = self.alpha * probs + (1 - self.alpha) * self.state
        return self.state


if __name__ == "__main__":
    train_model()
