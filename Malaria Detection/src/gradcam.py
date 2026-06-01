"""
Grad-CAM heatmap generation for malaria cell explainability.
Highlights parasite regions in the cell image.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import cv2
import torchvision.transforms as T

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _preprocess(img: Image.Image) -> torch.Tensor:
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0).to(DEVICE)


class GradCAMWrapper:
    """Simple Grad-CAM implementation targeting the last conv layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, img_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        self.model.eval()
        output = self.model(img_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        pooled_grads = self.gradients.mean(dim=[0, 2, 3])
        activations = self.activations[0]
        for i, w in enumerate(pooled_grads):
            activations[i] *= w

        heatmap = activations.mean(dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap


def generate_gradcam_overlay(model, img: Image.Image, class_idx: int = None) -> Image.Image:
    """Generate Grad-CAM overlay on original image."""
    img_tensor = _preprocess(img)
    img_np = np.array(img.resize((IMG_SIZE, IMG_SIZE)))

    # Try to find the last conv/attention layer
    try:
        # EfficientNetV2 backbone
        target_layer = model.backbone.conv_head
    except AttributeError:
        try:
            # Fallback to last block
            target_layer = list(model.backbone.children())[-2]
        except Exception:
            return img

    cam = GradCAMWrapper(model, target_layer)
    heatmap = cam.generate(img_tensor, class_idx)

    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_coloured = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_coloured = cv2.cvtColor(heatmap_coloured, cv2.COLOR_BGR2RGB)

    overlay = 0.5 * img_np + 0.5 * heatmap_coloured
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)
