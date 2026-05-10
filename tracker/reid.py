"""
ReID feature extractor.

Uses a MobileNetV3-Large pretrained on ImageNet (from torchvision).
The classification head is stripped; features come from the global
average-pooled activations — a 960-dim L2-normalised vector.

This is not a person-specific ReID model, but ImageNet features
contain strong shape/colour/texture priors that work well in practice.
"""

import numpy as np
import cv2

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T


# Standard ImageNet normalisation
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

# Input size expected by MobileNetV3
_INPUT_SIZE = (128, 64)   # (height, width) — standard ReID crop size


class ReIDExtractor:
    """Extract L2-normalised appearance embeddings from image crops."""

    def __init__(self, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load backbone; remove classifier so we get raw feature maps
        backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        # features → AdaptiveAvgPool → flatten  (960-dim)
        self.model = nn.Sequential(
            backbone.features,
            backbone.avgpool,
            nn.Flatten(),
        ).to(self.device).eval()

        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(_INPUT_SIZE),
            T.Normalize(mean=_MEAN, std=_STD),
        ])

    @torch.no_grad()
    def extract(self, image: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        """
        Extract one embedding per bounding box.

        Args:
            image:  BGR uint8 image (H, W, 3) — as returned by cv2.imread
            bboxes: (N, 4) array of [x, y, w, h] bounding boxes (pixel coords)

        Returns:
            (N, D) float32 array of L2-normalised embeddings.
            Returns empty (0, D) array when bboxes is empty.
        """
        if len(bboxes) == 0:
            return np.empty((0, 960), dtype=np.float32)

        h_img, w_img = image.shape[:2]
        crops = []

        for x, y, w, h in bboxes:
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(w_img, int(x + w))
            y2 = min(h_img, int(y + h))

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                # Degenerate box — use blank crop
                crop = np.zeros((*_INPUT_SIZE, 3), dtype=np.uint8)

            # cv2 gives BGR; torchvision expects RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(self.transform(crop_rgb))

        batch = torch.stack(crops).to(self.device)
        features = self.model(batch).cpu().numpy()                  # (N, D)

        # L2 normalise
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / (norms + 1e-6)
        return features.astype(np.float32)
