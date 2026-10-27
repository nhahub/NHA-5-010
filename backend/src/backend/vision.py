"""
MediVision AI - Chest X-ray Vision Inference & Grad-CAM Service
DenseNet-121 Multi-Label Thoracic Pathology Diagnostic Engine
"""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torchvision import models, transforms

logger = logging.getLogger("medivision.vision")

# Target image dimensions & normalization constants from training notebook
IMAGE_SIZE = 224
RESIZE_SHORT_SIDE = 256
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Preprocessing transforms (matches training pipeline exactly)
eval_transform = transforms.Compose([
    transforms.Resize(RESIZE_SHORT_SIDE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])

display_transform = transforms.Compose([
    transforms.Resize(RESIZE_SHORT_SIDE),
    transforms.CenterCrop(IMAGE_SIZE),
])


def build_densenet121_classifier(num_classes: int = 5) -> nn.Module:
    """Build DenseNet-121 model with customized multi-label classification head."""
    backbone = models.densenet121(weights=None)
    in_features = backbone.classifier.in_features
    backbone.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )
    return backbone


class GradCAM:
    """
    Grad-CAM implementation hook-registered on target convolutional layer.
    Extracts gradient-weighted activation maps to localize pathologies.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.fwd_handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._tensor_gradient_hook)

    def _tensor_gradient_hook(self, gradient):
        self.gradients = gradient

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Generate normalized Grad-CAM heatmap for a target class index.
        """
        self.model.zero_grad(set_to_none=True)
        tensor = input_tensor.clone()
        logits = self.model(tensor)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        weighted = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(weighted)

        # Fallback to unclipped attribution if ReLU zeros entire activation map
        if float(cam.detach().max()) <= 1e-8:
            cam = weighted - weighted.min()

        cam = torch.nn.functional.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam_np = cam[0, 0].detach().cpu().numpy()

        denom = float(cam_np.max() - cam_np.min())
        if denom > 1e-8:
            cam_norm = (cam_np - cam_np.min()) / denom
        else:
            cam_norm = np.zeros_like(cam_np)

        return cam_norm

    def close(self):
        if self.fwd_handle:
            self.fwd_handle.remove()


class ChestXrayVisionService:
    """
    Production-grade Chest X-ray vision diagnostic service:
    Loads DenseNet-121 weights, executes inference across 5 thoracic conditions,
    and generates visual Grad-CAM interpretability overlays.
    """

    def __init__(self, model_dir: Optional[Path] = None):
        if model_dir is None:
            # Resolves from backend/src/backend -> models/chest_xray
            self.model_dir = Path(__file__).resolve().parents[3] / "models" / "chest_xray"
        else:
            self.model_dir = Path(model_dir)

        self.weights_path = self.model_dir / "chest_xray_densenet121.pth"
        self.metadata_path = self.model_dir / "chest_xray_densenet121_metadata.json"
        self.metrics_path = self.model_dir / "chest_xray_densenet121_metrics.json"

        self.model: Optional[nn.Module] = None
        self.gradcam: Optional[GradCAM] = None
        self.metadata: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.target_conditions: List[str] = [
            "Cardiomegaly",
            "Pneumonia",
            "Effusion",
            "Infiltration",
            "Atelectasis",
        ]
        self.thresholds: Dict[str, float] = {
            "Cardiomegaly": 0.939453125,
            "Pneumonia": 0.7666015625,
            "Effusion": 0.78955078125,
            "Infiltration": 0.54931640625,
            "Atelectasis": 0.64990234375,
        }
        self.per_class_metrics: Dict[str, Dict[str, Any]] = {}
        self.is_ready: bool = False

    def load(self) -> bool:
        """Load model weights, metadata configuration, and test metrics."""
        try:
            # 1. Load metadata
            if self.metadata_path.is_file():
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                self.target_conditions = self.metadata.get("class_names", self.target_conditions)
                self.thresholds = self.metadata.get("thresholds_validation_f1", self.thresholds)
                logger.info("Loaded vision metadata for %s classes", len(self.target_conditions))

            # 2. Load metrics
            if self.metrics_path.is_file():
                with open(self.metrics_path, "r", encoding="utf-8") as f:
                    self.metrics = json.load(f)
                # Map per-class metrics by condition name
                val_metrics = self.metrics.get("per_class_test_metrics_validation_thresholds", [])
                for item in val_metrics:
                    cond = item.get("condition")
                    if cond:
                        self.per_class_metrics[cond] = item
                base_metrics = self.metrics.get("per_class_test_metrics_threshold_0_5", [])
                for item in base_metrics:
                    cond = item.get("condition")
                    if cond and cond in self.per_class_metrics:
                        self.per_class_metrics[cond]["ROC_AUC"] = item.get("ROC_AUC")
                        self.per_class_metrics[cond]["prevalence_pct"] = item.get("prevalence_pct")

            # 3. Build & load model weights
            if not self.weights_path.is_file():
                logger.error("Model weights file not found: %s", self.weights_path)
                return False

            backbone = build_densenet121_classifier(num_classes=len(self.target_conditions))
            state_dict = torch.load(self.weights_path, map_location="cpu")
            backbone.load_state_dict(state_dict, strict=True)
            backbone.eval()
            self.model = backbone

            # 4. Attach Grad-CAM on the final dense block feature map
            target_layer = self.model.features.denseblock4
            self.gradcam = GradCAM(self.model, target_layer)

            self.is_ready = True
            logger.info("MediVision DenseNet-121 Vision Engine initialized successfully!")
            return True
        except Exception as e:
            logger.exception("Failed to load Chest X-ray vision model: %s", e)
            self.is_ready = False
            return False

    def _encode_pil_to_base64(self, img: Image.Image, format: str = "PNG") -> str:
        """Convert a PIL Image to a Base64 data URL string."""
        buffer = io.BytesIO()
        img.save(buffer, format=format)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        mime = "image/png" if format.upper() == "PNG" else "image/jpeg"
        return f"data:{mime};base64,{encoded}"

    def _create_visualizations(
        self,
        cropped_pil: Image.Image,
        cams: Dict[str, np.ndarray],
    ) -> Tuple[str, Dict[str, str], Dict[str, str]]:
        """
        Create Base64 visualizations:
        - original_base64
        - overlays: condition -> base64
        - heatmaps: condition -> base64
        """
        # Convert cropped image to float RGB array [0, 1]
        rgb_np = np.array(cropped_pil, dtype=np.float32) / 255.0

        original_b64 = self._encode_pil_to_base64(cropped_pil, format="PNG")
        overlays_b64: Dict[str, str] = {}
        heatmaps_b64: Dict[str, str] = {}

        cmap = plt.get_cmap("jet")
        alpha = 0.42

        for cond, cam in cams.items():
            # Get RGBA from colormap
            heat_rgba = cmap(cam)
            heat_rgb = heat_rgba[:, :, :3]

            # Blend with original
            overlay = (1.0 - alpha) * rgb_np + alpha * heat_rgb
            overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
            heat_uint8 = np.clip(heat_rgb * 255.0, 0, 255).astype(np.uint8)

            overlay_img = Image.fromarray(overlay)
            heat_img = Image.fromarray(heat_uint8)

            overlays_b64[cond] = self._encode_pil_to_base64(overlay_img, format="PNG")
            heatmaps_b64[cond] = self._encode_pil_to_base64(heat_img, format="PNG")

        return original_b64, overlays_b64, heatmaps_b64

    def predict(
        self,
        image_bytes: bytes,
        selected_condition: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute full Chest X-ray prediction pipeline:
        1. Decode and preprocess input image.
        2. Run PyTorch forward pass to obtain condition probabilities.
        3. Compare against validation F1 thresholds for binary detection.
        4. Generate Grad-CAM heatmaps & visual overlays across conditions.
        5. Return comprehensive diagnostic analysis.
        """
        if not self.is_ready or self.model is None or self.gradcam is None:
            raise RuntimeError("Vision model is not loaded or initialized on the server.")

        # 1. Open and validate image
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            # Handle orientation if EXIF tags exist
            pil_img = ImageOps.exif_transpose(pil_img)
            # Convert grayscale / RGBA / palette to standard 3-channel RGB
            pil_img = pil_img.convert("RGB")
        except Exception as exc:
            raise ValueError(f"Invalid or corrupted image format: {exc}") from exc

        # 2. Preprocess
        input_tensor = eval_transform(pil_img).unsqueeze(0)
        cropped_pil = display_transform(pil_img)

        # 3. Model Forward Pass
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy().tolist()

        # 4. Compute Grad-CAM heatmaps for all conditions
        cams: Dict[str, np.ndarray] = {}
        for idx, cond in enumerate(self.target_conditions):
            cams[cond] = self.gradcam.generate(input_tensor, idx)

        # 5. Format findings & detection decisions
        findings: List[Dict[str, Any]] = []
        detected_conditions: List[str] = []

        for idx, cond in enumerate(self.target_conditions):
            prob = float(probs[idx])
            threshold = float(self.thresholds.get(cond, 0.5))
            is_detected = prob >= threshold
            if is_detected:
                detected_conditions.append(cond)

            # Performance telemetry
            perf = self.per_class_metrics.get(cond, {})
            roc_auc = perf.get("ROC_AUC")
            sensitivity = perf.get("sensitivity")
            specificity = perf.get("specificity")

            findings.append({
                "condition": cond,
                "probability": round(prob, 4),
                "probability_pct": f"{prob * 100:.1f}%",
                "confidence_pct": round(prob * 100, 1),
                "threshold": round(threshold, 4),
                "threshold_pct": f"{threshold * 100:.1f}%",
                "detected": is_detected,
                "status": "DETECTED" if is_detected else "NOT DETECTED",
                "severity": "high" if is_detected else ("elevated" if prob > 0.4 else "normal"),
                "roc_auc": round(float(roc_auc), 3) if roc_auc is not None else None,
                "sensitivity": round(float(sensitivity), 3) if sensitivity is not None else None,
                "specificity": round(float(specificity), 3) if specificity is not None else None,
            })

        # Determine primary condition for display
        if selected_condition and selected_condition in self.target_conditions:
            primary_condition = selected_condition
        elif detected_conditions:
            # Pick detected condition with highest margin above threshold
            primary_condition = max(
                detected_conditions,
                key=lambda c: probs[self.target_conditions.index(c)] - self.thresholds.get(c, 0.5),
            )
        else:
            # Pick condition with highest probability
            max_idx = int(np.argmax(probs))
            primary_condition = self.target_conditions[max_idx]

        # 6. Generate visualizations
        orig_b64, overlays_b64, heatmaps_b64 = self._create_visualizations(cropped_pil, cams)

        # 7. Build clinical diagnostic summary
        if detected_conditions:
            summary = (
                f"Positive indication for {', '.join(detected_conditions)} "
                f"exceeding validated F1 decision thresholds. "
                f"Remaining conditions ({', '.join([c for c in self.target_conditions if c not in detected_conditions])}) "
                f"fall below detection cutoffs."
            )
        else:
            summary = (
                "No target thoracic conditions detected above validation F1 decision boundaries. "
                "All 5 evaluated conditions are within normal limits."
            )

        return {
            "findings": findings,
            "detected_conditions": detected_conditions,
            "has_detected_pathology": len(detected_conditions) > 0,
            "primary_condition": primary_condition,
            "summary": summary,
            "visualizations": {
                "original": orig_b64,
                "overlay": overlays_b64.get(primary_condition, ""),
                "heatmap": heatmaps_b64.get(primary_condition, ""),
                "overlays_by_condition": overlays_b64,
                "heatmaps_by_condition": heatmaps_b64,
            },
            "metadata": {
                "model_name": self.metadata.get("model_name", "DenseNet-121 Chest X-ray Classifier"),
                "architecture": "DenseNet-121 (Convolutional Feature Extractor)",
                "target_layer": "features.denseblock4 (Grad-CAM Localization)",
                "test_macro_roc_auc": self.metadata.get("test_macro_roc_auc", 0.807),
                "clinical_status": self.metadata.get(
                    "clinical_status",
                    "Research and development decision support only. Not clinically certified for autonomous medical diagnosis.",
                ),
            },
        }

    def get_status(self) -> Dict[str, Any]:
        """Return diagnostic engine status, architecture details, and per-class thresholds."""
        return {
            "is_ready": self.is_ready,
            "model_name": self.metadata.get("model_name", "DenseNet-121 Chest X-ray"),
            "architecture": "DenseNet-121",
            "weights_path": str(self.weights_path),
            "weights_present": self.weights_path.is_file(),
            "target_conditions": self.target_conditions,
            "thresholds": self.thresholds,
            "test_macro_roc_auc": self.metadata.get("test_macro_roc_auc", 0.807),
            "per_class_metrics": self.per_class_metrics,
            "disclaimer": "MediVision AI Chest X-ray analysis is an AI decision-support research tool.",
        }


# Singleton service instance
vision_service = ChestXrayVisionService()
