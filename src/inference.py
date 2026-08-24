"""Reusable single-image inference for the selected V2-A parking classifier."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError

from src.resnet18_transfer import build_resnet18
from src.train_baseline import select_device
from src.v2_training import build_v2_transform


MODEL_PATH_ENV = "PARKING_MODEL_PATH"
DEFAULT_CHECKPOINT = Path("models/v2a_balanced_resnet18.pt")
SELECTED_CANDIDATE = "v2a_balanced_resnet18"
SELECTED_ARCHITECTURE = "ResNet18"
SELECTED_CONFIG_SHA256 = "57fb8133760cc7eded11eb77e9c1ce5aa67d5379bfbfe33539b279e95a024957"
DECISION_THRESHOLD = 0.5
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
LABEL_NAMES = {0: "EMPTY", 1: "OCCUPIED"}


def resolve_checkpoint_path(
    configured_path: str | Path | None = None,
    repository_root: Path | None = None,
) -> Path:
    """Resolve an explicit/env/default checkpoint and fail with setup guidance."""

    root = (repository_root or Path(__file__).resolve().parents[1]).resolve()
    raw_path = configured_path or os.environ.get(MODEL_PATH_ENV)
    path = Path(raw_path).expanduser() if raw_path else root / DEFAULT_CHECKPOINT
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {path}. Set {MODEL_PATH_ENV} to the "
            "selected V2-A .pt checkpoint before starting the demo."
        )
    return path


def decode_uploaded_image(
    contents: bytes,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> Image.Image:
    """Validate and decode one supported upload into an RGB PIL image."""

    if not contents:
        raise ValueError("The uploaded file is empty.")
    if len(contents) > max_bytes:
        raise ValueError(f"The uploaded image exceeds the {max_bytes // (1024**2)} MB limit.")
    try:
        with Image.open(io.BytesIO(contents)) as source:
            image_format = source.format
            source.verify()
        if image_format not in ALLOWED_IMAGE_FORMATS:
            formats = ", ".join(sorted(ALLOWED_IMAGE_FORMATS))
            raise ValueError(f"Unsupported image format. Use one of: {formats}.")
        with Image.open(io.BytesIO(contents)) as source:
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("The uploaded file is not a valid decodable image.") from error


def prediction_from_probabilities(probabilities: torch.Tensor) -> dict[str, Any]:
    """Create the user-facing binary result from a length-two probability tensor."""

    if probabilities.ndim != 1 or probabilities.numel() != 2:
        raise ValueError("Expected probabilities for exactly two classes")
    class_index = int(probabilities[1].item() >= DECISION_THRESHOLD)
    return {
        "class_index": class_index,
        "label": LABEL_NAMES[class_index],
        "confidence": float(probabilities[class_index].item()),
        "probabilities": {
            "EMPTY": float(probabilities[0].item()),
            "OCCUPIED": float(probabilities[1].item()),
        },
    }


class ParkingOccupancyPredictor:
    """Load the selected checkpoint once and predict cropped parking-space images."""

    def __init__(self, checkpoint_path: Path, device: str = "auto") -> None:
        self.checkpoint_path = checkpoint_path.resolve()
        self.device = select_device(device)
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if "model_state_dict" not in checkpoint:
            raise ValueError("Checkpoint is missing model_state_dict")
        if checkpoint.get("candidate_id") != SELECTED_CANDIDATE:
            raise ValueError("Checkpoint is not the selected V2-A candidate")
        if checkpoint.get("architecture") != SELECTED_ARCHITECTURE:
            raise ValueError("Checkpoint is not the selected ResNet18 architecture")
        if checkpoint.get("experiment_config_sha256") != SELECTED_CONFIG_SHA256:
            raise ValueError("Checkpoint does not match the locked V2 experiment config")
        if checkpoint.get("fresh_final_images_opened") != 0:
            raise ValueError("Checkpoint metadata does not preserve the pre-selection boundary")
        self.input_size = 224
        self.model = build_resnet18(pretrained=False, dropout=0.20)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.requires_grad_(False).to(self.device).eval()
        self.transform = build_v2_transform(False, self.input_size)

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict[str, Any]:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probabilities = self.model(tensor).softmax(dim=1)[0].cpu()
        result = prediction_from_probabilities(probabilities)
        result.update({"device": str(self.device), "input_size": self.input_size})
        return result
