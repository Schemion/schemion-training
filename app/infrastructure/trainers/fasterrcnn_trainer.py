import os
import uuid

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2
import yaml

from app.core.interfaces.detector_trainer_interface import IDetectorTrainer
from app.infrastructure.trainers.utils.fasterrcnn_yolo_dataset import FasterRCNNYoloDataset


def _resolve_path(base_path: str, value: str) -> str:
    if os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(base_path, value))

def _images_path_to_labels(images_path: str) -> str:
    parts = images_path.split(os.sep)
    try:
        idx = parts.index("images")
        parts[idx] = "labels"
        return os.sep.join(parts)
    except ValueError:
        raise RuntimeError(f"'images' not found in path: {images_path}")

def _parse_class_names(data: dict) -> list[str]:
    names = data.get("names", [])
    if isinstance(names, dict):
        items = []
        for k, v in names.items():
            try:
                items.append((int(k), str(v)))
            except Exception:
                continue
        items.sort(key=lambda x: x[0])
        return [v for _, v in items]
    if isinstance(names, list):
        return [str(v) for v in names]
    return []


def _collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


class FasterRCNNTrainer(IDetectorTrainer):
    def __init__(self, architecture_profile: str):
        self.architecture_profile = architecture_profile
        self.model = None
        self._weights_path: str | None = None
        self._class_names: list[str] = []
        self._last_metrics: list[dict] = []

    def load_model(self, weights_path: str) -> None:
        self._weights_path = weights_path
        if self.model is None:
            return
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict):
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            elif "state_dict" in state:
                state = state["state_dict"]
        if isinstance(state, dict):
            state = {k: v for k, v in state.items() if not k.startswith("roi_heads.box_predictor.")}
            self.model.load_state_dict(state, strict=False)

    def _build_model(self, num_classes: int, image_size: int | None) -> torch.nn.Module:
        profile = (self.architecture_profile or "").lower()
        if "v2" in profile:
            return fasterrcnn_resnet50_fpn_v2( weights=None, min_size=image_size if image_size else 800, max_size=image_size if image_size else 1333,  num_classes=num_classes)
        return fasterrcnn_resnet50_fpn(weights=None, min_size=image_size if image_size else 800, max_size=image_size if image_size else 1333, num_classes=num_classes)

    def _load_weights_if_needed(self) -> None:
        if not self._weights_path or self.model is None:
            return
        if not os.path.isfile(self._weights_path):
            return
        state = torch.load(self._weights_path, map_location="cpu")
        if isinstance(state, dict):
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            elif "state_dict" in state:
                state = state["state_dict"]
        if isinstance(state, dict):
            state = {k: v for k, v in state.items() if not k.startswith("roi_heads.box_predictor.")}
            self.model.load_state_dict(state, strict=False)

    def train(self, dataset_path: str, image_size: int | None = None, epochs: int | None = None, name: str | None = None):
        yaml_path = os.path.abspath(dataset_path)
        yaml_dir = os.path.dirname(yaml_path)

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        base_path = data.get("path") or yaml_dir
        base_path = _resolve_path(yaml_dir, str(base_path))

        train_images = data.get("train")
        val_images = data.get("val")
        if not train_images or not val_images:
            raise RuntimeError("Dataset yaml must contain 'train' and 'val' paths")

        train_images = _resolve_path(base_path, str(train_images))
        val_images = _resolve_path(base_path, str(val_images))

        train_labels = _images_path_to_labels(train_images)
        val_labels = _images_path_to_labels(val_images)

        self._class_names = _parse_class_names(data)
        num_classes = len(self._class_names) + 1

        train_dataset = FasterRCNNYoloDataset(train_images, train_labels, self._class_names)
        val_dataset = FasterRCNNYoloDataset(val_images, val_labels, self._class_names)

        train_loader = DataLoader( train_dataset, batch_size=2,  shuffle=True, num_workers=0, collate_fn=_collate_fn,)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(num_classes=num_classes, image_size=image_size)
        self._load_weights_if_needed()
        self.model.to(device)

        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        num_epochs = epochs if epochs is not None else 10
        self._last_metrics = []

        for epoch in range(num_epochs):
            self.model.train()
            loss_totals: dict[str, float] = {}
            batch_count = 0

            for images, targets in train_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                batch_count += 1
                for key, value in loss_dict.items():
                    loss_totals[key] = loss_totals.get(key, 0.0) + float(value.detach().cpu())
                loss_totals["loss_total"] = loss_totals.get("loss_total", 0.0) + float(losses.detach().cpu())

            scheduler.step()

            if batch_count == 0:
                continue

            record = {"epoch": epoch}
            for key, total in loss_totals.items():
                record[key] = total / batch_count
            self._last_metrics.append(record)

        return {
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "epochs": num_epochs,
        }

    def export(self, output_path: str) -> None:
        if os.path.isfile(output_path):
            os.remove(output_path)
        os.makedirs(output_path, exist_ok=True)
        model_file_path = os.path.join(output_path, f"model_{uuid.uuid4().hex}.pth")
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        torch.save(self.model.state_dict(), model_file_path)

    def get_classes(self) -> list[str]:
        return list(self._class_names)

    def get_metrics(self) -> list[dict]:
        return list(self._last_metrics)
