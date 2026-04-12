import os

import torch
from torch.utils.data import Dataset


class FasterRCNNYoloDataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str, class_names: list[str]):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.class_names = class_names
        self.image_paths = self._collect_images()

    def _collect_images(self) -> list[str]:
        if not os.path.isdir(self.images_dir):
            return []
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        paths = []
        for name in os.listdir(self.images_dir):
            ext = os.path.splitext(name)[1].lower()
            if ext in exts:
                paths.append(os.path.join(self.images_dir, name))
        return sorted(paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_labels(self, label_path: str, img_w: int, img_h: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not os.path.isfile(label_path):
            empty_boxes = torch.zeros((0, 4), dtype=torch.float32)
            empty_labels = torch.zeros((0,), dtype=torch.int64)
            return empty_boxes, empty_labels

        boxes = []
        labels = []
        with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    class_id = int(float(parts[0]))
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except ValueError:
                    continue

                if class_id < 0 or class_id >= len(self.class_names):
                    continue

                x1 = (xc - w / 2.0) * img_w
                y1 = (yc - h / 2.0) * img_h
                x2 = (xc + w / 2.0) * img_w
                y2 = (yc + h / 2.0) * img_h

                x1 = max(0.0, min(x1, img_w - 1.0))
                y1 = max(0.0, min(y1, img_h - 1.0))
                x2 = max(0.0, min(x2, img_w - 1.0))
                y2 = max(0.0, min(y2, img_h - 1.0))

                if x2 <= x1 or y2 <= y1:
                    continue

                boxes.append([x1, y1, x2, y2])
                labels.append(class_id + 1)