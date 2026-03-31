import os
import uuid
import csv
from contextlib import contextmanager

from ultralytics import YOLO

from app.core.interfaces.detector_trainer_interface import IDetectorTrainer


@contextmanager
def _chdir(path: str):
    prev = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(prev)


class YoloTrainer(IDetectorTrainer):
    def __init__(self, architecture_profile: str):
        self.architecture_profile = architecture_profile
        self.model = None
        self._last_metrics: list[dict] = []

    def load_model(self, weights_path: str) -> None:
        self.model = YOLO(weights_path)

    def _pick_float(self, row: dict, keys: list[str]) -> float | None:
        for key in keys:
            value = row.get(key)
            if value not in (None, ""):
                try:
                    return float(value)
                except ValueError:
                    pass
        return None

    def _parse_metrics_csv(self, csv_path: str) -> list[dict]:
        if not os.path.isfile(csv_path):
            return []

        metrics = []
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    epoch = int(float(row["epoch"]))
                except (KeyError, TypeError, ValueError):
                    continue

                record = {"epoch": epoch}
                for key, value in row.items():
                    if key == "epoch":
                        continue
                    if value in (None, ""):
                        continue
                    try:
                        record[key] = float(value)
                    except ValueError:
                        continue

                if len(record) > 1:
                    metrics.append(record)

        return metrics

    def train(self, dataset_yaml_path: str, image_size: int | None = None, epochs: int | None = None, name: str | None = None):
        yaml_parent = os.path.dirname(os.path.abspath(dataset_yaml_path))

        with _chdir(yaml_parent):
            epochs_val = epochs if epochs is not None else 10
            image_size_val = image_size if image_size is not None else 640
            name_val = name if name else "yolo_custom"
            result = self.model.train(
                data=os.path.abspath(dataset_yaml_path),
                epochs=epochs_val,
                imgsz=image_size_val,
                batch=4,
                name=name_val,
                workers=4,
                device="cuda",
                deterministic=True,
                exist_ok=True
            )
            save_dir = getattr(result, "save_dir", None)
            if save_dir:
                results_csv = os.path.join(str(save_dir), "results.csv")
            else:
                results_csv = os.path.join(os.getcwd(), "runs", "detect", name_val, "results.csv")
            self._last_metrics = self._parse_metrics_csv(results_csv)
            return result

    def export(self, output_path: str) -> None:
        if os.path.isfile(output_path):
            os.remove(output_path)
        os.makedirs(output_path, exist_ok=True)
        model_file_path = os.path.join(output_path, f"model_{uuid.uuid4().hex}.pt")
        self.model.save(model_file_path)

    # йоло классы в таком виде не нужны
    def get_classes(self) -> list[str]:
        pass

    def get_metrics(self) -> list[dict]:
        return list(self._last_metrics)

