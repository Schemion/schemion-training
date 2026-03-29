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

                loss = self._pick_float(row, ["val/box_loss", "train/box_loss", "val/loss", "train/loss"])
                acc = self._pick_float(row, ["metrics/mAP50-95(B)", "metrics/mAP50(B)", "metrics/precision(B)", "metrics/recall(B)"])

                if loss is not None or acc is not None:
                    metrics.append({"epoch": epoch, "loss": loss, "acc": acc})

        return metrics

    def train(self, dataset_yaml_path: str):
        yaml_parent = os.path.dirname(os.path.abspath(dataset_yaml_path))

        with _chdir(yaml_parent):
            result = self.model.train(
                data=os.path.abspath(dataset_yaml_path),
                epochs=10, # для теста, потом поставлю от 10 до 50
                imgsz=640,
                batch=4,
                name="yolo_custom",
                workers=0,
                device="cuda", # на время теста мой докер не использует гпу
                deterministic=True,
                exist_ok=True
            )
            save_dir = getattr(result, "save_dir", None)
            if save_dir:
                results_csv = os.path.join(str(save_dir), "results.csv")
            else:
                results_csv = os.path.join(os.getcwd(), "runs", "detect", "yolo_custom", "results.csv")
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
