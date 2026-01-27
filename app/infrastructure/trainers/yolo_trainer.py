import os
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

    def load_model(self, weights_path: str) -> None:
        self.model = YOLO(weights_path)

    def train(self, dataset_yaml_path: str):
        yaml_parent = os.path.dirname(os.path.abspath(dataset_yaml_path))

        with _chdir(yaml_parent):
            return self.model.train(
                data=os.path.abspath(dataset_yaml_path),
                epochs=50,
                imgsz=640,
                batch=16,
                name="yolo_custom",
                exist_ok=True
            )

    def export(self, output_path: str) -> None:
        self.model.export(format="pt", file=output_path)

    # йоло классы в таком виде не нужны
    def get_classes(self) -> list[str]:
        pass
