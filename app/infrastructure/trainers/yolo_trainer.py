from ultralytics import YOLO

from app.core.interfaces.detector_trainer_interface import IDetectorTrainer


class YoloTrainer(IDetectorTrainer):
    def __init__(self, architecture_profile: str):
        self.architecture_profile = architecture_profile
        self.model = None

    def load_model(self, weights_path: str) -> None:
        self.model = YOLO(weights_path)

    def train(self, dataset_path: str):
        return self.model.train(
            data=dataset_path,
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