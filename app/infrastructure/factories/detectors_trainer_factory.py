from app.core.interfaces.detector_trainer_factory_interface import IDetectorTrainerFactory
from app.core.interfaces.detector_trainer_interface import IDetectorTrainer
from app.infrastructure.models_config import FASTERRCNN_ALIASES


YoloTrainer = None
FasterRCNNTrainer = None


class DetectorTrainerFactory(IDetectorTrainerFactory):

    def create(self, architecture: str, architecture_profile: str) -> IDetectorTrainer:
        architecture = architecture.lower()

        if architecture == "yolo":
            trainer_cls = self._yolo_trainer_class()
            return trainer_cls(architecture_profile=architecture_profile)
        if architecture in FASTERRCNN_ALIASES:
            trainer_cls = self._fasterrcnn_trainer_class()
            return trainer_cls(architecture_profile=architecture_profile)

        raise ValueError(f"Unsupported architecture: {architecture}")

    @staticmethod
    def _yolo_trainer_class():
        if YoloTrainer is not None:
            return YoloTrainer
        from app.infrastructure.trainers.yolo_trainer import YoloTrainer as ImportedYoloTrainer

        return ImportedYoloTrainer

    @staticmethod
    def _fasterrcnn_trainer_class():
        if FasterRCNNTrainer is not None:
            return FasterRCNNTrainer
        from app.infrastructure.trainers.fasterrcnn_trainer import FasterRCNNTrainer as ImportedFasterRCNNTrainer

        return ImportedFasterRCNNTrainer
