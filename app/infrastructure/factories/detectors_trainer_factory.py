from app.core.interfaces.detector_trainer_factory_interface import IDetectorTrainerFactory
from app.core.interfaces.detector_trainer_interface import IDetectorTrainer
from app.infrastructure.trainers.yolo_trainer import YoloTrainer


class DetectorTrainerFactory(IDetectorTrainerFactory):

    def create(self, architecture: str, architecture_profile: str) -> IDetectorTrainer:
        architecture = architecture.lower()

        if architecture == "yolo":
            return YoloTrainer(architecture_profile=architecture_profile)

        raise ValueError(f"Unsupported architecture: {architecture}")