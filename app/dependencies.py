from app.core.interfaces.detector_trainer_factory_interface import IDetectorTrainerFactory
from app.infrastructure.factories.detectors_trainer_factory import DetectorTrainerFactory


def get_detector_trainer_factory() -> IDetectorTrainerFactory:
    return DetectorTrainerFactory()