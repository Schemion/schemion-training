from abc import ABC, abstractmethod

from app.core.interfaces.detector_trainer_interface import IDetectorTrainer


class IDetectorTrainerFactory(ABC):
    @abstractmethod
    def create(self, architecture: str, architecture_profile: str) -> IDetectorTrainer:
        ...