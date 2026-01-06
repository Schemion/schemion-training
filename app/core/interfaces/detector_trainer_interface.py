from abc import ABC, abstractmethod
from typing import Any, Optional


class IDetectorTrainer(ABC):
    @abstractmethod
    def load_model(self, weights_path: str) -> None:
        ...

    @abstractmethod
    def train(self, dataset_path: str) -> Any:
        ...

    @abstractmethod
    def export(self, output_path: str) -> None:
        ...

    # было бэд, стало наверное гуд, классы будут из датасета браться, зачем они фабрике?
    @abstractmethod
    def get_classes(self) -> list[str]:
        ...