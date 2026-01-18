from abc import ABC, abstractmethod

class IDatasetLoader(ABC):
    @abstractmethod
    def load(self, object_name: str) -> str:
        ...

    @abstractmethod
    def delete(self, path: str) -> None:
        ...