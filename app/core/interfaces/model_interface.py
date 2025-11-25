from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from app.core.entities.model import Model


class IModelRepository(ABC):
    @abstractmethod
    def get_by_id(self, model_id: UUID) -> Optional[Model]:
        ...

    @abstractmethod
    def upload_model(self, model: Model) -> Model:
        ...