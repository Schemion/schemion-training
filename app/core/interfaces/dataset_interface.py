from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from app.infrastructure.database.models import Dataset


class IDatasetRepository(ABC):
    @abstractmethod
    def get_by_id(self,dataset_id: UUID, user_id: Optional[UUID] = None,) -> Optional[Dataset]:
        ...
