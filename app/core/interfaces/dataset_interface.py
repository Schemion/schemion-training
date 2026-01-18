from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID


from app.core import entities
from sqlalchemy.orm import Session


class IDatasetRepository(ABC):
    @abstractmethod
    def get_by_id(self, db: Session, dataset_id: UUID, user_id: Optional[UUID] = None) -> Optional[entities.Dataset]:
        ...
