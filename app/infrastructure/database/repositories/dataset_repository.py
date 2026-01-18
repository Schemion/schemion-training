from uuid import UUID
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.interfaces import IDatasetRepository
from app.infrastructure.database import models
from app.infrastructure.mappers import OrmEntityMapper
from app.core.entities.dataset import Dataset as EntityDataset


class DatasetRepository(IDatasetRepository):
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, dataset_id: UUID, user_id: Optional[UUID] = None) -> Optional[EntityDataset]:
        with self.SessionLocal() as db:
            query = select(models.Dataset).where(models.Dataset.id == dataset_id)
            if user_id:
                query = query.where((models.Dataset.user_id == user_id) | (models.Dataset.user_id.is_(None)))
            result = db.execute(query)
            db_dataset = result.scalar_one_or_none()
            return OrmEntityMapper.to_entity(db_dataset, EntityDataset) if db_dataset else None