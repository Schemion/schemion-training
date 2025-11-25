from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.entities.model import Model as ModelEntity
from app.core.interfaces.model_interface import IModelRepository
from app.infrastructure.database.models.model import Model
from app.infrastructure.mappers import OrmEntityMapper


class ModelRepository(IModelRepository):
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, model_id: UUID) -> Optional[ModelEntity]:
        model = self.db.query(Model).filter(model_id == Model.id).first()
        return OrmEntityMapper.to_entity(model, ModelEntity)

    def upload_model(self, entity: ModelEntity) -> ModelEntity:
        model = OrmEntityMapper.to_model(entity, Model)
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        return OrmEntityMapper.to_entity(model, ModelEntity)
