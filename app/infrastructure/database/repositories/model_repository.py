from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.interfaces.model_interface import IModelRepository
from app.infrastructure.database.models.model import Model


class ModelRepository(IModelRepository):
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, model_id: UUID) -> Optional[Model]:
        model = self.db.query(Model).filter(Model.id == model_id).first()
        return model

    def upload_model(self, model: Model) -> Model:
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        return model
