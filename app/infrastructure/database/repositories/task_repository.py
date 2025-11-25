from typing import Optional, List
from uuid import UUID

from sqlalchemy.orm import Session

from app.infrastructure.database.models import Task as TaskModel
from app.core.entities.task import Task as TaskEntity
from app.core.interfaces import ITaskRepository
from app.infrastructure.mappers import OrmEntityMapper


class TaskRepository(ITaskRepository):
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, task_id: UUID) -> Optional[TaskEntity]:
        task = self.db.query(TaskModel).filter(task_id == TaskModel.id).first()
        return OrmEntityMapper.to_entity(task, TaskEntity)

    def update(self, task_entity: TaskEntity) -> Optional[TaskEntity]:
        task_model = self.db.query(TaskModel).filter(task_entity.id == TaskModel.id).first()
        if not task_model:
            return None

        for field in task_entity.__dict__:
            if hasattr(task_model, field):
                setattr(task_model, field, getattr(task_entity, field))

        self.db.commit()
        self.db.refresh(task_model)

        return OrmEntityMapper.to_entity(task_model, TaskEntity)