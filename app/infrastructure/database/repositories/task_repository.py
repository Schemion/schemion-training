from typing import Optional, List
from uuid import UUID

from sqlalchemy.orm import Session

from app.infrastructure.database.models import Task
from app.core.interfaces import ITaskRepository


class TaskRepository(ITaskRepository):
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, task_id: UUID) -> Optional[Task]:
        task = self.db.query(Task).filter(Task.id == task_id).first()
        return task

    def update(self, task: Task) -> Optional[Task]:
        task_model = self.db.query(Task).filter(Task.id == task.id).first()
        if not task_model:
            return None

        self.db.commit()
        self.db.refresh(task_model)

        return task_model # убрать надо варнинг