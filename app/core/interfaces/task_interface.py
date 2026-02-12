from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from app.infrastructure.database.models import Task


class ITaskRepository(ABC):
    @abstractmethod
    def get_by_id(self, task_id: UUID) -> Optional[Task]:
        ...

    @abstractmethod
    def update(self, task: Task) -> Task:
        ...
