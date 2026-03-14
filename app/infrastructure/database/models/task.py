import uuid

from sqlalchemy import Column, Enum, UUID, ForeignKey, String, Text, DateTime, func, text
from sqlalchemy.orm import mapped_column, Mapped
from datetime import datetime
from app.core.enums import TaskStatus
from app.infrastructure.database.models.base import Base


class Task(Base):
    __tablename__ = "tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"), default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    task_type = Column(String(50), nullable=False)
    status = Column(Enum(TaskStatus, name="task_status"), nullable=False, default=TaskStatus.queued, index=True)
    model_id = Column(UUID(as_uuid=True), nullable=True)
    dataset_id = Column(UUID(as_uuid=True), nullable=True)
    input_path = Column(String(512), nullable=True)
    output_path = Column(String(512), nullable=True)
    error_msg = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())