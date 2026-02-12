from sqlalchemy import Enum, UUID, String, Text, DateTime, func
from sqlalchemy.orm import mapped_column, Mapped
from datetime import datetime
from app.core.enums import TaskStatus
from app.infrastructure.database.models.base import Base


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    task_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(Enum(TaskStatus, name="task_status"), nullable=False, index=True)
    model_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=True)
    dataset_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=True)
    input_path: Mapped[str] = mapped_column(String(512), nullable=True)
    output_path: Mapped[str] = mapped_column(String(512), nullable=True)
    error_msg: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())