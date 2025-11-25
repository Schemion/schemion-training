import uuid

from sqlalchemy import UUID, Column, String, Text, DateTime, func

from app.infrastructure.database.models.base import Base


class Task(Base):
    __tablename__ = "tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    task_type = Column(String(50), nullable=False)
    model_id = Column(UUID(as_uuid=True), nullable=True)
    dataset_id = Column(UUID(as_uuid=True), nullable=True)
    input_path = Column(String(512), nullable=True)
    output_path = Column(String(512), nullable=True)
    error_msg = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())