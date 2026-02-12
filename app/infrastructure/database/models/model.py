from sqlalchemy import Boolean, UUID, String, func, DateTime, ARRAY, Text
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from datetime import datetime

from app.infrastructure.database.models import Base


class Model(Base):
    __tablename__ = "models"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    architecture: Mapped[str] = mapped_column(String(50), nullable=False)
    architecture_profile: Mapped[str] = mapped_column(String(50), nullable=False)
    classes: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=True) # тут текстовый массив для хранения классов для faster rcnn
    minio_model_path: Mapped[str] = mapped_column(String(512), nullable=False)
    user_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=True)
    is_system: Mapped[bool] = mapped_column(Boolean, nullable=False)
    base_model_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=True)
    dataset_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())