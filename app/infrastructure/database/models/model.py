import uuid

from sqlalchemy import Boolean, UUID, Column, ForeignKey, String, func, DateTime, ARRAY, Text, text
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from datetime import datetime, timezone
from typing import Optional

from app.infrastructure.database.models.base import Base

class Model(Base):
    __tablename__ = "models"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"), default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    architecture = Column(String(50), nullable=False)
    architecture_profile = Column(String(512), nullable=False)  # resnet или еще что-то
    classes = Column(ARRAY(Text), nullable=True)
    minio_model_path = Column(String(512), nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    is_system = Column(Boolean, default=False, nullable=False)
    base_model_id = Column(UUID(as_uuid=True), nullable=True)
    dataset_id = Column(UUID(as_uuid=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    