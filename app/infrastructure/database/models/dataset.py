from datetime import datetime
import uuid
from sqlalchemy import UUID, Column, String, Text, DateTime, func, text
from sqlalchemy.orm import Mapped, mapped_column
from app.infrastructure.database.models.base import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"), default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    minio_path = Column(String(512), nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())