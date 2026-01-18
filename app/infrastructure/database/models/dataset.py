import uuid

from sqlalchemy import Column, UUID, String, Text, Integer, DateTime, func

from app.infrastructure.database.models.base import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    minio_path = Column(String(512), nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    description = Column(Text, nullable=True)
    num_samples = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())