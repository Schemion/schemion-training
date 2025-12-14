import uuid
from sqlalchemy import Boolean, Column, UUID, String, func, DateTime, Enum, ARRAY, Text
from app.core.enums import ModelStatus
from app.infrastructure.database.models.base import Base


class Model(Base):
    __tablename__ = "models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    architecture = Column(String(50), nullable=False)
    architecture_profile = Column(String(50), nullable=False)
    classes = Column(ARRAY(Text), nullable=True) # тут текстовый массив для хранения классов для faster rcnn
    minio_model_path = Column(String(512), nullable=False)
    status = Column(Enum(ModelStatus, name="model_status"), nullable=False, default=ModelStatus.pending)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    is_system = Column(Boolean, default=False, nullable=False)
    base_model_id = Column(UUID(as_uuid=True), nullable=True)
    dataset_id = Column(UUID(as_uuid=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())