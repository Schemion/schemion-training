from datetime import datetime
from sqlalchemy import UUID, String, Text, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column
from app.infrastructure.database.models.base import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    minio_path: Mapped[str] = mapped_column(String(512), nullable=False)
    user_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())