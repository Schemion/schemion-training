from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import uuid


@dataclass
class Model:
    id: uuid.UUID
    name: str
    version: str
    architecture: str
    architecture_profile: str
    dataset_id: Optional[uuid.UUID]
    minio_model_path: str
    status: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))