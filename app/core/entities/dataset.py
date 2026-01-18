import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class Dataset:
    id: uuid.UUID
    name: str
    minio_path: str
    user_id: Optional[uuid.UUID] = None
    description: Optional[str] = None
    num_samples: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))