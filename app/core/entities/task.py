from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import uuid


@dataclass
class Task:
    id: uuid.UUID
    user_id: uuid.UUID
    task_type: str
    model_id: uuid.UUID
    dataset_id: Optional[uuid.UUID] = None
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    error_msg: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))