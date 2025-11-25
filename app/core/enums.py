from enum import Enum


class QueueTypes(str, Enum):
    training_queue = "training_queue"
    training_queue_result = "training_queue_result"


class ModelStatus(str, Enum):
    pending = "pending"
    training = "training"
    completed = "completed"
    failed = "failed"