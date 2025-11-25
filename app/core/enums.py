from enum import Enum


class QueueTypes(str, Enum):
    inference_queue = "training_queue"
    inference_queue_result = "training_queue_result"


class ModelStatus(str, Enum):
    pending = "pending"
    training = "training"
    completed = "completed"
    failed = "failed"