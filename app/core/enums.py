from enum import Enum


class QueueTypes(str, Enum):
    training_queue = "training_queue"
    training_queue_result = "training_queue_result"


class TaskStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"