from celery import Celery
from app.config import settings

celery_app = Celery(
    "worker",
    broker=settings.REDIS_BROKER_URL,
    backend=settings.REDIS_BROKER_URL
)

celery_app.conf.task_default_queue = "training_queue"
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)
import app.infrastructure.tasks.training