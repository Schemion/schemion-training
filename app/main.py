import asyncio
from app.infrastructure.cloud_storage import MinioStorage
from app.infrastructure.database.repositories.model_repository import ModelRepository
from app.infrastructure.database.repositories.task_repository import TaskRepository
from app.infrastructure.messaging import RabbitMQListener
from app.config import settings
from app.logger import setup_logger
from app.core.enums import QueueTypes
from app.database import SessionLocal
from app.dependencies import get_detector_factory


async def main():
    setup_logger()

    storage = MinioStorage(endpoint=settings.MINIO_ENDPOINT, access_key=settings.MINIO_ACCESS_KEY, secret_key=settings.MINIO_SECRET_KEY)

    db = SessionLocal()

    task_repository = TaskRepository(db)
    model_repository = ModelRepository(db)
    #detector_factory = get_detector_factory()

    #use_case = тут точно будет use_case

    listener = RabbitMQListener(
        queue_name=QueueTypes.training_queue,
        callback= None #use_case.execute
    )

    await listener.start()

if __name__ == "__main__":
    asyncio.run(main())