import asyncio

from app.core.use_cases.detectors_training import DetectorTrainingUseCase
from app.infrastructure.cloud_storage import MinioStorage
from app.infrastructure.database.repositories.model_repository import ModelRepository
from app.infrastructure.database.repositories.task_repository import TaskRepository
from app.infrastructure.messaging import RabbitMQListener
from app.config import settings
from app.infrastructure.services.model_weights_loader_service import ModelWeightsLoader
from app.logger import setup_logger
from app.core.enums import QueueTypes
from app.database import SessionLocal
from app.dependencies import get_detector_trainer_factory

# вообще наверное надо заранее думать о том, что мы тренеруем, то есть детектор или нет, но пока логика только для детекторов
async def main():
    setup_logger()

    storage = MinioStorage(endpoint=settings.MINIO_ENDPOINT, access_key=settings.MINIO_ACCESS_KEY, secret_key=settings.MINIO_SECRET_KEY)

    db = SessionLocal()

    task_repository = TaskRepository(db)
    model_repository = ModelRepository(db)
    weights_loader = ModelWeightsLoader(storage=storage, bucket=settings.MINIO_MODELS_BUCKET)
    trainer_factory = get_detector_trainer_factory()

    use_case = DetectorTrainingUseCase(storage=storage, weights_loader=weights_loader, task_repo=task_repository, model_repo=model_repository, trainer_factory=trainer_factory)

    listener = RabbitMQListener(
        queue_name=QueueTypes.training_queue,
        callback= use_case.execute
    )

    await listener.start()

if __name__ == "__main__":
    asyncio.run(main())