from app.core.use_cases.detectors_training import DetectorTrainingUseCase
from app.infrastructure.celery_app import celery_app
from app.infrastructure.cloud_storage import MinioStorage
from app.infrastructure.database.repositories import DatasetRepository
from app.infrastructure.database.repositories.model_repository import ModelRepository
from app.infrastructure.database.repositories.task_repository import TaskRepository
from app.config import settings
from app.infrastructure.services.dataset_loader_service import DatasetLoader
from app.infrastructure.services.model_weights_loader_service import ModelWeightsLoader
from app.database import SessionLocal
from app.dependencies import get_detector_trainer_factory

@celery_app.task(bind=False, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def process_training_task(message: dict):
    storage = MinioStorage(
        endpoint=settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY
    )

    db = SessionLocal()

    task_repository = TaskRepository(db)
    model_repository = ModelRepository(db)
    dataset_repository = DatasetRepository(db)
    weights_loader = ModelWeightsLoader(storage=storage, bucket=settings.MINIO_MODELS_BUCKET)
    dataset_loader = DatasetLoader(storage=storage, bucket=settings.MINIO_DATASETS_BUCKET)
    trainer_factory = get_detector_trainer_factory()

    use_case = DetectorTrainingUseCase(storage=storage, weights_loader=weights_loader, task_repo=task_repository,
                                       model_repo=model_repository, trainer_factory=trainer_factory,
                                       dataset_loader=dataset_loader, dataset_repo=dataset_repository)

    use_case.execute(message)
