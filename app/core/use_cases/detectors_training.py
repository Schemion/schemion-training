from uuid import UUID
from datetime import datetime, timezone
import logging

from app.core.interfaces.detector_trainer_factory_interface import IDetectorTrainerFactory
from app.core.interfaces.model_weights_loader_interface import IModelWeightsLoader
from app.core.interfaces import IDatasetLoader
from app.core.interfaces.storage_interface import IStorageRepository
from app.core.interfaces.model_interface import IModelRepository
from app.core.interfaces.task_interface import ITaskRepository
from app.core.interfaces import IDatasetRepository

logger = logging.getLogger(__name__)


class DetectorTrainingUseCase:

    def __init__(
        self,
        storage: IStorageRepository,
        weights_loader: IModelWeightsLoader,
        dataset_loader: IDatasetLoader,
        task_repo: ITaskRepository,
        model_repo: IModelRepository,
        dataset_repo: IDatasetRepository,
        trainer_factory: IDetectorTrainerFactory,
    ):
        self.storage = storage
        self.weights_loader = weights_loader
        self.dataset_loader = dataset_loader
        self.task_repo = task_repo
        self.model_repo = model_repo
        self.dataset_repo = dataset_repo
        self.trainer_factory = trainer_factory

    def execute(self, message: dict) -> None:
        task_id = UUID(message["task_id"])
        model_id = UUID(message["model_id"])
        dataset_id = message["dataset_id"]

        logger.info(f"Training task {task_id} started")

        task = self.task_repo.get_by_id(task_id)
        model = self.model_repo.get_by_id(model_id)

        try:
            if not model.is_system:
                raise RuntimeError("Only system models can be fine-tuned")

            logger.info(f"Task {task_id} - downloading base weights")
            weights_path = self.weights_loader.load(model.minio_model_path)

            dataset = self.dataset_repo.get_by_id(dataset_id)
            if not dataset:
                raise RuntimeError(f"Dataset {dataset_id} not found")

            dataset_path = self.dataset_loader.load(dataset.minio_path)

            logger.info(f"Task {task_id} - creating trainer")

            trainer = self.trainer_factory.create(
                architecture=model.architecture,
                architecture_profile=model.architecture_profile,
            )

            trainer.load_model(weights_path)

            logger.info(f"Task {task_id} - training started")
            # TODO: очевидно надо класс который скачает этот датасет, а еще получит из него классы
            trainer.train(dataset_path)

            output_path = f"trained/{model.id}/model"
            trainer.export(output_path)

            task.output_path = output_path
            task.updated_at = datetime.now(timezone.utc)
            self.task_repo.update(task)

            logger.info(f"Training task {task_id} finished successfully")

            self.weights_loader.delete(weights_path)

        except Exception as exc:
            logger.exception(f"Task {task_id} - training failed")
            task.error_msg = str(exc)
            task.updated_at = datetime.now(timezone.utc)
            self.task_repo.update(task)