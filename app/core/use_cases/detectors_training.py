from uuid import UUID
from datetime import datetime, timezone
import logging
import os
import json

from app.config import settings
from app.core.enums import TaskStatus
from app.core.interfaces import IDatasetLoader, IDatasetRepository
from app.core.interfaces.detector_trainer_factory_interface import IDetectorTrainerFactory
from app.core.interfaces.model_weights_loader_interface import IModelWeightsLoader
from app.core.interfaces.storage_interface import IStorageRepository
from app.core.interfaces.model_interface import IModelRepository
from app.infrastructure.database.models.model import Model

logger = logging.getLogger(__name__)


class DetectorTrainingUseCase:
    def __init__(
        self,
        storage: IStorageRepository,
        weights_loader: IModelWeightsLoader,
        dataset_loader: IDatasetLoader,
        model_repo: IModelRepository,
        dataset_repo: IDatasetRepository,
        trainer_factory: IDetectorTrainerFactory,
    ):
        self.storage = storage
        self.weights_loader = weights_loader
        self.dataset_loader = dataset_loader
        self.model_repo = model_repo
        self.dataset_repo = dataset_repo
        self.trainer_factory = trainer_factory

    def execute(self, message: dict) -> dict:
        dataset_dir = None
        weights_path = None
        metrics_path = None
        task_id_raw = str(message["task_id"])

        try:
            image_size = int(message.get("image_size")) if message.get("image_size") is not None else None
            epochs = int(message.get("epochs")) if message.get("epochs") is not None else None
            name = str(message.get("name")) if message.get("name") is not None else None

            task_id = UUID(task_id_raw)
            model_id = UUID(message["model_id"])
            dataset_id = UUID(message["dataset_id"])
            user_id = UUID(message["user_id"])

            started_at = datetime.now(timezone.utc)
            logger.info(f"Training task {task_id} started")

            model = self.model_repo.get_by_id(model_id)
            if not model:
                raise RuntimeError(f"Model {model_id} not found")

            if not model.is_system:
                raise RuntimeError("Only system models can be fine-tuned")

            logger.info(f"Task {task_id} - downloading base weights")
            weights_path = self.weights_loader.load(str(model.minio_model_path))

            dataset = self.dataset_repo.get_by_id(dataset_id)
            if not dataset:
                raise RuntimeError(f"Dataset {dataset_id} not found")

            dataset_dir, dataset_yaml = self.dataset_loader.load(dataset.minio_path)

            logger.info(f"Task {task_id} - creating trainer")

            trainer = self.trainer_factory.create(
                architecture=str(model.architecture),
                architecture_profile=str(model.architecture_profile),
            )

            trainer.load_model(weights_path)

            logger.info(f"Task {task_id} - training started")

            trainer.train(dataset_yaml, image_size=image_size, epochs=epochs, name=name)

            try:
                metrics_payload = {
                    "run_id": str(task_id),
                    "model_id": str(model_id),
                    "started_at": started_at.isoformat().replace("+00:00", "Z"),
                    "epochs": trainer.get_metrics(),
                }
                metrics_bytes = json.dumps(metrics_payload, ensure_ascii=False).encode("utf-8")
                metrics_path = self.storage.upload_file(
                    file_data=metrics_bytes,
                    filename=f"metrics_{task_id}.json",
                    content_type="application/json",
                    bucket=settings.MINIO_METRICS_BUCKET,
                )
            except Exception:
                logger.exception("Task %s - metrics upload failed", task_id)

            output_path = f"trained/{model.id}/model"
            trainer.export(output_path)
            
            model_files = [f for f in os.listdir(output_path) if f.endswith('.pt') or f.endswith('.pth')]
            if not model_files:
                raise RuntimeError(f"No model file found in {output_path}")

            model_file_path = os.path.join(output_path, model_files[0])

            with open(model_file_path, "rb") as f:
                minio_object_name = self.storage.upload_file(
                    file_data=f.read(),
                    filename=os.path.basename(model_file_path),
                    content_type="application/octet-stream",
                    bucket=settings.MINIO_MODELS_BUCKET
                )

            new_model = Model(
                name=f"{model.name}_fine_tuned",
                architecture=model.architecture,
                architecture_profile=model.architecture_profile,
                classes=model.classes,
                minio_model_path=minio_object_name,
                metrics_path=metrics_path,
                user_id=user_id,
                is_system=False,
                base_model_id=model.id,
                dataset_id=dataset.id
            )

            self.model_repo.upload_model(new_model)

            logger.info(f"Training task {task_id} finished successfully")

            return self._status_update(
                task_id=task_id_raw,
                status=TaskStatus.succeeded,
                output_path=minio_object_name,
            )

        except Exception as exc:
            logger.exception(f"Task {task_id_raw} - training failed")
            return self._status_update(
                task_id=task_id_raw,
                status=TaskStatus.failed,
                error_msg=str(exc),
            )
        finally:
            if dataset_dir:
                self.dataset_loader.delete(dataset_dir)
            if weights_path and os.path.exists(weights_path):
                self.weights_loader.delete(weights_path)

    @staticmethod
    def _status_update(
        task_id: str,
        status: TaskStatus,
        output_path: str | None = None,
        error_msg: str | None = None,
    ) -> dict:
        return {
            "task_id": task_id,
            "task_type": "training",
            "status": status.value,
            "output_path": output_path,
            "error_msg": error_msg,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
