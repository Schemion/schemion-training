from uuid import UUID
from datetime import datetime, timezone
import logging
import os
import json
from collections.abc import Callable
from typing import Any

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
        progress_callback: Callable[[dict], None] | None = None,
    ):
        self.storage = storage
        self.weights_loader = weights_loader
        self.dataset_loader = dataset_loader
        self.model_repo = model_repo
        self.dataset_repo = dataset_repo
        self.trainer_factory = trainer_factory
        self.progress_callback = progress_callback

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
            self._publish_progress(task_id_raw, "load_weights", 5, "Downloading base model weights")
            weights_path = self.weights_loader.load(str(model.minio_model_path))

            dataset = self.dataset_repo.get_by_id(dataset_id)
            if not dataset:
                raise RuntimeError(f"Dataset {dataset_id} not found")

            self._publish_progress(task_id_raw, "load_dataset", 15, "Downloading and preparing dataset")
            dataset_dir, dataset_yaml = self.dataset_loader.load(dataset.minio_path)

            logger.info(f"Task {task_id} - creating trainer")

            trainer = self.trainer_factory.create(
                architecture=str(model.architecture),
                architecture_profile=str(model.architecture_profile),
            )

            trainer.load_model(weights_path)

            logger.info(f"Task {task_id} - training started")

            total_epochs = epochs if epochs is not None else 10
            trainer.train(
                dataset_yaml,
                image_size=image_size,
                epochs=epochs,
                name=name,
                progress_callback=self._epoch_progress_callback(task_id_raw, total_epochs),
            )

            try:
                self._publish_progress(task_id_raw, "collect_metrics", 90, "Collecting training metrics")
                trainer_classes = self._trainer_classes(trainer)
                metrics_payload = self._build_metrics_payload(
                    task_id=task_id,
                    model=model,
                    model_id=model_id,
                    dataset=dataset,
                    dataset_id=dataset_id,
                    started_at=started_at,
                    finished_at=datetime.now(timezone.utc),
                    classes=trainer_classes or list(model.classes or []),
                    epochs=trainer.get_metrics(),
                )
                metrics_bytes = json.dumps(metrics_payload, ensure_ascii=False).encode("utf-8")
                metrics_path = self.storage.upload_file(
                    file_data=metrics_bytes,
                    filename=f"metrics_{task_id}.json",
                    content_type="application/json",
                    bucket=settings.MINIO_METRICS_BUCKET,
                )
            except Exception:
                logger.exception("Task %s - metrics upload failed", task_id)
                trainer_classes = self._trainer_classes(trainer)

            output_path = f"trained/{model.id}/model"
            self._publish_progress(task_id_raw, "export", 95, "Exporting trained model")
            trainer.export(output_path)
            
            model_files = [f for f in os.listdir(output_path) if f.endswith('.pt') or f.endswith('.pth')]
            if not model_files:
                raise RuntimeError(f"No model file found in {output_path}")

            model_file_path = os.path.join(output_path, model_files[0])

            self._publish_progress(task_id_raw, "upload_model", 98, "Uploading trained model")
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
                classes=trainer_classes or list(model.classes or []),
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
                progress_percent=100,
                progress_stage="succeeded",
                progress_message="Training completed",
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
        progress_percent: int | None = None,
        progress_stage: str | None = None,
        progress_message: str | None = None,
        current_epoch: int | None = None,
        total_epochs: int | None = None,
        latest_metrics: dict[str, Any] | None = None,
    ) -> dict:
        return {
            "task_id": task_id,
            "task_type": "training",
            "status": status.value,
            "output_path": output_path,
            "error_msg": error_msg,
            "progress_percent": progress_percent,
            "progress_stage": progress_stage,
            "progress_message": progress_message,
            "current_epoch": current_epoch,
            "total_epochs": total_epochs,
            "latest_metrics": latest_metrics,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _publish_progress(
        self,
        task_id: str,
        stage: str,
        percent: int,
        message: str,
        current_epoch: int | None = None,
        total_epochs: int | None = None,
        latest_metrics: dict[str, Any] | None = None,
    ) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(
                self._status_update(
                    task_id=task_id,
                    status=TaskStatus.running,
                    progress_percent=percent,
                    progress_stage=stage,
                    progress_message=message,
                    current_epoch=current_epoch,
                    total_epochs=total_epochs,
                    latest_metrics=latest_metrics,
                )
            )
        except Exception:
            logger.exception("Failed to publish training progress for task %s", task_id)

    def _epoch_progress_callback(self, task_id: str, total_epochs: int) -> Callable[[dict], None]:
        def _callback(update: dict) -> None:
            current_epoch = int(update.get("epoch") or 0)
            safe_total = max(total_epochs, 1)
            percent = min(85, 20 + round((current_epoch / safe_total) * 65))
            metrics = update.get("metrics") if isinstance(update.get("metrics"), dict) else None
            self._publish_progress(
                task_id=task_id,
                stage="training",
                percent=percent,
                message=f"Epoch {current_epoch}/{safe_total} finished",
                current_epoch=current_epoch,
                total_epochs=safe_total,
                latest_metrics=metrics,
            )

        return _callback

    @staticmethod
    def _trainer_classes(trainer) -> list[str]:
        try:
            classes = trainer.get_classes()
        except Exception:
            logger.exception("Failed to read trainer classes")
            return []
        if not classes:
            return []
        return [str(item) for item in classes]

    @classmethod
    def _build_metrics_payload(
        cls,
        task_id: UUID,
        model,
        model_id: UUID,
        dataset,
        dataset_id: UUID,
        started_at: datetime,
        finished_at: datetime,
        classes: list[str],
        epochs: list[dict],
    ) -> dict:
        return {
            "schema_version": 1,
            "task": {"id": str(task_id), "type": "training"},
            "base_model": {
                "id": str(model_id),
                "name": getattr(model, "name", None),
                "architecture": getattr(model, "architecture", None),
                "architecture_profile": getattr(model, "architecture_profile", None),
            },
            "dataset": {
                "id": str(dataset_id),
                "minio_path": getattr(dataset, "minio_path", None),
            },
            "timing": {
                "started_at": started_at.isoformat().replace("+00:00", "Z"),
                "finished_at": finished_at.isoformat().replace("+00:00", "Z"),
                "duration_seconds": max((finished_at - started_at).total_seconds(), 0.0),
            },
            "classes": classes,
            "epochs": epochs,
            "latest": epochs[-1] if epochs else None,
            "best": cls._best_summary(epochs),
        }

    @staticmethod
    def _best_summary(epochs: list[dict]) -> dict | None:
        if not epochs:
            return None

        maximize_keys = (
            "metrics/mAP50-95(B)",
            "metrics/mAP50(B)",
            "metrics/precision(B)",
            "metrics/recall(B)",
            "fitness",
        )
        minimize_keys = (
            "loss_total",
            "train/box_loss",
            "train/cls_loss",
            "train/dfl_loss",
        )

        for key in maximize_keys:
            candidates = [row for row in epochs if isinstance(row.get(key), (int, float))]
            if candidates:
                best = max(candidates, key=lambda row: row[key])
                return {"selection_key": key, "mode": "max", "epoch": best.get("epoch"), "metrics": best}

        for key in minimize_keys:
            candidates = [row for row in epochs if isinstance(row.get(key), (int, float))]
            if candidates:
                best = min(candidates, key=lambda row: row[key])
                return {"selection_key": key, "mode": "min", "epoch": best.get("epoch"), "metrics": best}

        return None
