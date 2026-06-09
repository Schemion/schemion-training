import json
import logging
import signal
import threading
import time
from datetime import datetime, timezone

from bobber import BobberClient

from app.core.use_cases.detectors_training import DetectorTrainingUseCase
from app.core.enums import QueueTypes, TaskStatus
from app.infrastructure.cloud_storage import MinioStorage
from app.infrastructure.database.repositories import DatasetRepository
from app.infrastructure.database.repositories.model_repository import ModelRepository
from app.config import settings
from app.infrastructure.services.dataset_loader_service import DatasetLoader
from app.infrastructure.services.model_weights_loader_service import ModelWeightsLoader
from app.database import SessionLocal
from app.dependencies import get_detector_trainer_factory
from app.logger import setup_logger

logger = logging.getLogger(__name__)

def _publish_task_status(client: BobberClient, message: dict) -> None:
    message.setdefault("task_type", "training")
    message.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

    task_id = message.get("task_id", "unknown")
    status = message.get("status", "unknown")
    key = f"training_{task_id}_{status}"
    success = client.produce(QueueTypes.training_queue_result.value, key, json.dumps(message))
    if not success:
        logger.error("Failed to publish training status update: %s", message)


def process_training_task(message: dict, client: BobberClient):
    _publish_task_status(
        client,
        {
            "task_id": message.get("task_id"),
            "status": TaskStatus.running.value,
        },
    )

    storage = MinioStorage(
        endpoint=settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY
    )
    db = SessionLocal()
    status_update = None
    try:
        model_repository = ModelRepository(db)
        dataset_repository = DatasetRepository(db)
        weights_loader = ModelWeightsLoader(storage=storage, bucket=settings.MINIO_MODELS_BUCKET)
        dataset_loader = DatasetLoader(storage=storage, bucket=settings.MINIO_DATASETS_BUCKET)
        trainer_factory = get_detector_trainer_factory()

        use_case = DetectorTrainingUseCase(storage=storage, weights_loader=weights_loader,
                                           model_repo=model_repository, trainer_factory=trainer_factory,
                                           dataset_loader=dataset_loader, dataset_repo=dataset_repository,
                                           progress_callback=lambda update: _publish_task_status(client, update))

        status_update = use_case.execute(message)
    except Exception as exc:
        logger.exception("Unhandled training worker failure")
        status_update = {
            "task_id": message.get("task_id"),
            "task_type": "training",
            "status": TaskStatus.failed.value,
            "error_msg": str(exc),
        }
    finally:
        db.close()

    if status_update:
        _publish_task_status(client, status_update)


def _parse_message(payload: dict) -> dict | None:
    raw_value = payload.get("value")
    if raw_value is None:
        logger.error("Broker message missing 'value': %s", payload)
        return None
    if isinstance(raw_value, dict):
        return raw_value
    if not isinstance(raw_value, str):
        logger.error("Broker message value must be str or dict, got %s", type(raw_value))
        return None
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        logger.exception("Failed to decode broker message JSON: %s", raw_value)
        return None


def _on_broker_message(client: BobberClient, payload: dict) -> None:
    message = _parse_message(payload)
    if not message:
        return
    logger.info("Received training task %s", message.get("task_id"))
    process_training_task(message, client)


def main() -> None:
    setup_logger()

    client = BobberClient(host=settings.BOBBER_HOST, port=settings.BOBBER_PORT)
    if not client.healthcheck():
        raise ConnectionError("Bobber broker unavailable")

    stop_event = threading.Event()

    def _handle_signal(_sig, _frame):
        logger.info("Shutdown signal received")
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    topic = QueueTypes.training_queue.value
    client.subscribe(topic, lambda payload: _on_broker_message(client, payload))
    logger.info("Listening to broker topic '%s'", topic)

    try:
        while not stop_event.is_set():
            time.sleep(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
