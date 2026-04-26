from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import MagicMock

from app.core.enums import TaskStatus
from app.core.use_cases.detectors_training import DetectorTrainingUseCase


def _build_message(task_id, model_id, dataset_id, image_size=640, epochs=3, name="exp"):
    return {
        "task_id": str(task_id),
        "model_id": str(model_id),
        "dataset_id": str(dataset_id),
        "image_size": image_size,
        "epochs": epochs,
        "name": name,
    }


def test_execute_successful_training_uploads_model_and_updates_task(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    task_id = uuid4()
    model_id = uuid4()
    dataset_id = uuid4()

    task = SimpleNamespace(
        id=task_id,
        user_id=uuid4(),
        status=TaskStatus.queued.value,
        output_path=None,
        error_msg=None,
        updated_at=None,
    )
    model = SimpleNamespace(
        id=model_id,
        name="base_model",
        architecture="yolo",
        architecture_profile="yolo11n.pt",
        classes=["cat", "dog"],
        is_system=True,
        minio_model_path="models/base.pt",
    )
    dataset = SimpleNamespace(id=dataset_id, minio_path="datasets/archive.zip")

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    dataset_yaml = dataset_dir / "data.yaml"
    dataset_yaml.write_text("train: images/train\nval: images/val\n", encoding="utf-8")

    weights_path = tmp_path / "weights.pt"
    weights_path.write_bytes(b"weights")

    storage = MagicMock()
    storage.upload_file.side_effect = ["metrics/metrics.json", "models/final.pt"]

    weights_loader = MagicMock()
    weights_loader.load.return_value = str(weights_path)

    def _delete_weights(path):
        path_obj = Path(path)
        if path_obj.exists():
            path_obj.unlink()

    weights_loader.delete.side_effect = _delete_weights

    dataset_loader = MagicMock()
    dataset_loader.load.return_value = (str(dataset_dir), str(dataset_yaml))

    trainer = MagicMock()
    trainer.get_metrics.return_value = [{"epoch": 0, "loss": 1.0}]

    def _export(output_path):
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "trained_model.pt").write_bytes(b"trained-model")

    trainer.export.side_effect = _export

    trainer_factory = MagicMock()
    trainer_factory.create.return_value = trainer

    task_repo = MagicMock()
    task_repo.get_by_id.return_value = task

    model_repo = MagicMock()
    model_repo.get_by_id.return_value = model

    dataset_repo = MagicMock()
    dataset_repo.get_by_id.return_value = dataset

    use_case = DetectorTrainingUseCase(
        storage=storage,
        weights_loader=weights_loader,
        dataset_loader=dataset_loader,
        task_repo=task_repo,
        model_repo=model_repo,
        dataset_repo=dataset_repo,
        trainer_factory=trainer_factory,
    )

    use_case.execute(_build_message(task_id, model_id, dataset_id))

    assert task.status == TaskStatus.succeeded.value
    assert task.error_msg is None
    assert task.output_path == "models/final.pt"
    assert task_repo.update.call_count == 1

    trainer_factory.create.assert_called_once_with(
        architecture="yolo",
        architecture_profile="yolo11n.pt",
    )
    trainer.train.assert_called_once_with(str(dataset_yaml), image_size=640, epochs=3, name="exp")
    dataset_loader.delete.assert_called_once_with(str(dataset_dir))
    weights_loader.delete.assert_called_once_with(str(weights_path))
    model_repo.upload_model.assert_called_once()

    uploaded_model = model_repo.upload_model.call_args[0][0]
    assert uploaded_model.base_model_id == model.id
    assert uploaded_model.dataset_id == dataset.id
    assert uploaded_model.metrics_path == "metrics/metrics.json"
    assert uploaded_model.is_system is False


def test_execute_fails_when_model_is_not_system():
    task_id = uuid4()
    model_id = uuid4()
    dataset_id = uuid4()

    task = SimpleNamespace(
        id=task_id,
        user_id=uuid4(),
        status=TaskStatus.queued.value,
        output_path=None,
        error_msg=None,
        updated_at=None,
    )
    model = SimpleNamespace(
        id=model_id,
        name="custom_model",
        architecture="yolo",
        architecture_profile="yolo11n.pt",
        classes=["cat", "dog"],
        is_system=False,
        minio_model_path="models/custom.pt",
    )

    use_case = DetectorTrainingUseCase(
        storage=MagicMock(),
        weights_loader=MagicMock(),
        dataset_loader=MagicMock(),
        task_repo=MagicMock(get_by_id=MagicMock(return_value=task)),
        model_repo=MagicMock(get_by_id=MagicMock(return_value=model)),
        dataset_repo=MagicMock(),
        trainer_factory=MagicMock(),
    )

    use_case.execute(_build_message(task_id, model_id, dataset_id))

    assert task.status == TaskStatus.failed.value
    assert "Only system models can be fine-tuned" in task.error_msg
    use_case.task_repo.update.assert_called_once_with(task)
    use_case.weights_loader.load.assert_not_called()
    use_case.dataset_loader.load.assert_not_called()
    use_case.model_repo.upload_model.assert_not_called()


def test_execute_continues_when_metrics_upload_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    task_id = uuid4()
    model_id = uuid4()
    dataset_id = uuid4()

    task = SimpleNamespace(
        id=task_id,
        user_id=uuid4(),
        status=TaskStatus.queued.value,
        output_path=None,
        error_msg=None,
        updated_at=None,
    )
    model = SimpleNamespace(
        id=model_id,
        name="base_model",
        architecture="yolo",
        architecture_profile="yolo11n.pt",
        classes=["cat", "dog"],
        is_system=True,
        minio_model_path="models/base.pt",
    )
    dataset = SimpleNamespace(id=dataset_id, minio_path="datasets/archive.zip")

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    dataset_yaml = dataset_dir / "data.yaml"
    dataset_yaml.write_text("train: images/train\nval: images/val\n", encoding="utf-8")
    weights_path = tmp_path / "weights.pt"
    weights_path.write_bytes(b"weights")

    storage = MagicMock()
    storage.upload_file.side_effect = [RuntimeError("metrics unavailable"), "models/final.pt"]

    weights_loader = MagicMock()
    weights_loader.load.return_value = str(weights_path)

    def _delete_weights(path):
        path_obj = Path(path)
        if path_obj.exists():
            path_obj.unlink()

    weights_loader.delete.side_effect = _delete_weights

    dataset_loader = MagicMock()
    dataset_loader.load.return_value = (str(dataset_dir), str(dataset_yaml))

    trainer = MagicMock()
    trainer.get_metrics.return_value = [{"epoch": 0, "loss": 1.0}]

    def _export(output_path):
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "trained_model.pt").write_bytes(b"trained-model")

    trainer.export.side_effect = _export

    trainer_factory = MagicMock(create=MagicMock(return_value=trainer))

    use_case = DetectorTrainingUseCase(
        storage=storage,
        weights_loader=weights_loader,
        dataset_loader=dataset_loader,
        task_repo=MagicMock(get_by_id=MagicMock(return_value=task)),
        model_repo=MagicMock(get_by_id=MagicMock(return_value=model)),
        dataset_repo=MagicMock(get_by_id=MagicMock(return_value=dataset)),
        trainer_factory=trainer_factory,
    )

    use_case.execute(_build_message(task_id, model_id, dataset_id))

    assert task.status == TaskStatus.succeeded.value
    assert task.output_path == "models/final.pt"

    uploaded_model = use_case.model_repo.upload_model.call_args[0][0]
    assert uploaded_model.metrics_path is None
