from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import ANY, MagicMock

from app.core.enums import TaskStatus
from app.core.use_cases.detectors_training import DetectorTrainingUseCase


def _build_message(task_id, model_id, dataset_id, user_id, image_size=640, epochs=3, name="exp"):
    return {
        "task_id": str(task_id),
        "model_id": str(model_id),
        "dataset_id": str(dataset_id),
        "user_id": str(user_id),
        "image_size": image_size,
        "epochs": epochs,
        "name": name,
    }


def test_execute_successful_training_uploads_model_and_returns_status(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    task_id = uuid4()
    model_id = uuid4()
    dataset_id = uuid4()
    user_id = uuid4()
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
    trainer.get_classes.return_value = ["dataset_cat", "dataset_dog"]

    def _export(output_path):
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "trained_model.pt").write_bytes(b"trained-model")

    trainer.export.side_effect = _export

    trainer_factory = MagicMock()
    trainer_factory.create.return_value = trainer

    model_repo = MagicMock()
    model_repo.get_by_id.return_value = model

    dataset_repo = MagicMock()
    dataset_repo.get_by_id.return_value = dataset

    use_case = DetectorTrainingUseCase(
        storage=storage,
        weights_loader=weights_loader,
        dataset_loader=dataset_loader,
        model_repo=model_repo,
        dataset_repo=dataset_repo,
        trainer_factory=trainer_factory,
    )

    result = use_case.execute(_build_message(task_id, model_id, dataset_id, user_id))

    assert result["task_id"] == str(task_id)
    assert result["task_type"] == "training"
    assert result["status"] == TaskStatus.succeeded.value
    assert result["error_msg"] is None
    assert result["output_path"] == "models/final.pt"

    trainer_factory.create.assert_called_once_with(
        architecture="yolo",
        architecture_profile="yolo11n.pt",
    )
    trainer.train.assert_called_once_with(
        str(dataset_yaml),
        image_size=640,
        epochs=3,
        name="exp",
        progress_callback=ANY,
    )
    dataset_loader.delete.assert_called_once_with(str(dataset_dir))
    weights_loader.delete.assert_called_once_with(str(weights_path))
    model_repo.upload_model.assert_called_once()

    uploaded_model = model_repo.upload_model.call_args[0][0]
    assert uploaded_model.base_model_id == model.id
    assert uploaded_model.dataset_id == dataset.id
    assert uploaded_model.metrics_path == "metrics/metrics.json"
    assert uploaded_model.classes == ["dataset_cat", "dataset_dog"]
    assert uploaded_model.user_id == user_id
    assert uploaded_model.is_system is False

    metrics_upload = storage.upload_file.call_args_list[0].kwargs
    metrics_payload = __import__("json").loads(metrics_upload["file_data"].decode("utf-8"))
    assert metrics_payload["schema_version"] == 1
    assert metrics_payload["task"]["id"] == str(task_id)
    assert metrics_payload["base_model"]["id"] == str(model_id)
    assert metrics_payload["dataset"]["id"] == str(dataset_id)
    assert metrics_payload["classes"] == ["dataset_cat", "dataset_dog"]
    assert metrics_payload["epochs"] == [{"epoch": 0, "loss": 1.0}]
    assert metrics_payload["latest"] == {"epoch": 0, "loss": 1.0}


def test_execute_fails_when_model_is_not_system():
    task_id = uuid4()
    model_id = uuid4()
    dataset_id = uuid4()
    user_id = uuid4()
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
        model_repo=MagicMock(get_by_id=MagicMock(return_value=model)),
        dataset_repo=MagicMock(),
        trainer_factory=MagicMock(),
    )

    result = use_case.execute(_build_message(task_id, model_id, dataset_id, user_id))

    assert result["task_id"] == str(task_id)
    assert result["status"] == TaskStatus.failed.value
    assert "Only system models can be fine-tuned" in result["error_msg"]
    use_case.weights_loader.load.assert_not_called()
    use_case.dataset_loader.load.assert_not_called()
    use_case.model_repo.upload_model.assert_not_called()


def test_execute_continues_when_metrics_upload_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    task_id = uuid4()
    model_id = uuid4()
    dataset_id = uuid4()
    user_id = uuid4()
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
    trainer.get_classes.return_value = ["cat", "dog"]

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
        model_repo=MagicMock(get_by_id=MagicMock(return_value=model)),
        dataset_repo=MagicMock(get_by_id=MagicMock(return_value=dataset)),
        trainer_factory=trainer_factory,
    )

    result = use_case.execute(_build_message(task_id, model_id, dataset_id, user_id))

    assert result["status"] == TaskStatus.succeeded.value
    assert result["output_path"] == "models/final.pt"

    uploaded_model = use_case.model_repo.upload_model.call_args[0][0]
    assert uploaded_model.metrics_path is None


def test_execute_publishes_stage_and_epoch_progress(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    task_id = uuid4()
    model_id = uuid4()
    dataset_id = uuid4()
    user_id = uuid4()
    progress_updates = []
    model = SimpleNamespace(
        id=model_id,
        name="base_model",
        architecture="yolo",
        architecture_profile="yolo11n.pt",
        classes=["base_cat"],
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
    dataset_loader = MagicMock()
    dataset_loader.load.return_value = (str(dataset_dir), str(dataset_yaml))

    trainer = MagicMock()
    trainer.get_metrics.return_value = [{"epoch": 1, "loss_total": 0.7}]
    trainer.get_classes.return_value = ["dataset_cat"]

    def _train(*_args, **kwargs):
        kwargs["progress_callback"]({"epoch": 1, "metrics": {"epoch": 1, "loss_total": 0.7}})

    def _export(output_path):
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "trained_model.pt").write_bytes(b"trained-model")

    trainer.train.side_effect = _train
    trainer.export.side_effect = _export

    use_case = DetectorTrainingUseCase(
        storage=storage,
        weights_loader=weights_loader,
        dataset_loader=dataset_loader,
        model_repo=MagicMock(get_by_id=MagicMock(return_value=model)),
        dataset_repo=MagicMock(get_by_id=MagicMock(return_value=dataset)),
        trainer_factory=MagicMock(create=MagicMock(return_value=trainer)),
        progress_callback=progress_updates.append,
    )

    result = use_case.execute(_build_message(task_id, model_id, dataset_id, user_id, epochs=5))

    assert result["status"] == TaskStatus.succeeded.value
    assert result["progress_percent"] == 100
    assert result["progress_stage"] == "succeeded"
    stages = [update["progress_stage"] for update in progress_updates]
    assert stages[:3] == ["load_weights", "load_dataset", "training"]
    assert "export" in stages
    assert "upload_model" in stages
    training_update = next(update for update in progress_updates if update["progress_stage"] == "training")
    assert training_update["progress_percent"] == 33
    assert training_update["current_epoch"] == 1
    assert training_update["total_epochs"] == 5
    assert training_update["latest_metrics"] == {"epoch": 1, "loss_total": 0.7}
