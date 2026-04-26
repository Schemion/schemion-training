import pytest

from app.infrastructure.factories.detectors_trainer_factory import DetectorTrainerFactory


def test_create_returns_yolo_trainer(monkeypatch):
    captured = {}

    class FakeYoloTrainer:
        def __init__(self, architecture_profile):
            captured["profile"] = architecture_profile
            captured["instance"] = self

    monkeypatch.setattr(
        "app.infrastructure.factories.detectors_trainer_factory.YoloTrainer",
        FakeYoloTrainer,
    )

    factory = DetectorTrainerFactory()
    trainer = factory.create("YOLO", "yolo11n.pt")

    assert trainer is captured["instance"]
    assert captured["profile"] == "yolo11n.pt"


def test_create_returns_fasterrcnn_trainer_for_alias(monkeypatch):
    captured = {}

    class FakeFasterRCNNTrainer:
        def __init__(self, architecture_profile):
            captured["profile"] = architecture_profile
            captured["instance"] = self

    monkeypatch.setattr(
        "app.infrastructure.factories.detectors_trainer_factory.FasterRCNNTrainer",
        FakeFasterRCNNTrainer,
    )

    factory = DetectorTrainerFactory()
    trainer = factory.create("frcnn", "resnet50_fpn")

    assert trainer is captured["instance"]
    assert captured["profile"] == "resnet50_fpn"


def test_create_raises_for_unknown_architecture():
    factory = DetectorTrainerFactory()

    with pytest.raises(ValueError, match="Unsupported architecture"):
        factory.create("unsupported", "profile")
