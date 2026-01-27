import os
import tempfile
import zipfile
import shutil
import yaml

from app.core.interfaces.storage_interface import IStorageRepository
from app.core.interfaces import IDatasetLoader

#TODO: Протестировать пути и для faster rcnn возможно yml не подойдет, так как в моем скрипте был .txt
class DatasetLoader(IDatasetLoader):
    def __init__(self, storage: IStorageRepository, bucket: str):
        self.storage = storage
        self.bucket = bucket

    def load(self, object_name: str) -> tuple[str, str]:
        dataset_dir = tempfile.mkdtemp(prefix="dataset_")
        zip_path = os.path.join(dataset_dir, "dataset.zip")

        self.storage.download_file_to_path(
            object_name=object_name,
            bucket=self.bucket,
            local_path=zip_path,
        )

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dataset_dir)
        os.remove(zip_path)

        yaml_path = None
        for root, _, files in os.walk(dataset_dir):
            for f in files:
                if f.lower().endswith((".yaml", ".yml")):
                    yaml_path = os.path.join(root, f)
                    break
            if yaml_path:
                break

        if not yaml_path:
            shutil.rmtree(dataset_dir, ignore_errors=True)
            raise RuntimeError("Dataset yaml file not found")

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        yaml_dir = os.path.dirname(yaml_path)

        original_path = data.get("path")
        dataset_root = yaml_dir

        if isinstance(original_path, str):
            candidate = os.path.join(yaml_dir, os.path.basename(original_path))
            if os.path.isdir(candidate):
                dataset_root = candidate

        data["path"] = dataset_root

        for key in ("train", "val", "test"):
            if key not in data:
                continue

            value = data[key]

            if os.path.isabs(value):
                value = os.path.basename(value)

            data[key] = value

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)

        return dataset_dir, yaml_path

    @staticmethod
    def delete(dataset_dir: str) -> None:
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
