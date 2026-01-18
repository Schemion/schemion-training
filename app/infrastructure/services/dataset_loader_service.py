import os
import tempfile
import zipfile
import shutil
import yaml

from app.core.interfaces.storage_interface import IStorageRepository
from app.core.interfaces import IDatasetLoader

#TODO: тут короче надо пути делать абсолютными и скорее всего весь проект заработает =)
class DatasetLoader(IDatasetLoader):
    def __init__(self, storage: IStorageRepository, bucket: str):
        self.storage = storage
        self.bucket = bucket

    def load(self, object_name: str) -> str:
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
            raise RuntimeError("Dataset yaml file not found")

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        yaml_parent = os.path.dirname(yaml_path)

        data["path"] = yaml_parent

        for key in ("train", "val", "test"):
            if key in data and isinstance(data[key], str):
                if not os.path.isabs(data[key]):
                    data[key] = os.path.join(yaml_parent, data[key])

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)

        return yaml_path

    def delete(self, dataset_dir: str) -> None:
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
