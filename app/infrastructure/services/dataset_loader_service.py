import os
import tempfile
import zipfile
import shutil

from app.core.interfaces.storage_interface import IStorageRepository
from app.core.interfaces import IDatasetLoader


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

        return dataset_dir

    def delete(self, dataset_dir: str) -> None:
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
