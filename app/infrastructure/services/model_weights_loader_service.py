import tempfile
import os

from app.core.interfaces.model_weights_loader_interface import IModelWeightsLoader
from app.core.interfaces.storage_interface import IStorageRepository

class ModelWeightsLoader(IModelWeightsLoader):
    def __init__(self, storage: IStorageRepository, bucket: str):
        self.storage = storage
        self.bucket = bucket

    def load(self, object_name: str) -> str:
        extension = os.path.splitext(object_name)[1] or ".pth"

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
        tmp.close()

        self.storage.download_file_to_path(
            object_name=object_name,
            bucket=self.bucket,
            local_path=tmp.name
        )

        return tmp.name

    def delete(self, path: str) -> None:
        if os.path.exists(path):
            os.unlink(path)