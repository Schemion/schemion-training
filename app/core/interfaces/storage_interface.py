from abc import ABC, abstractmethod

class IStorageRepository(ABC):
    @abstractmethod
    def upload_file(self, file_data: bytes, filename: str, content_type: str, bucket: str) -> str:
        ...

    @abstractmethod
    def delete_file(self, object_name: str, bucket: str) -> None:
        ...

    @abstractmethod
    def get_file_url(self, object_name: str, bucket: str) -> str:
        ...

    @abstractmethod
    def download_file_to_bytes(self, object_name: str, bucket: str) -> bytes:
        ...

    @abstractmethod
    def download_file_to_path(self, object_name: str, bucket: str, local_path: str) -> None:
        ...