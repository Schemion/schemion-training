from datetime import timedelta
from minio import Minio, S3Error
from app.core.interfaces import IStorageRepository
import uuid
import io
import logging

logger = logging.getLogger(__name__)


class MinioStorage(IStorageRepository):
    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str | None = None, secure: bool = False):
        self.client = Minio(endpoint=endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self.endpoint = endpoint
        self.public_endpoint = "files.localhost"
        self.bucket = bucket

    def _ensure_bucket_exists(self, bucket: str) -> None:
        if not self.client.bucket_exists(bucket_name=bucket):
            self.client.make_bucket(bucket_name=bucket)

    def upload_file(self, file_data: bytes, filename: str, content_type: str, bucket: str) -> str:
        self._ensure_bucket_exists(bucket)
        object_name = f"{uuid.uuid4()}_{filename}"
        self.client.put_object(
            bucket_name=bucket,
            object_name=object_name,
            data=io.BytesIO(file_data),
            length=len(file_data),
            content_type=content_type,
        )
        return object_name

    def delete_file(self, object_name: str, bucket: str) -> None:
        self.client.remove_object(bucket_name=bucket, object_name=object_name)

    #TODO: Работать не будет, так как бакет не публичный
    def get_file_url(self, object_name: str, bucket: str) -> str:
        return f"http://{self.public_endpoint}/{bucket}/{object_name}"

    def get_presigned_url(self, object_name: str, bucket: str, expires: int = 3600) -> str:
        return self.client.presigned_get_object(
            bucket_name=bucket,
            object_name=object_name,
            expires=timedelta(seconds=expires)
        )

    def download_file_to_bytes(self, object_name: str, bucket: str) -> bytes:
        try:
            response = self.client.get_object(bucket_name=bucket, object_name=object_name)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as e:
            logger.error(f"Failed to download {object_name} from bucket {bucket}: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error downloading {object_name}: {e}")
            raise

    def download_file_to_path(self, object_name: str, bucket: str, local_path: str) -> None:
        try:
            self.client.fget_object(bucket_name=bucket, object_name=object_name, file_path=local_path)
        except S3Error as e:
            logger.error(f"Failed to download {object_name} to {local_path}: {e}")
            raise