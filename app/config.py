import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://admin:admin@database:5432/schemion")
    REDIS_BROKER_URL: str = os.getenv("REDIS_BROKER_URL", "redis://:adminpass@redis:6379/1")
    RABBITMQ_URL: str = os.getenv("RABBITMQ_URL", "amqp://admin:admin@rabbitmq:5672/")
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "minio:9000")
    MINIO_PUBLIC_ENDPOINT: str = os.getenv("MINIO_PUBLIC_ENDPOINT", "files.localhost")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_SCHEMAS_BUCKET: str = os.getenv("MINIO_SCHEMAS_BUCKET", "schemas-images")
    MINIO_MODELS_BUCKET: str = os.getenv("MINIO_MODELS_BUCKET", "models")
    MINIO_DATASETS_BUCKET: str = os.getenv("MINIO_DATASETS_BUCKET", "datasets")

settings = Settings()