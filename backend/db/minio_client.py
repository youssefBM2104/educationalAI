import hashlib

from backend.core.config import settings
from minio import Minio



_client = None

def get_minio_client():
    global _client
    if _client:
        return _client
    else:
        client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_root_user,
            secret_key=settings.minio_root_password,
            secure=False,
        )
        if not client.bucket_exists(settings.minio_bucket_originals):
            client.make_bucket(settings.minio_bucket_originals)
        if not client.bucket_exists(settings.minio_bucket_markdown):
            client.make_bucket(settings.minio_bucket_markdown)
        if not client.bucket_exists(settings.minio_bucket_images):
            client.make_bucket(settings.minio_bucket_images)
        _client = client
        return _client


def upload_file(local_path,object_key,bucket):
    client = get_minio_client()
    client.fput_object(bucket, object_key, local_path)

def download_file(local_path,object_key,bucket):
    client = get_minio_client()
    client.fget_object(bucket, object_key, local_path)
def upload_image_bytes(pil_image, object_key: str, bucket: str):
    from io import BytesIO
    client = get_minio_client()
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    size = buffer.getbuffer().nbytes
    client.put_object(bucket, object_key, buffer, length=size, content_type="image/png")

def get_presigned_image_url(object_key: str, bucket: str, expires_seconds: int = 3600):
    from datetime import timedelta
    client = get_minio_client()
    return client.presigned_get_object(bucket, object_key, expires=timedelta(seconds=expires_seconds))

def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()





