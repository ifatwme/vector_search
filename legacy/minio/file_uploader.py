import os
from minio import Minio
from minio.error import S3Error

client = Minio(
    endpoint="localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

bucket_name = "product-images"

if not client.bucket_exists(bucket_name):
    client.make_bucket(bucket_name)

base_path = "/root/vector_search/data/product/images"

for root_dir, _, files in os.walk(base_path):
    for file in files:
        print(f"---image: {file}---")
        local_file_path = os.path.join(root_dir, file)
        relative_path = os.path.relpath(local_file_path, base_path)

        try:
            client.fput_object(
                bucket_name=bucket_name,
                object_name=relative_path,
                file_path=local_file_path
            )
            print(f"Uploaded: {relative_path}")
        except S3Error as err:
            print(f"Failed to upload {relative_path}: {err}")
