from minio import Minio
from minio.error import S3Error

def main():
    # Create a client with the MinIO server playground, its access key
    # and secret key.
    client = Minio("play.min.io",
        access_key="Q3AM3UQ867SPQQA43P2F",
        secret_key="zuf+tfteSlswRu7BJ86wekitnifILbZam1KYY3TG",
    )

    # Download data of an object.
    client.fget_object("my-bucket", "my-object", "my-filename")

    # Download data of an object of version-ID.
    client.fget_object(
        "my-bucket", "my-object", "my-filename",
        version_id="dfbd25b3-abec-4184-a4e8-5a35a5c1174d",
    )

    # Download data of an SSE-C encrypted object.
    client.fget_object(
        "my-bucket", "my-object", "my-filename",
        ssec=SseCustomerKey(b"32byteslongsecretkeymustprovided"),
    )
    
if __name__ == "__main__":
    try:
        main()
    except S3Error as exc:
        print("error occurred.", exc)