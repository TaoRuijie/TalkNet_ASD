import boto3

def upload_file_to_s3(local_file, bucket, s3_key):
    """
    Upload a single file to an S3 bucket.

    :param local_file: Path to the local file.
    :param bucket: Name of the S3 bucket.
    :param s3_key: Path and name of the file in the S3 bucket.
    """
    # Create an S3 client
    s3_client = boto3.client("s3")
    
    try:
        # Upload the file
        s3_client.upload_file(local_file, bucket, s3_key)
        print(f"Uploaded {local_file} to s3://{bucket}/{s3_key}")
    except Exception as e:
        print(f"Error uploading file: {e}")
