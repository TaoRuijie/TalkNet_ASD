import boto3

# Define the bucket name and file paths
bucket_name = "scenespart1"
local_file_path = "/path/to/your/local/file.txt"  # File to upload
s3_object_key = "croppedData/TalkNetOutput/"  # Path in the bucket

# Create an S3 client
s3_client = boto3.client("s3")

# Upload the file
try:
    s3_client.upload_file(local_file_path, bucket_name, s3_object_key)
    print(f"File {local_file_path} uploaded to {bucket_name}/{s3_object_key}")
except Exception as e:
    print(f"Error uploading file: {e}")
