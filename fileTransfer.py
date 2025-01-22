import boto3
import os

# Define the bucket name and local folder path
bucket_name = "scenespart1"
local_folder_path = "/home/ubuntu/ZeeNews"  # Folder to upload
s3_object_key_prefix = "croppedData/ZeeNews_output/"  # S3 destination prefix

# Create an S3 client
s3_client = boto3.client("s3")

def upload_folder_to_s3(local_folder, bucket, s3_prefix):
    try:
        for root, dirs, files in os.walk(local_folder):
            for file in files:
                local_file_path = os.path.join(root, file)
                # Create the relative path for S3 object key
                relative_path = os.path.relpath(local_file_path, local_folder)
                s3_object_key = os.path.join(s3_prefix, relative_path)
                
                # Upload the file
                s3_client.upload_file(local_file_path, bucket, s3_object_key)
                print(f"Uploaded {local_file_path} to s3://{bucket}/{s3_object_key}")

    except Exception as e:
        print(f"Error uploading folder to S3: {e}")

# Call the function to upload the folder
upload_folder_to_s3(local_folder_path, bucket_name, s3_object_key_prefix)
