import boto3

s3_client = boto3.client('s3')
data_bucket = "toxic-text"
local_file_name = "../../input_data/train_clean.csv"
key_name = "train_clean"

def upload_data():
    response = s3_client.upload_file(local_file_name, data_bucket, key_name)
    return response

def download_data():
    s3 = boto3.client('s3')
    response = s3.download_file(data_bucket, key_name, local_file_name)
    return response
    
def main():
    # upload_data()
    download_data()

if __name__ == "__main__":
    main()
