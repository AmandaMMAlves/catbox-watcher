import json
import base64
import boto3


BUCKET_NAME = 'cat-watcher-input'


def lambda_handler(event, context):
    # print("Received event: " + json.dumps(event, indent=2))
    print("Received body: " + json.dumps(event['body'], indent=2))

    body = json.loads(event['body'])
    file_content = base64.b64decode(body['content'])
    file_path = 'upload/' + body['file_name']
    print("File path: " + file_path)
    s3 = boto3.client('s3')
    try:
        print("Uploading file to S3")
        s3_response = s3.put_object(Bucket=BUCKET_NAME, Key=file_path, Body=file_content)
        print("Response: " + json.dumps(s3_response))
    except Exception as e:
        raise IOError(e)
    return {
        'statusCode': 200,
        'body': json.dumps({
            'file_path': file_path
        })
    }
