'''
This script takes 3 additional arguments and runs like so:
    $ python write_to_s3.py file_name bucket_name write_name
'''

from sys import argv
import boto3


def to_bucket(f, bucket, write_name):
    '''
    Write files to s3 bucket.

    Parameters:
    -----------
    f: STR - file to write
    bucket: STR - bucket to write to
    write_name: STR - name for S3

    Returns:
    --------
    None
    '''
    # Specify the service
    s3 = boto3.resource('s3')
    data = open(f, 'rb')
    s3.Bucket(bucket).put_object(Key=write_name, Body=data)
    print('Success! {0} added to {1} bucket'.format(write_name, bucket))


if __name__ == '__main__':
    _, f, bucket_name, write_name = argv
    to_bucket(f, bucket_name, write_name)
