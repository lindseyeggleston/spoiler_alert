'''
This script takes 3 additional arguments and runs like so:
    python download_bucket.py file_name bucket_name path_to_write
'''
from sys import argv
import boto3


def download_file(f, bucket, write_path):
    '''
    This function reads specified files from a bucket and downloads them to a
    specified path.

    Parameters
    ----------
    f: STR - file to download from
    bucket: STR - s3 bucket to download from
    write_path: STR - file path to write to on local machine or ec2 instance

    Returns
    -------
    None
    '''
    # specify s3 as the resource
    s3 = boto3.resource('s3')
    # download file
    s3.meta.client.download_file(bucket, f, '{}/{}'.format(write_path, f))
    print('Complete...')


if __name__ == '__main__':
    _, f, bucket_name, write_path = argv
    download_file(f, bucket_name, write_path)
