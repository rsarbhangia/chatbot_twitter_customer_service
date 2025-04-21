import boto3
import os
from pathlib import Path
from typing import List, Optional
from logger_config import setup_logger
from dotenv import load_dotenv

logger = setup_logger(__name__)

def get_r2_client():
    """Initialize and return an R2 client"""
    load_dotenv()
    
    # Check if all required env vars are present
    required_vars = ['R2_ENDPOINT_URL', 'R2_ACCESS_KEY_ID', 'R2_SECRET_ACCESS_KEY', 'R2_BUCKET_NAME']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    # Initialize R2 client
    client = boto3.client(
        's3',
        endpoint_url=os.getenv('R2_ENDPOINT_URL'),
        aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY')
    )
    
    return client, os.getenv('R2_BUCKET_NAME')

def list_objects(prefix: str = "") -> List[str]:
    """List all objects in the bucket with the given prefix"""
    client, bucket = get_r2_client()
    
    try:
        response = client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            logger.info(f"No objects found with prefix '{prefix}'")
            return []
        
        # Extract object keys
        objects = [obj['Key'] for obj in response['Contents']]
        logger.info(f"Found {len(objects)} objects with prefix '{prefix}'")
        return objects
    
    except Exception as e:
        logger.error(f"Error listing objects: {str(e)}")
        return []

def move_to_archive(prefix: str = "", exclude_patterns: Optional[List[str]] = None):
    """
    Move objects to an archive directory
    
    Args:
        prefix: Only move objects with this prefix
        exclude_patterns: List of patterns to exclude from moving
    """
    client, bucket = get_r2_client()
    
    # Get all objects
    objects = list_objects(prefix)
    
    if not objects:
        logger.info("No objects to move")
        return
    
    # Filter out objects that match exclude patterns
    if exclude_patterns:
        filtered_objects = []
        for obj in objects:
            if not any(pattern in obj for pattern in exclude_patterns):
                filtered_objects.append(obj)
        objects = filtered_objects
        logger.info(f"After filtering, {len(objects)} objects will be moved")
    
    # Create archive directory if it doesn't exist
    archive_prefix = "archive/"
    try:
        # Check if archive directory exists
        archive_objects = list_objects(archive_prefix)
        if not archive_objects:
            # Create an empty object to represent the directory
            client.put_object(
                Bucket=bucket,
                Key=archive_prefix
            )
            logger.info(f"Created archive directory '{archive_prefix}'")
    except Exception as e:
        logger.error(f"Error creating archive directory: {str(e)}")
        return
    
    # Move objects to archive
    moved_count = 0
    for obj_key in objects:
        try:
            # Skip if object is already in archive
            if obj_key.startswith(archive_prefix):
                logger.debug(f"Skipping {obj_key} - already in archive")
                continue
            
            # Create new key in archive directory
            new_key = f"{archive_prefix}{obj_key}"
            
            # Copy object to new location
            client.copy_object(
                Bucket=bucket,
                CopySource={'Bucket': bucket, 'Key': obj_key},
                Key=new_key
            )
            
            # Delete original object
            client.delete_object(
                Bucket=bucket,
                Key=obj_key
            )
            
            moved_count += 1
            logger.debug(f"Moved {obj_key} to {new_key}")
            
        except Exception as e:
            logger.error(f"Error moving {obj_key}: {str(e)}")
    
    logger.info(f"Successfully moved {moved_count} objects to archive")

def download_from_r2(key: str, local_path: Optional[Path] = None):
    """
    Download a file from R2 to local storage
    
    Args:
        key: The key of the object in R2
        local_path: Local path to save the file (default: current directory)
    """
    client, bucket = get_r2_client()
    
    if local_path is None:
        local_path = Path.cwd()
    
    # Create directory if it doesn't exist
    local_path.mkdir(parents=True, exist_ok=True)
    
    # Get filename from key
    filename = Path(key).name
    if not filename:
        filename = "downloaded_file"
    
    # Full path for the file
    file_path = local_path / filename
    
    try:
        # Download the file
        client.download_file(
            Bucket=bucket,
            Key=key,
            Filename=str(file_path)
        )
        
        logger.info(f"Successfully downloaded {key} to {file_path}")
        return file_path
    
    except Exception as e:
        logger.error(f"Error downloading {key}: {str(e)}")
        return None

def upload_to_r2(local_path: Path, key: Optional[str] = None):
    """
    Upload a file from local storage to R2
    
    Args:
        local_path: Path to the local file
        key: Key to use in R2 (default: filename)
    """
    client, bucket = get_r2_client()
    
    if not local_path.exists():
        logger.error(f"File not found: {local_path}")
        return False
    
    # Use filename as key if not provided
    if key is None:
        key = local_path.name
    
    try:
        # Upload the file
        client.upload_file(
            Filename=str(local_path),
            Bucket=bucket,
            Key=key
        )
        
        logger.info(f"Successfully uploaded {local_path} to {key}")
        return True
    
    except Exception as e:
        logger.error(f"Error uploading {local_path}: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="R2 Storage Utilities")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List objects in R2 bucket")
    list_parser.add_argument("--prefix", default="", help="Prefix to filter objects")
    
    # Move command
    move_parser = subparsers.add_parser("move", help="Move objects to archive directory")
    move_parser.add_argument("--prefix", default="", help="Prefix to filter objects")
    move_parser.add_argument("--exclude", nargs="+", help="Patterns to exclude")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download a file from R2")
    download_parser.add_argument("key", help="Key of the object to download")
    download_parser.add_argument("--path", help="Local path to save the file")
    
    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload a file to R2")
    upload_parser.add_argument("path", help="Path to the local file")
    upload_parser.add_argument("--key", help="Key to use in R2")
    
    args = parser.parse_args()
    
    if args.command == "list":
        objects = list_objects(args.prefix)
        for obj in objects:
            print(obj)
    
    elif args.command == "move":
        move_to_archive(args.prefix, args.exclude)
    
    elif args.command == "download":
        path = Path(args.path) if args.path else None
        download_from_r2(args.key, path)
    
    elif args.command == "upload":
        path = Path(args.path)
        upload_to_r2(path, args.key)
    
    else:
        parser.print_help() 