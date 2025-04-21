import boto3
import os
from datetime import datetime
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Storage:
    def __init__(self, use_r2: bool = False):
        self.use_r2 = use_r2
        if use_r2:
            try:
                self.s3 = boto3.client(
                    's3',
                    endpoint_url=os.getenv('R2_ENDPOINT_URL'),
                    aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'),
                    aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY'),
                    region_name='auto'
                )
                self.bucket = os.getenv('R2_BUCKET_NAME')
                if not all([self.s3, self.bucket]):
                    logger.warning("R2 credentials not properly configured. Falling back to local storage.")
                    self.use_r2 = False
            except Exception as e:
                logger.error(f"Failed to initialize R2 storage: {e}. Falling back to local storage.")
                self.use_r2 = False
        
        # Create local directories if not using R2
        if not self.use_r2:
            self.cache_dir = Path("cache")
            self.logs_dir = Path("logs")
            self.static_dir = Path("static")
            
            # Create parent directory if needed
            parent_dir = Path("..") 
            parent_dir.mkdir(exist_ok=True)
            
            # Create subdirectories under parent
            for directory in [self.cache_dir, self.logs_dir, self.static_dir]:
                (parent_dir / directory).mkdir(exist_ok=True)

    # Cache operations
    async def save_cache(self, key: str, data: dict):
        try:
            if self.use_r2:
                self.s3.put_object(
                    Bucket=self.bucket,
                    Key=f'test/{self.cache_dir}/{key}',
                    Body=json.dumps(data)
                )
            else:
                cache_file = ".." / self.cache_dir / f"{key}"
                with open(cache_file, 'w') as f:
                    json.dump(data, f)
            return True
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
            return False

    async def get_cache(self, key: str):
        try:
            if self.use_r2:
                response = self.s3.get_object(
                    Bucket=self.bucket,
                    Key=f'test/{self.cache_dir}/{key}'
                )
                return json.loads(response['Body'].read())
            else:
                cache_file = ".." / self.cache_dir / f"{key}"
                if cache_file.exists():
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                return None
        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            return None

    # Log operations
    async def write_log(self, log_message: str, level: str = 'INFO'):
        date_str = datetime.now().strftime('%Y-%m-%d')
        try:
            if self.use_r2:
                # First try to get existing logs for today
                try:
                    response = self.s3.get_object(
                        Bucket=self.bucket,
                        Key=f'test/logs/{date_str}.log'
                    )
                    existing_logs = response['Body'].read().decode('utf-8')
                    log_content = f"{existing_logs}\n{datetime.now().isoformat()} - {level} - {log_message}"
                except self.s3.exceptions.NoSuchKey:
                    log_content = f"{datetime.now().isoformat()} - {level} - {log_message}"

                self.s3.put_object(
                    Bucket=self.bucket,
                    Key=f'test/logs/{date_str}.log',
                    Body=log_content
                )
            else:
                log_file = self.logs_dir / f"{date_str}.log"
                with open(log_file, 'a') as f:
                    f.write(f"{datetime.now().isoformat()} - {level} - {log_message}\n")
            return True
        except Exception as e:
            logger.error(f"Error writing log: {e}")
            return False

    # Static file operations
    async def upload_static_file(self, file_path: Path, content_type: str = None):
        try:
            if self.use_r2:
                extra_args = {}
                if content_type:
                    extra_args['ContentType'] = content_type

                with open(file_path, 'rb') as f:
                    self.s3.upload_fileobj(
                        f,
                        self.bucket,
                        f'test/{self.static_dir}/{file_path.name}',
                        ExtraArgs=extra_args
                    )
            else:
                # Copy file to static directory
                dest_path = self.static_dir / file_path.name
                with open(file_path, 'rb') as src, open(dest_path, 'wb') as dst:
                    dst.write(src.read())
            return True
        except Exception as e:
            logger.error(f"Error handling static file: {e}")
            return False
