from abc import ABC, abstractmethod
import pickle
import numpy as np
import faiss
import boto3
from pathlib import Path
import os
from typing import Any, Optional
from logger_config import setup_logger

logger = setup_logger(__name__)

class StorageStrategy(ABC):
    @abstractmethod
    def save_file(self, key: str, data: Any, file_type: str) -> bool:
        pass
    
    @abstractmethod
    def load_file(self, key: str, file_type: str) -> Any:
        pass

class LocalStorageStrategy(StorageStrategy):
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        
    def save_file(self, key: str, data: Any, file_type: str) -> bool:
        try:
            if file_type == "dataset":
                with open(self.cache_dir / f"{key}.pkl", 'wb') as f:
                    pickle.dump(data, f)
            elif file_type == "embeddings":
                np.save(self.cache_dir / f"{key}.npy", data)
            elif file_type == "index":
                faiss.write_index(data, str(self.cache_dir / f"{key}.bin"))
            logger.info(f"Successfully saved {file_type} file locally")
            return True
        except Exception as e:
            logger.error(f"Error saving {file_type} file locally: {str(e)}")
            return False
        
    def load_file(self, key: str, file_type: str) -> Any:
        try:
            if file_type == "dataset":
                with open(self.cache_dir / f"{key}.pkl", 'rb') as f:
                    return pickle.load(f)
            elif file_type == "embeddings":
                return np.load(self.cache_dir / f"{key}.npy")
            elif file_type == "index":
                return faiss.read_index(str(self.cache_dir / f"{key}.bin"))
        except Exception as e:
            logger.error(f"Error loading {file_type} file locally: {str(e)}")
            return None

class R2StorageStrategy(StorageStrategy):
    def __init__(self, bucket_name: str, endpoint_url: str, access_key: str, secret_key: str):
        self.bucket_name = bucket_name
        self.client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
    
    def save_file(self, key: str, data: Any, file_type: str) -> bool:
        try:
            # Use consistent file extensions with local storage
            file_extension = ""
            if file_type == "dataset":
                file_extension = ".pkl"
                # Use pickle for dataset
                file_data = pickle.dumps(data)
            elif file_type == "embeddings":
                file_extension = ".npy"
                # Use numpy's save format for embeddings
                # Save to a temporary file first
                temp_file = Path(f"../tmp/{key}.npy")
                np.save(temp_file, data)
                with open(temp_file, 'rb') as f:
                    file_data = f.read()
                # Clean up temp file
                if temp_file.exists():
                    temp_file.unlink()
            elif file_type == "index":
                file_extension = ".bin"
                # Use FAISS's write_index format
                # Save to a temporary file first
                temp_file = Path(f"../tmp/{key}.bin")
                faiss.write_index(data, str(temp_file))
                with open(temp_file, 'rb') as f:
                    file_data = f.read()
                # Clean up temp file
                if temp_file.exists():
                    temp_file.unlink()
            
            # Upload to R2 with proper file extension
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=f"{key}{file_extension}",
                Body=file_data
            )
            logger.info(f"Successfully saved {file_type} file to R2")
            return True
        except Exception as e:
            logger.error(f"Error saving {file_type} to R2: {str(e)}")
            return False
    
    def load_file(self, key: str, file_type: str) -> Any:
        try:
            # Use consistent file extensions with local storage
            file_extension = ""
            if file_type == "dataset":
                file_extension = ".pkl"
            elif file_type == "embeddings":
                file_extension = ".npy"
            elif file_type == "index":
                file_extension = ".bin"
            
            # Download from R2 with proper file extension
            response = self.client.get_object(
                Bucket=self.bucket_name,
                Key=f"{key}{file_extension}"
            )
            file_data = response['Body'].read()
            
            # Process based on file type
            if file_type == "dataset":
                return pickle.loads(file_data)
            elif file_type == "embeddings":
                # Save to a temporary file and use np.load
                temp_file = Path(f"../tmp/{key}.npy")
                with open(temp_file, 'wb') as f:
                    f.write(file_data)
                data = np.load(temp_file)
                # Clean up temp file
                if temp_file.exists():
                    temp_file.unlink()
                return data
            elif file_type == "index":
                # Save to a temporary file and use faiss.read_index
                temp_file = Path(f"../tmp/{key}.bin")
                with open(temp_file, 'wb') as f:
                    f.write(file_data)
                data = faiss.read_index(str(temp_file))
                # Clean up temp file
                if temp_file.exists():
                    temp_file.unlink()
                return data
        except Exception as e:
            logger.error(f"Error loading {file_type} from R2: {str(e)}")
            return None

class StorageFactory:
    @staticmethod
    def create_storage(storage_type: str = "local") -> StorageStrategy:
        if storage_type == "local":
            return LocalStorageStrategy(Path("../cache_945k_bak"))
        elif storage_type == "r2":
            required_vars = ['R2_ENDPOINT_URL', 'R2_ACCESS_KEY_ID', 'R2_SECRET_ACCESS_KEY', 'R2_BUCKET_NAME']
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            if missing_vars:
                raise ValueError(f"Missing required environment variables for R2 storage: {missing_vars}")
            
            return R2StorageStrategy(
                os.getenv("R2_BUCKET_NAME"),
                os.getenv("R2_ENDPOINT_URL"),
                os.getenv("R2_ACCESS_KEY_ID"),
                os.getenv("R2_SECRET_ACCESS_KEY")
            )
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}") 