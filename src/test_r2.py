from storage import Storage
import asyncio
from dotenv import load_dotenv
import os

async def test_r2_connection():
    print("Testing R2 Connection...")
    
    # Load environment variables
    load_dotenv()
    
    # Check if all required env vars are present
    required_vars = ['R2_ENDPOINT_URL', 'R2_ACCESS_KEY_ID', 'R2_SECRET_ACCESS_KEY', 'R2_BUCKET_NAME']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Error: Missing environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        return
    
    try:
        # Initialize storage with R2
        storage = Storage(use_r2=True)
        
        # Test writing to cache
        print("\nTesting cache operations...")
        test_data = {"test": "data"}
        cache_success = await storage.save_cache("test_key", test_data)
        if cache_success:
            print("✅ Successfully wrote to cache")
        else:
            print("❌ Failed to write to cache")
        
        # Test reading from cache
        cached_data = await storage.get_cache("test_key")
        if cached_data == test_data:
            print("✅ Successfully read from cache")
        else:
            print("❌ Failed to read from cache")
        
        # Test logging
        print("\nTesting logging...")
        log_success = await storage.write_log("Test log message", "INFO")
        if log_success:
            print("✅ Successfully wrote log")
        else:
            print("❌ Failed to write log")
        
        print("\nR2 connection test completed!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        print("\nPlease verify your R2 credentials and bucket configuration.")

if __name__ == "__main__":
    asyncio.run(test_r2_connection()) 