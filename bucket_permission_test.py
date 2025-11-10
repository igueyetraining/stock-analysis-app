import oci
import logging
from datetime import datetime

# --- SCRIPT CONFIGURATION ---
# Make sure this matches your bucket name exactly
PDF_BUCKET_NAME = "stock-analysis-pdfs-20251109-0702" 
FILE_CONTENT = f"API Key Auth Test at {datetime.utcnow().isoformat()} UTC."
FILE_NAME = "api_key_permission_test.txt"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def test_bucket_upload_with_api_key():
    logger.info("--- Starting Bucket Permission Test with API KEY AUTH ---")
    
    try:
        # --- [API KEY AUTHENTICATION] ---
        # This method reads the ~/.oci/config file directly.
        # It is not affected by the broken Cloud Shell environment.
        config = oci.config.from_file()
        object_storage_client = oci.object_storage.ObjectStorageClient(config)
        # --- [END API KEY AUTHENTICATION] ---

        logger.info("Successfully authenticated using API Key from ~/.oci/config file.")

        namespace = object_storage_client.get_namespace().data
        logger.info(f"Successfully retrieved namespace: '{namespace}'")
        
        # Sanity check
        if namespace != 'idxhyx7qeouz':
            logger.error(f"FATAL: The namespace '{namespace}' from the config file does not match the expected 'idxhyx7qeouz'. Please check your ~/.oci/config file.")
            return

        logger.info(f"Preparing to upload '{FILE_NAME}' to bucket '{PDF_BUCKET_NAME}'...")
        
        content_as_bytes = FILE_CONTENT.encode('utf-8')

        object_storage_client.put_object(
            namespace_name=namespace,
            bucket_name=PDF_BUCKET_NAME,
            object_name=FILE_NAME,
            put_object_body=content_as_bytes,
            content_type="text/plain"
        )

        logger.info("---------------------------------------------------------------")
        logger.info(">>> SUCCESS! The file was uploaded successfully using API Key.")
        logger.info("This definitively proves your user has the correct permissions.")
        logger.info("---------------------------------------------------------------")

    except Exception as e:
        logger.error(">>> FAILED: An error occurred.", exc_info=True)


if __name__ == "__main__":
    test_bucket_upload_with_api_key()
