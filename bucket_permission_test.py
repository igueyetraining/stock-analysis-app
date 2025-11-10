
import oci
import logging
from datetime import datetime

# --- SCRIPT CONFIGURATION ---
# !!! IMPORTANT !!!
# REPLACE THIS with the EXACT name of your private PDF bucket.
PDF_BUCKET_NAME = "stock-analysis-pdfs-20251109-0702" 
# For your case, it would be:
# PDF_BUCKET_NAME = "stock-analysis-pdfs-20251109-0702"

# The content of the file we will create
FILE_CONTENT = f"This is a test file created at {datetime.utcnow().isoformat()} UTC."
FILE_NAME = "permission_test.txt"

# Setup basic logging to see the script's progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def test_bucket_upload():
    """
    Connects to OCI Object Storage using Instance Principals (for Cloud Shell)
    and attempts to upload a simple text file.
    """
    logger.info("--- Starting Bucket Permission Test ---")
    
    try:
        logger.info("Attempting to get an OCI signer using Instance Principals...")
        # This is how services running on an OCI compute instance (like Cloud Shell) authenticate.
        signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
        logger.info("Successfully obtained OCI signer. Authentication seems to be working.")

        # --- MODIFICATION START ---
        # Create a config dictionary and explicitly set the region
        config = {'region': 'us-ashburn-1'} # <-- REPLACE with your correct region identifier
                   
        # Initialize the Object Storage client with the specific region config
        object_storage_client = oci.object_storage.ObjectStorageClient(config=config, signer=signer)
        # --- MODIFICATION END ---
        
        logger.info("Attempting to get the Object Storage namespace...")
        namespace = object_storage_client.get_namespace().data
        logger.info(f"Successfully retrieved namespace: '{namespace}'")

        logger.info(f"Preparing to upload '{FILE_NAME}' to bucket '{PDF_BUCKET_NAME}'...")
        
        # The OCI SDK's put_object method requires the content to be in bytes.
        content_as_bytes = FILE_CONTENT.encode('utf-8')

        # Make the API call to upload the object
        object_storage_client.put_object(
            namespace_name=namespace,
            bucket_name=PDF_BUCKET_NAME,
            object_name=FILE_NAME,
            put_object_body=content_as_bytes,
            content_type="text/plain"
        )

        logger.info("---------------------------------------------------------------")
        logger.info(">>> SUCCESS! The file was uploaded successfully.")
        logger.info("This confirms your IAM policy and Dynamic Group are configured correctly for this Cloud Shell instance.")
        logger.info("---------------------------------------------------------------")

    except oci.exceptions.ServiceError as e:
        logger.error(">>> FAILED: An OCI Service Error occurred.", exc_info=False)
        logger.error("--------------------- ERROR DETAILS ---------------------")
        logger.error(f"HTTP Status: {e.status}")
        logger.error(f"Error Code: {e.code}")
        logger.error(f"Message: {e.message}")
        logger.error(f"Operation: {e.operation_name}")
        logger.error("-------------------------------------------------------")
        logger.error("Troubleshooting:")
        if e.status == 404:
            logger.error("- Check for typos in your bucket name ('{}').".format(PDF_BUCKET_NAME))
            logger.error("- Verify the bucket exists in your compartment.")
            logger.error("- This error can also mean your IAM policy is incorrect. Double-check your policy statements and dynamic group rules.")
        elif e.status == 401:
            logger.error("- This is a 'Not Authorized' error. Your IAM policy is likely missing permissions or has an incorrect dynamic group rule.")

    except Exception as e:
        logger.error(">>> FAILED: A general Python error occurred.", exc_info=True)


if __name__ == "__main__":
    test_bucket_upload()
