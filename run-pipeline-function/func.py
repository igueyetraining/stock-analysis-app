import io
import oci
import os
import glob
import sys
import logging
import json
from fdk import response

# Add the function's own directory to the path to import the main script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your merged script as a module
import find_best_stair_and_ch_patterns as pipeline

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def run_and_upload_pipeline(args_list=None):
    """
    Executes the 3-step pipeline in memory and uploads the result.
    The working directory for functions is ephemeral, so we use /tmp.
    """
    # --- CONFIGURATION ---
    PDF_BUCKET_NAME = os.environ.get("PDF_BUCKET_NAME")
    if not PDF_BUCKET_NAME:
        raise ValueError("Environment variable PDF_BUCKET_NAME is not set.")

    # Use the /tmp directory, as it's the only writable path in OCI Functions
    os.chdir("/tmp")
    logger.info("Changed working directory to /tmp")

    # Clean up any old files from previous runs
    for f in glob.glob("/tmp/*.pdf"):
        os.remove(f)
    logger.info("Cleaned up old PDF files from /tmp.")

    try:
        # --- EXECUTE THE MAIN SCRIPT'S LOGIC ---
        logger.info("--- Invoking main pipeline logic ---")
        # We need to simulate command-line arguments for your script
        if args_list is None:
            args_list = [] # Default to no args (runs for today)
        
        # This is a key part: we call the main() function from your script
        pipeline.main(args_list)

        # --- FIND THE GENERATED PDF ---
        pdf_filename = find_latest_file("stock_analysis_report_*.pdf")
        if not pdf_filename:
             raise RuntimeError("Pipeline ran, but could not find the final PDF output.")
        logger.info(f"Pipeline generated PDF: {pdf_filename}")

        # --- UPLOAD PDF TO OCI OBJECT STORAGE ---
        logger.info(f"Uploading {pdf_filename} to bucket '{PDF_BUCKET_NAME}'...")
        signer = oci.auth.signers.get_resource_principals_signer()
        object_storage_client = oci.object_storage.ObjectStorageClient(config={}, signer=signer)
        namespace = object_storage_client.get_namespace().data

        with open(pdf_filename, "rb") as f:
            object_storage_client.put_object(
                namespace_name=namespace,
                bucket_name=PDF_BUCKET_NAME,
                object_name=os.path.basename(pdf_filename),
                put_object_body=f,
                content_type="application/pdf"
            )
        logger.info(f"Successfully uploaded {os.path.basename(pdf_filename)}.")
        return f"Pipeline completed. Report '{os.path.basename(pdf_filename)}' uploaded."

    except SystemExit as e:
        # Your script uses sys.exit(0) on success, which we can treat as a success.
        if e.code == 0:
            logger.info("Pipeline exited with code 0, assuming success.")
            # Re-check for the PDF, as the script might have finished successfully
            return run_and_upload_pipeline(args_list)
        else:
            logger.error(f"Pipeline exited with error code: {e.code}")
            return f"Pipeline failed with exit code: {e.code}"
    except Exception as e:
        logger.error(f"An error occurred during the pipeline: {str(e)}", exc_info=True)
        return f"Pipeline failed with error: {str(e)}"

def handler(ctx, data: io.BytesIO = None):
    """The entrypoint for the OCI Function."""
    logger.info("Pipeline function invoked.")
    
    # Check for arguments passed from API Gateway (for manual triggers with tickers/dates)
    args_to_pass = []
    try:
        body = json.loads(data.getvalue())
        if body.get('ticker'):
            args_to_pass.extend(['--ticker'] + body['ticker'])
        if body.get('endDate'):
            args_to_pass.extend(['--endDate', body['endDate']])
        logger.info(f"Manual trigger with args: {args_to_pass}")
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.info("No valid JSON body found. Running with default arguments.")

    result_message = run_and_upload_pipeline(args_to_pass)
    
    return response.Response(
        ctx,
        response_data=json.dumps({"status": "completed", "message": result_message}),
        headers={"Content-Type": "application/json"}
    )

def find_latest_file(pattern):
    """Finds the most recently created file in the current directory."""
    try:
        list_of_files = glob.glob(pattern)
        return max(list_of_files, key=os.path.getmtime) if list_of_files else None
    except Exception as e:
        logger.error(f"Error finding latest file for pattern '{pattern}': {e}")
        return None