import io
import oci
import os
import glob
import sys
import logging
import json
from fdk import response

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import find_best_stair_and_ch_patterns as pipeline

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def setup_oci_api_key_auth(ctx):
    """
    Builds an OCI config dictionary from individual function configuration keys.
    """
    logger.info("Setting up OCI client using individual API Key values from function config...")
    
    key_dir = "/tmp/.oci"
    os.makedirs(key_dir, exist_ok=True)
    private_key_path = os.path.join(key_dir, "oci_api_key.pem")

    try:
        cfg = ctx.Config()
        
        # Write the private key to a temporary file
        with open(private_key_path, "w") as f:
            f.write(cfg["OCI_PRIVATE_KEY_CONTENT"])
        os.chmod(private_key_path, 0o600)

        # Manually build the config dictionary
        config = {
            "user": cfg["OCI_USER_OCID"],
            "tenancy": cfg["OCI_TENANCY_OCID"],
            "fingerprint": cfg["OCI_FINGERPRINT"],
            "region": cfg["OCI_REGION"],
            "key_file": private_key_path
        }
        
        # Validate the manually built config
        oci.config.validate_config(config)
        logger.info("Successfully built and validated OCI config.")
        
        # Create and return the authenticated client
        return oci.object_storage.ObjectStorageClient(config)

    except KeyError as e:
        logger.error(f"FATAL: Missing required function configuration key: {e}. Please check all 6 OCI keys.")
        raise
    except Exception as e:
        logger.error(f"FATAL: An error occurred during OCI setup: {e}", exc_info=True)
        raise

def run_and_upload_pipeline(ctx, object_storage_client, args_list=None):
    """
    Executes the pipeline and uploads the result using the provided client.
    """
    try:
        cfg = ctx.Config()
        PDF_BUCKET_NAME = cfg["bucket_name"]
    except KeyError:
        raise ValueError("FATAL: 'bucket_name' is not set in the Function Application configuration.")

    os.chdir("/tmp")
    logger.info("Changed working directory to /tmp and starting pipeline...")
    
    for f in glob.glob("/tmp/*.pdf"):
        os.remove(f)

    pipeline.main(args_list)

    pdf_filename = find_latest_file("stock_analysis_report_*.pdf")
    if not pdf_filename:
         raise RuntimeError("Pipeline completed, but could not find the PDF output.")
    logger.info(f"Pipeline generated PDF: {pdf_filename}")
    logger.info(f"Uploading {pdf_filename} to bucket '{PDF_BUCKET_NAME}'...")

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

def handler(ctx, data: io.BytesIO = None):
    """The entrypoint for the OCI Function."""
    logger.info("Pipeline function invoked.")
    
    try:
        object_storage_client = setup_oci_api_key_auth(ctx)
        
        args_to_pass = []
        body = {}
        if data and data.getvalue():
            try: body = json.loads(data.getvalue())
            except json.JSONDecodeError: pass
        
        if body.get('ticker'): args_to_pass.extend(['--ticker'] + body['ticker'])
        if body.get('endDate'): args_to_pass.extend(['--endDate', body['endDate']])
        
        result_message = run_and_upload_pipeline(ctx, object_storage_client, args_to_pass)

    except Exception as e:
        result_message = f"Function failed with a critical error: {str(e)}"
        logger.error(result_message, exc_info=True)

    return response.Response(
        ctx,
        response_data=json.dumps({"status": "completed", "message": result_message}),
        headers={"Content-Type": "application/json"}
    )

def find_latest_file(pattern):
    try:
        list_of_files = glob.glob(pattern)
        return max(list_of_files, key=os.path.getmtime) if list_of_files else None
    except Exception as e:
        logger.error(f"Error finding latest file for pattern '{pattern}': {e}")
        return None
