import os
import sys
import glob
import oci
import logging

# --- SCRIPT CONFIGURATION ---
# Manually set the bucket name for the test
PDF_BUCKET_NAME = "stock-analysis-pdfs-20251109-0702" 

# --- Add the function directory to the Python path ---
# This allows us to import the main pipeline script
# --- Add the function directory to the Python path ---
# This allows us to import the main pipeline script
# Get the directory where this test_runner.py script lives
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to the subdirectory containing the pipeline
function_path = os.path.join(script_dir, 'run-pipeline-function')
# Add this path to the beginning of Python's search paths
sys.path.insert(0, function_path)

try:
    import find_best_stair_and_ch_patterns as pipeline
except ImportError:
    print(f"FATAL: Could not import the main script. Ensure it is located at: {function_path}/find_best_stair_and_ch_patterns.py")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def find_latest_file(pattern):
    """Finds the most recently created file in the current directory."""
    try:
        list_of_files = glob.glob(pattern)
        return max(list_of_files, key=os.path.getmtime) if list_of_files else None
    except Exception as e:
        logger.error(f"Error finding latest file for pattern '{pattern}': {e}")
        return None

def main_test():
    """
    Runs the full pipeline logic locally in the Cloud Shell environment.
    """
    # Cloud Shell's home directory is writable, so we'll use a subdir for output
    output_dir = os.path.expanduser("~/pipeline_output")
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    logger.info(f"Changed working directory to: {output_dir}")
    
    # Clean up old files
    for f in glob.glob(f"{output_dir}/*"):
        os.remove(f)
    logger.info("Cleaned up output directory.")

    try:
        # --- EXECUTE THE MAIN SCRIPT'S LOGIC ---
        logger.info("--- [TEST] Invoking main pipeline logic ---")
        
        # Simulate command-line args here if you want
        # To test with parameters, uncomment and modify this line:
        # args_to_pass = ["--ticker", "NVDA"] 
        args_to_pass = [] # No args for a default run
        
        pipeline.main(args_to_pass)

        # --- FIND THE GENERATED PDF ---
        pdf_filename = find_latest_file("stock_analysis_report_*.pdf")
        if not pdf_filename:
            raise RuntimeError("Pipeline ran, but could not find the final PDF output in the output directory.")
        logger.info(f"Pipeline generated PDF: {pdf_filename}")

        # --- UPLOAD PDF TO OCI OBJECT STORAGE ---
        logger.info(f"Attempting to upload {pdf_filename} to bucket '{PDF_BUCKET_NAME}'...")
        
        # Cloud Shell is a VM, so we use InstancePrincipalsSecurityTokenSigner
        signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
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
        logger.info(f"SUCCESS! Successfully uploaded {os.path.basename(pdf_filename)}.")
        return f"Pipeline test completed. Report '{os.path.basename(pdf_filename)}' uploaded."

    except SystemExit as e:
        if e.code == 0:
            logger.info("Pipeline exited with code 0, which is considered a successful run.")
            # After a successful sys.exit, we can assume the PDF was created and try uploading again.
            return main_test()
        else:
            logger.error(f"Pipeline script exited with a non-zero error code: {e.code}")
    except Exception as e:
        logger.error("An error occurred during the pipeline test.", exc_info=True)
        # This will print the full Python traceback for debugging

if __name__ == "__main__":
    main_test()
