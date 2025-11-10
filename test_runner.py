import os
import sys
import glob
import oci
import logging

# --- SCRIPT CONFIGURATION ---
PDF_BUCKET_NAME = "stock-analysis-pdfs-20251109-0702" 

# --- Add the function directory to the Python path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
function_path = os.path.join(script_dir, 'run-pipeline-function')
sys.path.insert(0, function_path)

try:
    import find_best_stair_and_ch_patterns as pipeline
except ImportError:
    print(f"FATAL: Could not import the main script. Ensure it is located at: {function_path}/find_best_stair_and_ch_patterns.py")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def find_latest_file(pattern):
    try:
        list_of_files = glob.glob(pattern)
        return max(list_of_files, key=os.path.getmtime) if list_of_files else None
    except Exception as e:
        logger.error(f"Error finding latest file for pattern '{pattern}': {e}")
        return None

def main_test():
    output_dir = os.path.expanduser("~/pipeline_output")
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    logger.info(f"Changed working directory to: {output_dir}")
    
    for f in glob.glob(f"{output_dir}/*"):
        os.remove(f)
    logger.info("Cleaned up output directory.")

    try:
        logger.info("--- [TEST] Invoking main pipeline logic ---")
        pipeline.main([]) # Pass empty list for default run

        pdf_filename = find_latest_file("stock_analysis_report_*.pdf")
        if not pdf_filename:
            raise RuntimeError("Pipeline ran, but could not find the final PDF output.")
        logger.info(f"Pipeline generated PDF: {pdf_filename}")

        logger.info(f"Attempting to upload {pdf_filename} to bucket '{PDF_BUCKET_NAME}'...")
        
        # --- [CORRECTED API KEY AUTHENTICATION] ---
        logger.info("Authenticating using API Key from ~/.oci/config file...")
        config = oci.config.from_file()
        object_storage_client = oci.object_storage.ObjectStorageClient(config)
        logger.info("Successfully authenticated using API Key.")
        # --- [END CORRECTION] ---

        namespace = object_storage_client.get_namespace().data
        logger.info(f"Successfully retrieved namespace: '{namespace}'")

        with open(pdf_filename, "rb") as f:
            object_storage_client.put_object(
                namespace_name=namespace,
                bucket_name=PDF_BUCKET_NAME,
                object_name=os.path.basename(pdf_filename),
                put_object_body=f,
                content_type="application/pdf"
            )
        logger.info(f"SUCCESS! Successfully uploaded {os.path.basename(pdf_filename)}.")

    except Exception as e:
        logger.error("An error occurred during the pipeline test.", exc_info=True)

if __name__ == "__main__":
    main_test()
