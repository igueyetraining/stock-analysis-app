import io
import json
import oci
import os
import logging
from fdk import response
from datetime import datetime, timedelta

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def handler(ctx, data: io.BytesIO=None):
    """
    Handles both listing reports and generating download URLs based on the request path.
    - GET /reports -> lists files
    - GET /reports/some-file.pdf -> gets download URL
    """
    signer = oci.auth.signers.get_resource_principals_signer()
    object_storage_client = oci.object_storage.ObjectStorageClient(config={}, signer=signer)
    
    PDF_BUCKET_NAME = os.environ.get("PDF_BUCKET_NAME")
    if not PDF_BUCKET_NAME:
        return response.Response(ctx, status_code=500, response_data=json.dumps({"error": "PDF_BUCKET_NAME env var not set."}))

    try:
        namespace = object_storage_client.get_namespace().data
        request_url_path = ctx.RequestURL()

        # Logic to LIST reports
        if request_url_path.endswith('/reports'):
            list_objects_response = object_storage_client.list_objects(namespace, PDF_BUCKET_NAME)
            files = sorted(
                [{"name": obj.name, "size": obj.size, "last_modified": obj.time_modified.isoformat()} 
                 for obj in list_objects_response.data.objects if obj.name.endswith('.pdf')],
                key=lambda x: x['last_modified'],
                reverse=True
            )
            return response.Response(ctx, response_data=json.dumps(files), headers={"Content-Type": "application/json"})

        # Logic to GET DOWNLOAD URL for a single report
        else:
            filename = os.path.basename(request_url_path)
            if not filename or not filename.endswith('.pdf'):
                return response.Response(ctx, status_code=400, response_data=json.dumps({"error": "Invalid or missing filename in URL."}))

            par_details = oci.object_storage.models.CreatePreauthenticatedRequestDetails(
                name=f"par-for-{filename}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                object_name=filename,
                access_type='ObjectRead',
                time_expires=datetime.utcnow().replace(microsecond=0) + timedelta(minutes=5)
            )
            
            par = object_storage_client.create_preauthenticated_request(namespace, PDF_BUCKET_NAME, par_details)
            base_url = f"https://objectstorage.{signer.region}.oraclecloud.com"
            download_url = f"{base_url}{par.data.access_uri}"
            
            return response.Response(ctx, response_data=json.dumps({"download_url": download_url}), headers={"Content-Type": "application/json"})

    except Exception as e:
        logging.error(f"Error in API helper function: {str(e)}", exc_info=True)
        return response.Response(ctx, status_code=500, response_data=json.dumps({"error": str(e)}))