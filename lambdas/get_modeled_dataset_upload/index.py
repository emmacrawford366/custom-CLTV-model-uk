import json
import logging
import acm_api
import os
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def format_error(error_category, error_message, lambda_name, suggestion=None):
    """
    Format error information in a unified structure for consistent error handling.
    
    Args:
        error_category (str): Category of the error (e.g., 'VALIDATION', 'API_ERROR')
        error_message (str): Detailed error message describing what went wrong
        lambda_name (str): Name of the lambda function where the error occurred
        suggestion (str, optional): Suggested action to resolve the error. 
            Defaults to 'Check Step Functions console for detailed logs'
    
    Returns:
        dict: Formatted error dictionary containing:
            - error_category: The error category
            - error_message: The detailed error message
            - lambda_name: The lambda function name
            - timestamp: ISO format timestamp of when error occurred
            - suggestion: Recommended action to resolve the error
    """
    return {
        'error_category': error_category,
        'error_message': error_message,
        'lambda_name': lambda_name,
        'timestamp': datetime.now().isoformat(),
        'suggestion': suggestion or 'Check Step Functions console for detailed logs'
    }

def lambda_handler(event, context):
    """
    Monitor the status of a Modeled Dataset Upload job in Amazon Clean Rooms ML (ACM).
    
    This function polls a dataset upload job to check if the upload of inference results
    into a modeled dataset has completed, is still in progress, or has failed. The upload
    job populates the dataset with predictions from an inference job, making them available
    for audience creation or analysis. The function is called repeatedly by Step Functions
    until the job reaches a terminal state.
    
    Args:
        event (dict): Lambda event object containing:
            - datasetUpload (dict): Upload job information with:
                - uploadJobId (str): ID of the upload job to monitor
            - workflow_type (str): Type of workflow, typically 'dataset_upload'
            - account_id (str): The AMC account ID for API access
            - instance_id (str): The AMC instance ID
        context: Lambda context object (unused)
    
    Returns:
        dict: Response with status-dependent content:
        
            For ACTIVE (completed successfully):
            - statusCode: 200
            - body:
                - message: Success message
                - uploadJobId: The upload job ID
                - status: 'ACTIVE'
                - workflow_type: The workflow type
                - uploadDetails: Full upload details from API including:
                    - modeledDatasetId: Target dataset ID
                    - createTime: When upload started
                    - completionTime: When upload completed
                    - recordCount: Number of records uploaded
            
            For CREATE_IN_PROGRESS (still uploading):
            - statusCode: 202
            - body:
                - message: In progress message
                - uploadJobId: The upload job ID
                - status: 'CREATE_IN_PROGRESS'
                - workflow_type: The workflow type
                - uploadDetails: 'CREATE_IN_PROGRESS'
            
            For CREATE_FAILED (upload failed):
            - statusCode: 400
            - body:
                - message: Failure message with reason
                - uploadJobId: The upload job ID
                - status: 'CREATE_FAILED'
                - workflow_type: The workflow type
                - error_info: Formatted error information
                - uploadDetails: 'CREATE_FAILED'
    
    Raises:
        ValueError: If validation fails with formatted error info
        Exception: If API call fails or unexpected status received, with formatted error info
        
    Environment Variables:
        SECRETS_NAME (str): Name of AWS Secrets Manager secret containing API credentials
    
    Note:
        - This function is designed to be called repeatedly (polling) until the upload
          job reaches a terminal state. Step Functions handles the retry logic.
        - Upload jobs typically complete quickly but can take longer for large datasets
        - Common failures include dataset schema mismatches or insufficient permissions
        - Once ACTIVE, the dataset is ready for audience creation or export
    """
    try:
        logger.info("Received event: %s", json.dumps(event, indent=2))
        
        # Validate required input fields
        if 'workflow_type' not in event:
            error_info = format_error('VALIDATION', "Missing 'workflow_type' field", 'get_modeled_dataset_upload')
            raise ValueError(json.dumps(error_info))

        if 'account_id' not in event:
            error_info = format_error('VALIDATION', "Missing 'account_id' field", 'get_modeled_dataset_upload')
            raise ValueError(json.dumps(error_info))
        
        if 'instance_id' not in event:
            error_info = format_error('VALIDATION', "Missing 'instance_id' field", 'get_modeled_dataset_upload')
            raise ValueError(json.dumps(error_info))
        
        if 'uploadJobId' not in event['datasetUpload']:
            error_info = format_error('VALIDATION', "Missing required field: 'uploadJobId'", 'get_modeled_dataset_upload')
            raise ValueError(json.dumps(error_info))

        upload_job_id = event['datasetUpload']['uploadJobId']
        logger.info(f"Dataset Upload Job ID: {upload_job_id}")
        
        secret_name = os.environ.get('SECRETS_NAME')
        
        # Initialize handler
        modeled_dataset_handler = acm_api.ManageModeledDatasets(
            secret_name=secret_name, 
            ACCOUNT_ID=event['account_id'], 
            INSTANCE_ID=event['instance_id']
        )
        
        # Get dataset upload status
        upload_details = modeled_dataset_handler.get_dataset_upload(upload_job_id)
        logger.info(f"Dataset upload details: {upload_details}")
        status = upload_details.get('status')
        status_reason = upload_details.get('statusReason')
        
        logger.info(f"Dataset upload status: {status}")
        
        if status == 'ACTIVE':
            logger.info("Dataset upload completed successfully")
            return {
                'statusCode': 200,
                'body': {
                    'message': 'Dataset upload completed successfully',
                    'uploadJobId': upload_job_id,
                    'status': 'ACTIVE',
                    'workflow_type': event['workflow_type'],
                    'uploadDetails': upload_details
                }
            }
        elif status == 'CREATE_IN_PROGRESS':
            logger.info("Dataset upload still in progress")
            return {
                'statusCode': 202,
                'body': {
                    'message': 'Dataset upload is still in progress',
                    'uploadJobId': upload_job_id,
                    'status': 'CREATE_IN_PROGRESS',
                    'workflow_type': event['workflow_type'],
                    'uploadDetails': 'CREATE_IN_PROGRESS'
                }
            }
        elif status == 'CREATE_FAILED':
            logger.error(f"Dataset upload failed. Error message from API: {status_reason}")
            error_info = format_error('DATASET UPLOAD', f"Dataset upload failed: {status_reason}", 'get_modeled_dataset_upload', 'Check dataset format and upload parameters')
            return {
                'statusCode': 400,
                'body': {
                    'message': f"Dataset upload failed. Error message from API: {status_reason}",
                    'uploadJobId': upload_job_id,
                    'status': 'CREATE_FAILED',
                    'workflow_type': event['workflow_type'],
                    'error_info': error_info,
                    'uploadDetails': 'CREATE_FAILED'
                }
            }
        else:
            error_info = format_error('API_ERROR', f"Unexpected dataset upload status: {status}", 'get_modeled_dataset_upload')
            raise Exception(json.dumps(error_info))
        
    except Exception as e:
        if e.args and e.args[0].startswith('{'):
            raise e
        error_info = format_error('INFRASTRUCTURE', f"Lambda execution failed: {str(e)}", 'get_modeled_dataset_upload')
        logger.error(json.dumps(error_info))
        raise Exception(json.dumps(error_info))
