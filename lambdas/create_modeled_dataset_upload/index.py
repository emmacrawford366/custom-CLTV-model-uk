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
    Create a Modeled Dataset Upload job to populate an ACM dataset with inference results.
    
    This function initiates the upload of inference job results into a previously created
    modeled dataset. The upload uses the FULL_REPLACE strategy, meaning all existing data
    in the dataset will be replaced with the new inference results.
    
    Args:
        event (dict): Lambda event object containing:
            - account_id (str): The AMC account ID for API access
            - instance_id (str): The AMC instance ID
            - modeledDataset (dict): Dataset information with:
                - modeledDatasetId (str): ID of the target dataset
                - inferenceJobId (str): ID of the inference job whose results to upload
        context: Lambda context object (unused)
    
    Returns:
        dict: Response containing:
            - statusCode (int): HTTP status code (200 for success)
            - body (dict): Response body with:
                - message (str): Success message
                - uploadJobId (str): ID of the created upload job
                - modeledDatasetId (str): The target dataset ID
    
    Raises:
        ValueError: If validation fails or API call fails, with formatted error info
        
    Environment Variables:
        SECRETS_NAME (str): Name of AWS Secrets Manager secret containing API credentials
    
    Note:
        The upload strategy is hardcoded to FULL_REPLACE, which replaces all existing
        dataset contents. The upload job runs asynchronously and should be monitored
        via the get_modeled_dataset_upload lambda.
    """
    try:
        logger.info("Received event: %s", json.dumps(event, indent=2))
        
        # Validate required input fields
        required_fields = ['account_id', 'instance_id']
        for field in required_fields:
            if field not in event:
                error_info = format_error('VALIDATION', f"Missing required field: '{field}'", 'create_modeled_dataset_upload')
                raise ValueError(json.dumps(error_info))
        
        required_inference_fields = ['modeledDatasetId', 'inferenceJobId'] 
        for field in required_inference_fields:
            if field not in event['modeledDataset']:
                error_info = format_error('VALIDATION', f"Missing required field: '{field}'", 'create_modeled_dataset_upload')
                raise ValueError(json.dumps(error_info))
        
        # Extract needed values from event
        account_id = event['account_id']
        instance_id = event['instance_id']
        modeled_dataset_id = event['modeledDataset']['modeledDatasetId']
        inference_job_id = event['modeledDataset']['inferenceJobId']
        
        logger.info(f"Creating dataset upload for dataset: {modeled_dataset_id}")
        
        # Create the dataset upload
        secret_name = os.environ.get('SECRETS_NAME')
        modeled_dataset_handler = acm_api.ManageModeledDatasets(
            secret_name=secret_name, 
            ACCOUNT_ID=account_id, 
            INSTANCE_ID=instance_id
        )
        
        response = modeled_dataset_handler.create_dataset_upload(
            modeled_dataset_id=modeled_dataset_id,
            inference_job_ids=inference_job_id,
            update_strategy="FULL_REPLACE"
        )
        
        logger.info("ACM API response: %s", json.dumps(response, indent=2))
        
        return {
            'statusCode': 200,
            'body': {
                'message': 'Dataset upload created successfully',
                'uploadJobId': response.get('modeledDatasetUploadJobId'),
                'modeledDatasetId': modeled_dataset_id
            }
        }
        
    except Exception as e:
        if e.args and e.args[0].startswith('{'):
            raise e
        error_info = format_error('API_ERROR', f"Error creating dataset upload: {str(e)}", 'create_modeled_dataset_upload', 'Check modeled dataset ID and inference job status')
        logger.error(json.dumps(error_info))
        raise ValueError(json.dumps(error_info))
