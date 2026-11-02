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
    Retrieve ML Data Export details and status from Amazon Clean Rooms ML (ACM).
    
    This function monitors an ML data export job that was previously created. For
    measurement (MMM) use cases, the export contains model performance metrics that
    are downloaded and stored in S3 via pre-signed URLs.
    
    Args:
        event (dict): Lambda event object containing:
            - account_id (str): The AMC account ID for API access
            - instance_id (str): The AMC instance ID
            - mlDataExport.mlDataExportId (str): ID of the export job to monitor
              OR
            - mlDataExportId (str): Alternative location for the export ID
        context: Lambda context object (unused)
    
    Returns:
        dict: Response containing:
            - statusCode (int): HTTP status code (200 for success)
            - body (dict): Response body with:
                - message (str): Status message
                - mlDataExport (dict): Export details including:
                    - mlDataExportId: The export job ID
                    - createTime: When the export was created
                    - status: Current status (ACTIVE, CREATE_IN_PROGRESS, etc.)
                    - statusReason: Reason for current status
                    - validUntil: Expiration time for pre-signed URLs
                    - preSignedS3Uris: List of URLs for downloading export files
                    - dataType: Type of data exported (e.g., 'MMM')
                    - instanceId: The instance ID
                    - trainedModelId: ID of the model that generated this export
    
    Raises:
        ValueError: If validation fails, export is in failed state, or API call fails,
                   with formatted error info
        
    Environment Variables:
        SECRETS_NAME (str): Name of AWS Secrets Manager secret containing API credentials
    
    Note:
        - Pre-signed URLs are temporary and expire based on validUntil timestamp
        - If status is CREATE_FAILED or INACTIVE, an error is raised
        - Pre-signed URLs should be downloaded promptly via the download_url lambda
    """
    try:
        # Log received event
        logger.info("Received event: %s", json.dumps(event, indent=2))
        
        # Validate required input fields
        required_fields = ['account_id', 'instance_id']
        for field in required_fields:
            if field not in event:
                error_info = format_error('VALIDATION', f"Missing required field: '{field}'", 'get_ml_data_export', 'Ensure all required fields are provided in the payload')
                raise ValueError(json.dumps(error_info))
        
        # Extract needed values from event
        account_id = event['account_id']
        instance_id = event['instance_id']
        ml_data_export_id = event.get('mlDataExport', {}).get('mlDataExportId') or event.get('mlDataExportId')
        
        logger.info(f"Extracted parameters - mlDataExportId: {ml_data_export_id}")

    except Exception as e:
        if e.args and e.args[0].startswith('{'):
            raise e
        error_info = format_error('VALIDATION', f"Error parsing configurations: {str(e)}", 'get_ml_data_export')
        logger.error(json.dumps(error_info))
        raise ValueError(json.dumps(error_info))
    
    try:
        # Get the ML data export details
        secret_name = os.environ.get('SECRETS_NAME')
        
        logger.info("Retrieving ML data export details")
        ml_data_exports_instance = acm_api.MlDataExports(
            secret_name=secret_name, 
            ACCOUNT_ID=account_id, 
            INSTANCE_ID=instance_id
        )
        
        response = ml_data_exports_instance.get_ml_data_export(ml_data_export_id)
        logger.info("ACM API response: %s", json.dumps(response, indent=2))
        
        # Extract key fields from response
        ml_data_export_details = {
            'mlDataExportId': response.get('mlDataExportId'),
            'createTime': response.get('createTime'),
            'status': response.get('status'),
            'statusReason': response.get('statusReason'),
            'validUntil': response.get('validUntil'),
            'preSignedS3Uris': response.get('preSignedS3Uris', []),
            'dataType': response.get('dataType'),
            'instanceId': response.get('instanceId'),
            'trainedModelId': response.get('trainedModelId')
        }
        
        # Add error-specific fields if status indicates failure
        if response.get('status') in ['CREATE_FAILED', 'INACTIVE']:
            logger.info("ML data export is in failed state")
            error_info = format_error('API_ERROR', f"Error getting ml export: {response.get('statusReason')}", 'get_ml_export', 'Check AMC API credentials and model configuration')
            logger.error(json.dumps(error_info))
            raise ValueError(json.dumps(error_info))            
        
        # Return success response with ML data export details
        return {
            'statusCode': 200,
            'body': {
                'message': 'ML Data Export details retrieved successfully',
                'mlDataExport': ml_data_export_details
            }
        }
        
    except Exception as e:
        if e.args and e.args[0].startswith('{'):
            raise e
        error_info = format_error('API_ERROR', f"Error retrieving ML data export: {str(e)}", 'get_ml_data_export', 'Check ML data export ID and AMC API credentials')
        logger.error(json.dumps(error_info))
        raise ValueError(json.dumps(error_info))
