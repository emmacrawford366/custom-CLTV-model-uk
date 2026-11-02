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
    Create an ML Data Export for a trained model using the Amazon Clean Rooms ML (ACM) API.
    
    This function creates an export of ML data (measurement metrics) from a completed
    training job. The export contains model performance metrics and is typically used
    for Marketing Mix Modeling (MMM) measurement use cases.
    
    Args:
        event (dict): Lambda event object containing:
            - account_id (str): The AMC account ID for API access
            - instance_id (str): The AMC instance ID
            - data_type (str): Type of data to export (e.g., 'MMM' for measurement)
            - training_job_monitoring (dict): Nested structure containing:
                - training_job_monitoring_result (dict): Training job results with:
                    - trainedModelId (str): ID of the trained model to export data from
        context: Lambda context object (unused)
    
    Returns:
        dict: Response containing:
            - statusCode (int): HTTP status code (200 for success)
            - body (dict): Response body with:
                - message (str): Success message
                - mlDataExportId (str): Created ML data export ID
                - data_type (str): The data type that was exported
    
    Raises:
        ValueError: If validation fails or API call fails, with formatted error info
        
    Environment Variables:
        SECRETS_NAME (str): Name of AWS Secrets Manager secret containing API credentials
    
    Note:
        This function is specifically designed for measurement (MMM) use cases where
        model performance metrics need to be exported for analysis.
    """
    try:
        # Log received event
        logger.info("Received event: %s", json.dumps(event, indent=2))
        
        # Validate required input fields
        required_fields = ['account_id', 'instance_id', 'data_type']
        for field in required_fields:
            if field not in event:
                error_info = format_error('VALIDATION', f"Missing required field: '{field}'", 'create_ml_data_export', 'Ensure all required fields are provided in the payload')
                raise ValueError(json.dumps(error_info))
        
        # Extract trainedModelId from the nested structure
        trained_model_id = event.get('training_job_monitoring', {}).get('training_job_monitoring_result', {}).get('trainedModelId')
        if not trained_model_id:
            error_info = format_error('VALIDATION', "trainedModelId not found in training_job_monitoring.training_job_monitoring_result", 'create_ml_data_export')
            raise ValueError(json.dumps(error_info))
    
        
        logger.info(f"Data type for measurement is: {event['data_type']}")
        
        # Extract needed values from event
        account_id = event['account_id']
        instance_id = event['instance_id']
        data_type = event['data_type']
        
        logger.info("Extracted parameters successfully")

    except Exception as e:
        if e.args and e.args[0].startswith('{'):
            raise e
        error_info = format_error('VALIDATION', f"Error parsing configurations: {str(e)}", 'create_ml_data_export')
        logger.error(json.dumps(error_info))
        raise ValueError(json.dumps(error_info))
    
    try:
        # Create the ML data export
        secret_name = os.environ.get('SECRETS_NAME')
        
        logger.info("Creating ML data export")
        ml_data_exports_instance = acm_api.MlDataExports(
            secret_name=secret_name, 
            ACCOUNT_ID=account_id, 
            INSTANCE_ID=instance_id
        )
        
        # Prepare payload for API call
        api_payload = {
            'data_type': data_type,
            'trained_model_id': trained_model_id
        }
        
        logger.info(f"Creating ML data export with parameters: {json.dumps(api_payload, default=str)}")
        
        response = ml_data_exports_instance.create_ml_data_export(**api_payload)
        logger.info("ACM API response: %s", json.dumps(response, indent=2))
        
        ml_data_export_id = response.get('mlDataExportId')
        
        # Return success response with needed data for Step Functions
        return {
            'statusCode': 200,
            'body': {
                'message': 'ML Data Export creation initiated',
                'mlDataExportId': ml_data_export_id,
                'data_type': data_type
            }
        }
        
    except Exception as e:
        if e.args and e.args[0].startswith('{'):
            raise e
        error_info = format_error('API_ERROR', f"Error creating ML data export: {str(e)}", 'create_ml_data_export', 'Check AMC API credentials and parameters')
        logger.error(json.dumps(error_info))
        raise ValueError(json.dumps(error_info))
