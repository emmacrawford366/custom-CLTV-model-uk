import json
import logging
import acm_api
import time
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
    Monitor the status of a Trained Model (training job) in Amazon Clean Rooms ML (ACM).
    
    This function polls a training job to check if model training has completed, is still
    in progress, or has failed. Training jobs execute ML algorithms on prepared data from
    ML input channels. The function is called repeatedly by Step Functions until the job
    reaches a terminal state.
    
    Args:
        event (dict): Lambda event object containing:
            - trainedModelId (str): ID of the trained model/training job to monitor
            - workflow_type (str): Type of workflow, must be 'train'
            - account_id (str): The AMC account ID for API access
            - instance_id (str): The AMC instance ID
        context: Lambda context object (unused)
    
    Returns:
        dict: Response with status-dependent content:
        
            For ACTIVE (completed successfully):
            - statusCode: 200
            - body:
                - message: Success message
                - trainedModelId: The training job ID
                - status: 'ACTIVE'
                - workflow_type: 'train'
                - modelDetails: Full model details from API including:
                    - name: Model name
                    - configuredModelAlgorithmAssociation: Algorithm used
                    - trainingDataStartTime/trainingDataEndTime: Data time range
                    - createTime: When training started
                    - completionTime: When training completed
                    - metrics: Training performance metrics
            
            For CREATE_IN_PROGRESS (still training):
            - statusCode: 202
            - body:
                - message: In progress message
                - trainedModelId: The training job ID
                - status: 'CREATE_IN_PROGRESS'
                - workflow_type: 'train'
            
            For CREATE_FAILED (training failed):
            - statusCode: 400
            - body:
                - message: Failure message with reason
                - trainedModelId: The training job ID
                - status: 'CREATE_FAILED'
                - workflow_type: 'train'
                - error_info: Formatted error information
    
    Raises:
        ValueError: If validation fails with formatted error info
        Exception: If API call fails or unexpected status received, with formatted error info
        
    Environment Variables:
        SECRETS_NAME (str): Name of AWS Secrets Manager secret containing API credentials
    
    Note:
        - This function is designed to be called repeatedly (polling) until the training
          job reaches a terminal state. Step Functions handles the retry logic.
        - Training can take significant time depending on data volume and algorithm
        - Common failures include insufficient data, incorrect parameters, or algorithm issues
        - Error logs are available via the download_url lambda when training fails
    """
    try:
        # Log received event
        logger.info("Received event: %s", json.dumps(event, indent=2))
        
        # Validate input
        if 'trainedModelId' not in event:
            error_info = format_error('VALIDATION', "Missing 'trainedModelId' field", 'monitor_training', 'Ensure trainedModelId is passed from previous step')
            raise ValueError(json.dumps(error_info))
            
        if 'workflow_type' not in event:
            error_info = format_error('VALIDATION', "Missing 'workflow_type' field", 'monitor_training')
            raise ValueError(json.dumps(error_info))

        if 'account_id' not in event:
            error_info = format_error('VALIDATION', "Missing 'account_id' field", 'monitor_training')
            raise ValueError(json.dumps(error_info))
        
        if 'instance_id' not in event:
            error_info = format_error('VALIDATION', "Missing 'instance_id' field", 'monitor_training')
            raise ValueError(json.dumps(error_info))
        
        trained_model_id = event['trainedModelId']
        logger.info(f"TrainedModel ID: {trained_model_id}")
        
        secret_name = os.environ.get('SECRETS_NAME')
        
        # Initialize handler
        trained_model_handler = acm_api.TrainedModels(
            secret_name=secret_name,
            ACCOUNT_ID=event['account_id'],
            INSTANCE_ID=event['instance_id']
        )
        
        # Get training status
        model_details = trained_model_handler.get_trained_model(trained_model_id)
        logger.info(f"Model details: {model_details}")
        status = model_details.get('status')
        status_reason = model_details.get('statusReason')
        
        logger.info(f"TrainedModel status: {status}")
        
        if status == 'ACTIVE':
            logger.info("Training completed successfully")
            # Training completed successfully
            return {
                'statusCode': 200,
                'body': {
                    'message': 'Training completed successfully',
                    'trainedModelId': trained_model_id,
                    'status': 'ACTIVE',
                    'workflow_type': event['workflow_type'],
                    'modelDetails': model_details
                }
            }
        elif status == 'CREATE_IN_PROGRESS' or status == 'QUEUED':
            logger.info("Training still in progress or queued")
            # Training still in progress or queued
            return {
                'statusCode': 202,
                'body': {
                    'message': 'Model training is still in progress',
                    'trainedModelId': trained_model_id,
                    'status': 'CREATE_IN_PROGRESS',
                    'workflow_type': event['workflow_type']
                }
            }
        elif status == 'CREATE_FAILED':
            # Training failed
            logger.error(f"Training failed. Error message from API: {status_reason}")
            error_info = format_error('MODEL TRAINING', f"Training failed: {status_reason}", 'monitor_training', 'Check training parameters and data quality')
            return {
                'statusCode': 400,
                'body': {
                    'message': f"Training failed. Error message from API: {status_reason}",
                    'trainedModelId': trained_model_id,
                    'status': 'CREATE_FAILED',
                    'workflow_type': event['workflow_type'],
                    'error_info': error_info
                }
            }
        else:
            error_info = format_error('API_ERROR', f"Unexpected training status: {status}", 'monitor_training')
            raise Exception(json.dumps(error_info))
            
    except Exception as e:
        if e.args and e.args[0].startswith('{'):
            # Already formatted error
            raise e
        error_info = format_error('INFRASTRUCTURE', f"Lambda execution failed: {str(e)}", 'monitor_training')
        logger.error(json.dumps(error_info))
        raise Exception(json.dumps(error_info))
