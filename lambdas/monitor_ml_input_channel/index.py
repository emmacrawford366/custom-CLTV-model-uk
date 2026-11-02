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
    Monitor the status of an ML Input Channel in Amazon Clean Rooms ML (ACM).
    
    This function polls an ML input channel to check if data extraction has completed,
    is still in progress, or has failed. ML input channels execute SQL queries against
    AMC data and prepare datasets for training or inference. The function is called
    repeatedly by Step Functions until the channel reaches a terminal state.
    
    Args:
        event (dict): Lambda event object containing:
            - mlInputChannelId (str): ID of the ML input channel to monitor
            - workflow_type (str): Type of workflow ('train' or 'inference')
            - account_id (str): The AMC account ID for API access
            - instance_id (str): The AMC instance ID
            - maxAttempts (int, optional): Maximum retry attempts (unused, for future use)
            - attempt (int, optional): Current attempt number (unused, for future use)
        context: Lambda context object (unused)
    
    Returns:
        dict: Response with status-dependent content:
        
            For ACTIVE (completed successfully):
            - statusCode: 200
            - body:
                - message: Success message
                - mlInputChannelId: The channel ID
                - status: 'ACTIVE'
                - workflow_type: The workflow type
                - channelDetails: Full channel details from API including:
                    - name: Channel name
                    - configuredModelAlgorithmAssociations: Associated models
                    - sqlQuery: The SQL query executed
                    - timeStart/timeEnd: Query time range
                    - recordCount: Number of records extracted
            
            For CREATE_IN_PROGRESS (still processing):
            - statusCode: 202
            - body:
                - message: In progress message
                - mlInputChannelId: The channel ID
                - status: 'CREATE_IN_PROGRESS'
                - workflow_type: The workflow type
            
            For CREATE_FAILED (extraction failed):
            - statusCode: 400
            - body:
                - message: Failure message with reason
                - mlInputChannelId: The channel ID
                - status: 'CREATE_FAILED'
                - workflow_type: The workflow type
                - error_info: Formatted error information
    
    Raises:
        ValueError: If validation fails with formatted error info
        Exception: If API call fails or unexpected status received, with formatted error info
        
    Environment Variables:
        SECRETS_NAME (str): Name of AWS Secrets Manager secret containing API credentials
    
    Note:
        - This function is designed to be called repeatedly (polling) until the channel
          reaches a terminal state. Step Functions handles the retry logic.
        - Common failures include SQL syntax errors or queries returning no data
        - SQL queries can be tested in AMC Query Editor before use
    """
    try:
        # Log received event
        logger.info("Received event: %s", json.dumps(event, indent=2))
        
        # Validate input
        if 'mlInputChannelId' not in event:
            error_info = format_error('VALIDATION', "Missing 'mlInputChannelId' field", 'monitor_ml_input_channel')
            raise ValueError(json.dumps(error_info))
            
        if 'workflow_type' not in event:
            error_info = format_error('VALIDATION', "Missing 'workflow_type' field", 'monitor_ml_input_channel')
            raise ValueError(json.dumps(error_info))

        if 'account_id' not in event:
            error_info = format_error('VALIDATION', "Missing 'account_id' field", 'monitor_ml_input_channel')
            raise ValueError(json.dumps(error_info))
        
        if 'instance_id' not in event:
            error_info = format_error('VALIDATION', "Missing 'instance_id' field", 'monitor_ml_input_channel')
            raise ValueError(json.dumps(error_info))
        
        ml_input_channel_id = event['mlInputChannelId']
        max_attempts = event.get('maxAttempts', 3)
        current_attempt = event.get('attempt', 1)
        
        secret_name = os.environ.get('SECRETS_NAME')
        
        # Initialize handler
        ml_handler = acm_api.MlInputChannel(
            secret_name=secret_name,
            ACCOUNT_ID=event['account_id'],
            INSTANCE_ID=event['instance_id']
        )
        
        # Get channel status
        channel_details = ml_handler.get_ml_input_channel(ml_input_channel_id)

        logger.info(f"ML Input Channel details: {channel_details}")
        status = channel_details.get('status')
        statusReason = channel_details.get('statusReason')
        
        logger.info(f"ML Input Channel status: {status}")
        
        if status == 'ACTIVE':
            # Channel is ready
            return {
                'statusCode': 200,
                'body': {
                    'message': 'ML Input Channel is active',
                    'mlInputChannelId': ml_input_channel_id,
                    'status': 'ACTIVE',
                    'workflow_type': event['workflow_type'],
                    'channelDetails': channel_details
                }
            }
        elif status == 'CREATE_IN_PROGRESS' or status == 'QUEUED':
            return {
                'statusCode': 202,
                'body': {
                    'message': 'ML Input Channel is still being created',
                    'mlInputChannelId': ml_input_channel_id,
                    'status': 'CREATE_IN_PROGRESS',
                    'workflow_type': event['workflow_type']
                }
            }
        elif status == 'CREATE_FAILED':
            # Channel creation failed
            error_info = format_error('MLIC API ERROR', f"ML Input Channel creation failed: {statusReason}", 'monitor_ml_input_channel', 'Check SQL query syntax in AMC Query Editor')
            return {
                'statusCode': 400,
                'body': {
                    'message': f'Create Failed for ML Input Channel. Error message from API: {statusReason}',
                    'mlInputChannelId': ml_input_channel_id,
                    'status': 'CREATE_FAILED',
                    'workflow_type': event['workflow_type'],
                    'error_info': error_info
                }
            }
        else:
            error_info = format_error('API_ERROR', f"Unexpected ML Input Channel status: {status}", 'monitor_ml_input_channel')
            raise Exception(json.dumps(error_info))
            
    except Exception as e:
        if e.args and e.args[0].startswith('{'):
            raise e
        error_info = format_error('INFRASTRUCTURE', f"Lambda execution failed: {str(e)}", 'monitor_ml_input_channel')
        logger.error(json.dumps(error_info))
        raise Exception(json.dumps(error_info))
