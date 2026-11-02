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
    Create an Inference Job using the Amazon Clean Rooms ML (ACM) API.
    
    This function orchestrates the creation of a trained model inference job by:
    1. Validating required input parameters
    2. Extracting ML input channel information from processed channels
    3. Creating the inference job via ACM API
    
    Args:
        event (dict): Lambda event object containing:
            - workflow_type (str): Type of workflow, must be 'inference'
            - account_id (str): The AMC account ID for API access
            - instance_id (str): The AMC instance ID
            - configured_model_algorithm_associations (str): Model ARN to use
            - inference_parameters (dict): Parameters for the inference job including:
                - name (str): Name for the inference job
                - trained_model_id (str): ID of the trained model to use
                - instance_type (str): EC2 instance type for inference
            - processed_channels (list): List of ML input channel objects with:
                - monitoring.monitoring_result.mlInputChannelId (str): Channel ID
                - monitoring.monitoring_result.channelDetails.name (str): Channel name
        context: Lambda context object (unused)
    
    Returns:
        dict: Response containing:
            - statusCode (int): HTTP status code (200 for success)
            - body (dict): Response body with:
                - message (str): Success message
                - trainedModelInferenceJobId (str): Created inference job ID
                - workflow_type (str): The workflow type
    
    Raises:
        ValueError: If validation fails or API call fails, with formatted error info
        
    Environment Variables:
        SECRETS_NAME (str): Name of AWS Secrets Manager secret containing API credentials
    """
    try:
        # Log received event
        logger.info("Received event: %s", json.dumps(event, indent=2))
        
        # Validate required input fields
        required_fields = ['workflow_type', 'account_id', 'configured_model_algorithm_associations', 'instance_id', 
                         'inference_parameters', 'processed_channels']
        for field in required_fields:
            if field not in event:
                error_info = format_error('VALIDATION', f"Missing required field: '{field}'", 'create_inference_job')
                raise ValueError(json.dumps(error_info))
        
        # Extract needed values from event
        account_id = event['account_id']
        instance_id = event['instance_id']
        inference_params = event['inference_parameters']
        processed_channels = event['processed_channels']
        model_arn = event['configured_model_algorithm_associations']
        
        # Validate inference parameters
        required_inference_params = [
            'name',
            'trained_model_id',
            'instance_type'
        ]
        
        missing_params = [param for param in required_inference_params 
                         if param not in inference_params]
        if missing_params:
            error_info = format_error('VALIDATION', f"Missing required inference parameter(s): {', '.join(missing_params)}", 'create_inference_job')
            raise ValueError(json.dumps(error_info))
            
        logger.info(f"Inference parameters: {inference_params}")

    except Exception as e:
        if e.args and e.args[0].startswith('{'):
            raise e
        error_info = format_error('VALIDATION', f"Error parsing configurations: {str(e)}", 'create_inference_job')
        logger.error(json.dumps(error_info))
        raise ValueError(json.dumps(error_info))
    
    # Extract ML Input Channel IDs and prepare input channels
    try:
        input_channels = []
        
        for channel in processed_channels:    
            # Check for MInputChannelId
            ml_input_channel_id = channel.get('monitoring', {}).get('monitoring_result', {}).get('mlInputChannelId')
            
            if not ml_input_channel_id:
                error_info = format_error('VALIDATION', f"Missing mlInputChannelId in channel: {channel}", 'create_inference_job')
                raise ValueError(json.dumps(error_info))

            logger.info(f"Found mlInputChannelId: {ml_input_channel_id}")
            
            # Get channel name
            channel_name = channel.get('monitoring', {}).get('monitoring_result', {}).get('channelDetails', {}).get('name')
            
            if not channel_name:
                error_info = format_error('VALIDATION', f"Missing channel name in channel: {channel}", 'create_inference_job')
                logger.error(json.dumps(error_info))
                raise ValueError(json.dumps(error_info))
            
            logger.info(f"Found channel name: {channel_name}")
            
            # Add to input channels list
            input_channels.append({
                "mlInputChannelId": ml_input_channel_id,
                "channelName": channel_name
            })
    
        logger.info(f"Prepared {len(input_channels)} input channels")
        logger.info(f"Input channels: {input_channels}")    

        mlInputChannelId = input_channels[0]['mlInputChannelId']
        logger.info(f"mlInputChannelId: {mlInputChannelId}")
        
    except Exception as e:
        if e.args and e.args[0].startswith('{'):
            raise e
        error_info = format_error('VALIDATION', f"Error preparing input channels: {str(e)}", 'create_inference_job')
        logger.error(json.dumps(error_info))
        raise ValueError(json.dumps(error_info))
    
    try:
        # Create the inference job
        secret_name = os.environ.get('SECRETS_NAME')
        
        logger.info("Creating inference job")
        inference_job_instance = acm_api.TrainedModelInferenceJobs(
            secret_name=secret_name, 
            ACCOUNT_ID=account_id, 
            INSTANCE_ID=instance_id
        )
        
        logger.info(f"Creating inference job with parameters: {json.dumps(inference_params, default=str)}")
        response = inference_job_instance.create_inference_job(model_arn = model_arn, input_channel_id = mlInputChannelId, **inference_params)
        logger.info("ACM API response: %s", json.dumps(response, indent=2))
        
        # Return success response with needed data for Step Functions
        return {
            'statusCode': 200,
            'body': {
                'message': 'Inference Job creation initiated',
                'trainedModelInferenceJobId': response['trainedModelInferenceJobId'],
                'workflow_type': event['workflow_type']
            }
        }
        
    except Exception as e:
        if e.args and e.args[0].startswith('{'):
            raise e
        error_info = format_error('API_ERROR', f"Error creating inference job: {str(e)}", 'create_inference_job', 'Check trained model ID and AMC API credentials')
        logger.error(json.dumps(error_info))
        raise ValueError(json.dumps(error_info))
