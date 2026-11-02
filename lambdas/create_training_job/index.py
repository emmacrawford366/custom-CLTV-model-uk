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
    Create a Trained Model (training job) in Amazon Clean Rooms ML (ACM).
    
    This function orchestrates the creation of a model training job by:
    1. Validating required input parameters and training configuration
    2. Extracting ML input channel IDs from processed channels
    3. Creating the training job via ACM API with specified parameters
    
    Args:
        event (dict): Lambda event object containing:
            - workflow_type (str): Type of workflow, must be 'train'
            - account_id (str): The AMC account ID for API access
            - instance_id (str): The AMC instance ID
            - configured_model_algorithm_associations (str): Model ARN to use for training
            - training_parameters (dict): Training configuration including:
                - name (str): Name for the training job (required)
                - instance_type (str): EC2 instance type for training (required)
                - instance_count (int, optional): Number of instances
                - vol_size (int, optional): Volume size in GB
                - timeout (int, optional): Training timeout in seconds
                - description (str, optional): Training job description
            - processed_channels (list): List of ML input channel objects with:
                - monitoring.monitoring_result.mlInputChannelId (str): Channel ID
                - monitoring.monitoring_result.channelDetails.name (str): Channel name
        context: Lambda context object (unused)
    
    Returns:
        dict: Response containing:
            - statusCode (int): HTTP status code (200 for success)
            - body (dict): Response body with:
                - message (str): Success message
                - trainedModelId (str): Created training job ID
                - workflow_type (str): The workflow type
    
    Raises:
        ValueError: If validation fails or API call fails, with formatted error info
        
    Environment Variables:
        SECRETS_NAME (str): Name of AWS Secrets Manager secret containing API credentials
    
    Note:
        Training jobs are executed asynchronously. The trainedModelId should be used
        to monitor training progress via the monitor_training lambda.
    """
    try:
        # Log received event
        logger.info("Received event: %s", json.dumps(event, indent=2))
        
        # Validate required input fields
        required_fields = ['workflow_type', 'account_id', 'instance_id', 
                           'configured_model_algorithm_associations', 'processed_channels']
        for field in required_fields:
            if field not in event:
                error_info = format_error('VALIDATION', f"Missing required field: '{field}'", 'create_training_job', 'Ensure all required fields are provided in the payload')
                raise ValueError(json.dumps(error_info))
        
        # Extract needed values from event
        account_id = event['account_id']
        instance_id = event['instance_id']
        model_arn = event['configured_model_algorithm_associations']
        processed_channels = event['processed_channels']
        logger.info(f"Extracted top level parameters")

        ## Extract Training Parameters
        # Check if training_parameters exists in event
        if 'training_parameters' not in event:
            error_info = format_error('VALIDATION', "Missing required field: 'training_parameters'", 'create_training_job')
            logger.error(json.dumps(error_info))
            raise ValueError(json.dumps(error_info))
    
        training_params = event['training_parameters']

        required_params = ['name', 'instance_type']
    
        missing_params = [param for param in required_params if param not in training_params]
        if missing_params:
            error_info = format_error('VALIDATION', f"Missing required training parameter(s): {', '.join(missing_params)}", 'create_training_job')
            raise ValueError(json.dumps(error_info))
        
        logger.info(f"Training parameters: {training_params}")
    
    except Exception as e:
        if e.args and e.args[0].startswith('{'):
            raise e
        error_info = format_error('VALIDATION', f"Error parsing configurations: {str(e)}", 'create_training_job')
        logger.error(json.dumps(error_info))
        raise ValueError(json.dumps(error_info))
    
    
    # Extract ML Input Channel IDs and prepare input channels
    try:
        input_channels = []
        
        for channel in processed_channels:    

            # Check for MInputChannelId - Raise error if not found
            ml_input_channel_id = channel.get('monitoring', {}).get('monitoring_result', {}).get('mlInputChannelId')
            
            if not ml_input_channel_id:
                error_info = format_error('VALIDATION', f"Missing mlInputChannelId in channel: {channel}", 'create_training_job', 'Ensure ML Input Channels were created successfully')
                raise ValueError(json.dumps(error_info))

            logger.info(f"Found mlInputChannelId: {ml_input_channel_id}")
            
            # Get channel name - first try from channelDetails, then from channel_details
            channel_name = channel.get('monitoring', {}).get('monitoring_result', {}).get('channelDetails', {}).get('name')
            
            if not channel_name:
                error_info = format_error('VALIDATION', f"Missing channel name in channel: {channel}", 'create_training_job')
                logger.error(json.dumps(error_info))
                raise ValueError(json.dumps(error_info))
            
            logger.info(f"Found channel name: {channel_name}")
            
            # If still no name, use a default with the index
            if not channel_name:
                channel_name = f"input_channel_{len(input_channels)}"
            
            # Add to input channels list
            input_channels.append({
                "mlInputChannelId": ml_input_channel_id,
                "channelName": channel_name
            })
    
        logger.info(f"Prepared {len(input_channels)} input channels")
        logger.info(f"Input channels: {input_channels}")    
    except Exception as e:
        if e.args and e.args[0].startswith('{'):
            raise e
        error_info = format_error('VALIDATION', f"Error preparing input channels: {str(e)}", 'create_training_job')
        logger.error(json.dumps(error_info))
        raise ValueError(json.dumps(error_info))
    
    try:
        # Create the trained model
        secret_name = os.environ.get('SECRETS_NAME')


        logger.info("Creating trained model")
        trained_models_instance = acm_api.TrainedModels(secret_name=secret_name, ACCOUNT_ID=account_id, INSTANCE_ID=instance_id)

        logger.info(f"Creating trained model with parameters: {json.dumps(training_params, default=str)}")
        logger.info(f"Using Input Channels: {input_channels}")
        response = trained_models_instance.create_trained_model(
            model_arn = model_arn, input_channels = input_channels, **training_params)
        logger.info("ACM API response: %s", json.dumps(response, indent=2))

        trainedModelId = response.get('trainedModelId')

        # Return the response from the ACM API
                # Return success response with needed data for Step Functions
        return {
            'statusCode': 200,
            'body': {
                'message': 'Training Job creation initiated',
                'trainedModelId': trainedModelId,
                'workflow_type': event['workflow_type']
            }
        }
    except Exception as e:
        if e.args and e.args[0].startswith('{'):
            raise e
        error_info = format_error('API_ERROR', f"Error creating trained model: {str(e)}", 'create_training_job', 'Check AMC API credentials and model configuration')
        logger.error(json.dumps(error_info))
        raise ValueError(json.dumps(error_info))
