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

def is_matching_mlic(mlic: dict, model_arn: str, sql_query: str, time_start: str, time_end: str) -> bool:
    """
    Check if an existing MLIC matches the requested parameters.

    Args:
        mlic (dict): The MLIC object from the API containing:
            - configuredModelAlgorithmAssociations: List of model ARNs
            - queryParameters: Dict with amcQuery, timeWindowStart, timeWindowEnd
        model_arn (str): The model ARN to match against
        sql_query (str): The SQL query to match against
        time_start (str): The time window start to match against
        time_end (str): The time window end to match against

    Returns:
        bool: True if all parameters match, False otherwise
    """
    mlic_params = mlic.get('queryParameters', {})
    mlic_associations = mlic.get('configuredModelAlgorithmAssociations', [])

    return (
        mlic_associations == [model_arn] and
        mlic_params.get('amcQuery') == sql_query and
        mlic_params.get('timeWindowStart') == time_start and
        mlic_params.get('timeWindowEnd') == time_end
    )

def check_existing_mlics(ml_handler, model_arn: str, sql_query: str, time_start: str, time_end: str) -> dict:
    """
    Check if an MLIC already exists with matching parameters.

    Searches through ACTIVE and CREATE_IN_PROGRESS MLICs to find a match.
    Prioritizes ACTIVE MLICs over IN_PROGRESS ones, and newer MLICs over older ones.

    Args:
        ml_handler: The MlInputChannel handler instance
        model_arn (str): The configured model algorithm association ARN
        sql_query (str): The SQL query for the MLIC
        time_start (str): The time window start in ISO format
        time_end (str): The time window end in ISO format

    Returns:
        dict: Result containing:
            - exists (bool): Whether a matching MLIC was found
            - mlic_id (str, optional): The ID of the matching MLIC if found
            - status (str, optional): The status of the matching MLIC ('ACTIVE' or 'CREATE_IN_PROGRESS')
    """
    try:
        # Get all ACTIVE MLICs
        active_mlics = ml_handler.list_all_ml_input_channels(status='ACTIVE')
        logger.info(f"Found {len(active_mlics)} ACTIVE MLICs")

        # Get all IN_PROGRESS MLICs
        in_progress_mlics = ml_handler.list_all_ml_input_channels(status='CREATE_IN_PROGRESS')
        logger.info(f"Found {len(in_progress_mlics)} CREATE_IN_PROGRESS MLICs")

        # Combine and sort: ACTIVE first, then by creation time (newest first)
        all_mlics = active_mlics + in_progress_mlics
        sorted_mlics = sorted(
            all_mlics,
            key=lambda x: (
                0 if x.get('status') == 'ACTIVE' else 1,
                -datetime.fromisoformat(
                    x.get('createTime', '1970-01-01T00:00:00Z').replace('Z', '+00:00').split('.')[0]
                ).timestamp()
            )
        )

        # Check each MLIC for a match
        for mlic in sorted_mlics:
            if is_matching_mlic(mlic, model_arn, sql_query, time_start, time_end):
                logger.info(f"Found matching MLIC: {mlic['mlInputChannelId']} with status {mlic['status']}")
                return {
                    'exists': True,
                    'mlic_id': mlic['mlInputChannelId'],
                    'status': mlic['status']
                }

        logger.info("No matching MLIC found")
        return {'exists': False}

    except Exception as e:
        logger.warning(f"Error checking existing MLICs: {str(e)}. Proceeding to create new MLIC.")
        return {'exists': False}

def lambda_handler(event, context):
    """
    Create an ML Input Channel in Amazon Clean Rooms ML (ACM) for training or inference.

    Before creating a new MLIC, this function checks if an existing MLIC with the same
    parameters (model ARN, SQL query, time window) already exists. If found in ACTIVE
    status, the existing MLIC ID is returned instead of creating a new one.

    An ML Input Channel represents a dataset that will be used as input for model
    training or inference. It's created by executing a SQL query against AMC data
    within a specified time range.

    Args:
        event (dict): Lambda event object containing:
            - channel_data (dict): Channel configuration with:
                - name (str): Name for the ML input channel
                - sql_query (str): SQL query to extract data from AMC
                - time_start (str): Start time in ISO format (e.g., "2024-05-01T00:00:00Z")
                - time_end (str): End time in ISO format
                - user_level_targeting (bool, optional): Enable user-level targeting. Defaults to True
            - workflow_type (str): Type of workflow ('train' or 'inference')
            - account_id (str): The AMC account ID for API access
            - instance_id (str): The AMC instance ID
            - model_arn (str): ARN of the configured model algorithm
        context: Lambda context object (unused)

    Returns:
        dict: Response containing:
            - statusCode (int): HTTP status code (200 for success)
            - body (dict): Response body with:
                - message (str): Success message
                - mlInputChannelId (str): Created or existing ML input channel ID
                - workflow_type (str): The workflow type
                - reused_existing (bool): Whether an existing MLIC was reused

    Raises:
        ValueError: If validation fails with formatted error info
        Exception: If API call fails with formatted error info

    Environment Variables:
        SECRETS_NAME (str): Name of AWS Secrets Manager secret containing API credentials

    Note:
        The SQL query must be valid for the AMC instance and the time range must
        contain data. The channel will be processed asynchronously by ACM.
    """
    try:
        # Log received event
        logger.info("Received event: %s", json.dumps(event, indent=2))

        # Validate input
        if 'channel_data' not in event:
            error_info = format_error('VALIDATION', "Missing 'channel_data' field", 'create_ml_input_channel')
            raise ValueError(json.dumps(error_info))

        if 'workflow_type' not in event:
            error_info = format_error('VALIDATION', "Missing 'workflow_type' field", 'create_ml_input_channel')
            raise ValueError(json.dumps(error_info))

        if 'account_id' not in event:
            error_info = format_error('VALIDATION', "Missing 'account_id' field", 'create_ml_input_channel')
            raise ValueError(json.dumps(error_info))

        if 'instance_id' not in event:
            error_info = format_error('VALIDATION', "Missing 'instance_id' field", 'create_ml_input_channel')
            raise ValueError(json.dumps(error_info))

        if 'model_arn' not in event:
            error_info = format_error('VALIDATION', "Missing 'model_arn' field", 'create_ml_input_channel')
            raise ValueError(json.dumps(error_info))

        # Extract and validate channel data
        channel_data = event['channel_data']
        required_fields = ['name', 'sql_query', 'time_start', 'time_end']

        for field in required_fields:
            if field not in channel_data:
                error_info = format_error('VALIDATION', f"Missing required field in channel_data: '{field}'", 'create_ml_input_channel', 'Check your input channel configuration')
                raise ValueError(json.dumps(error_info))

        logger.info("Input validation successful")

        # Get secret name from environment variable
        secret_name = os.environ.get('SECRETS_NAME')

        # Initialize handler
        ml_handler = acm_api.MlInputChannel(secret_name=secret_name, ACCOUNT_ID=event['account_id'], INSTANCE_ID=event['instance_id'])
        logger.info("ML Input Channel Handler initialized")

        # Check for existing MLIC with matching parameters
        logger.info("Checking for existing MLIC with matching parameters...")
        existing_check = check_existing_mlics(
            ml_handler=ml_handler,
            model_arn=event['model_arn'],
            sql_query=channel_data['sql_query'],
            time_start=channel_data['time_start'],
            time_end=channel_data['time_end']
        )

        # If an ACTIVE MLIC exists with matching parameters, return it
        if existing_check['exists'] and existing_check['status'] == 'ACTIVE':
            logger.info(f"Reusing existing ACTIVE MLIC: {existing_check['mlic_id']}")
            return {
                'statusCode': 200,
                'body': {
                    'message': 'Reusing existing ML Input Channel with matching parameters',
                    'mlInputChannelId': existing_check['mlic_id'],
                    'workflow_type': event['workflow_type'],
                    'reused_existing': True
                }
            }

        # If an IN_PROGRESS MLIC exists, we can also return it (the monitoring will wait for it)
        if existing_check['exists'] and existing_check['status'] == 'CREATE_IN_PROGRESS':
            logger.info(f"Found existing IN_PROGRESS MLIC: {existing_check['mlic_id']}")
            return {
                'statusCode': 200,
                'body': {
                    'message': 'Found existing ML Input Channel in progress with matching parameters',
                    'mlInputChannelId': existing_check['mlic_id'],
                    'workflow_type': event['workflow_type'],
                    'reused_existing': True
                }
            }

        # No existing MLIC found, create a new one
        logger.info("No existing MLIC found, creating new ML Input Channel...")
        response = ml_handler.create_ml_input_channel(
            name=channel_data['name'],
            model_arn=[event['model_arn']],
            sql_query=channel_data['sql_query'],
            time_start=channel_data['time_start'],
            time_end=channel_data['time_end'],
            user_level_targeting=channel_data.get('user_level_targeting', True),
            verbose=False
        )

        ml_input_channel_id = response.get('mlInputChannelId')
        logger.info(f"Successfully created ML Input Channel: {ml_input_channel_id}")

        # Return success response with needed data for Step Functions
        return {
            'statusCode': 200,
            'body': {
                'message': 'ML Input Channel creation initiated',
                'mlInputChannelId': ml_input_channel_id,
                'workflow_type': event['workflow_type'],
                'reused_existing': False
            }
        }
        
    except ValueError as ve:
        if ve.args and ve.args[0].startswith('{'):
            raise ve
        error_info = format_error('VALIDATION', f"Validation error: {str(ve)}", 'create_ml_input_channel')
        logger.error(json.dumps(error_info))
        raise Exception(json.dumps(error_info))
        
    except Exception as e:
        if e.args and e.args[0].startswith('{'):
            raise e
        error_info = format_error('API_ERROR', f"Lambda execution failed: {str(e)}", 'create_ml_input_channel', 'Check AMC API credentials and SQL query syntax')
        logger.error(json.dumps(error_info))
        raise Exception(json.dumps(error_info))
