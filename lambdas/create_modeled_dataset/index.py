import json
import logging
import acm_api
import os
import re
import boto3
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
    Create or retrieve a Modeled Dataset in Amazon Clean Rooms ML (ACM) for inference results.
    
    A Modeled Dataset is a destination for inference job outputs in ACM. This function
    checks if a dataset already exists for the algorithm, reusing it if found, or creates
    a new one with the provided schema. The dataset ID is persisted in DynamoDB for reuse.
    
    Args:
        event (dict): Lambda event object containing:
            - account_id (str): The AMC account ID for API access
            - instance_id (str): The AMC instance ID
            - acm_algorithm_name (str): Name of the algorithm (used for dataset naming and lookup)
            - inferenceJob (dict): Inference job information with:
                - trainedModelInferenceJobId (str): ID of the completed inference job
            - schema (dict): Schema definition for the dataset structure
        context: Lambda context object (unused)
    
    Returns:
        dict: Response containing:
            - statusCode (int): HTTP status code (200 for success)
            - body (dict): Response body with:
                - message (str): Success message
                - modeledDatasetId (str): Dataset ID (existing or newly created)
                - inferenceJobId (str): The inference job ID
    
    Raises:
        ValueError: If validation fails or API call fails, with formatted error info
        
    Environment Variables:
        SECRETS_NAME (str): Name of AWS Secrets Manager secret containing API credentials
        DYNAMODB_TABLE_CONFIGS (str): Name of DynamoDB table for storing dataset IDs
    
    Note:
        Dataset names are normalized to lowercase with underscores and stored in DynamoDB
        to enable reuse across multiple inference jobs for the same algorithm.
    """
    try:
        logger.info("Received event: %s", json.dumps(event, indent=2))
        
        # Validate required input fields
        required_fields = ['account_id', 'instance_id', 'inferenceJob', 'acm_algorithm_name']
        for field in required_fields:
            if field not in event:
                error_info = format_error('VALIDATION', f"Missing required field: '{field}'", 'create_modeled_dataset')
                raise ValueError(json.dumps(error_info))
        
        # Validate schema field
        if 'schema' not in event:
            error_info = format_error('VALIDATION', "Missing required field: 'schema'", 'create_modeled_dataset', 'Ensure schema is provided in the payload')
            raise ValueError(json.dumps(error_info))
        
        # Extract needed values from event
        account_id = event['account_id']
        instance_id = event['instance_id']
        inference_job_id = event['inferenceJob']['trainedModelInferenceJobId']
        acm_algorithm_name = event['acm_algorithm_name']
        
        # Check DynamoDB for existing modeledDatasetId
        dynamodb = boto3.resource('dynamodb')
        table_name = os.environ.get('DYNAMODB_TABLE_CONFIGS')
        table = dynamodb.Table(table_name)
        
        try:
            response = table.get_item(
                Key={
                    'acm_algorithm_name': acm_algorithm_name,
                    'instance_id': instance_id
                }
            )
            
            if 'Item' in response and 'modeledDatasetId' in response['Item']:
                logger.info(f"Found existing modeledDatasetId: {response['Item']['modeledDatasetId']}")
                return {
                    'statusCode': 200,
                    'body': {
                        'message': 'Retrieved existing modeled dataset',
                        'modeledDatasetId': response['Item']['modeledDatasetId'],
                        'inferenceJobId': inference_job_id
                    }
                }
        except Exception as e:
            logger.warning(f"Error checking DynamoDB: {str(e)}. Proceeding to create new dataset.")
        
        # Get schema from event payload
        schema = event['schema']
        logger.info(f"Using schema: {json.dumps(schema, indent=2)}")
        
        # Create dataset name based on algorithm name (normalized for AMC compliance)
        dataset_name = f"modeled_dataset_{acm_algorithm_name}".lower()
        dataset_name = re.sub(r'[^a-z0-9_]', '_', dataset_name)
        logger.info(f"AMC dataset compliant name is: {dataset_name}")
        
        logger.info(f"Creating a new modeled dataset: {dataset_name}")
        
        # Create the modeled dataset
        secret_name = os.environ.get('SECRETS_NAME')
        modeled_dataset_handler = acm_api.ManageModeledDatasets(
            secret_name=secret_name, 
            ACCOUNT_ID=account_id, 
            INSTANCE_ID=instance_id
        )
        
        response = modeled_dataset_handler.create_dataset(
            name=dataset_name,
            schema=schema,
            description=f"Modeled dataset for inference job {inference_job_id}"
        )
        
        logger.info("ACM API response: %s", json.dumps(response, indent=2))
        
        # Save modeledDatasetId to DynamoDB
        modeled_dataset_id = response.get('modeledDatasetId')
        if modeled_dataset_id:
            try:
                table.update_item(
                    Key={
                        'acm_algorithm_name': acm_algorithm_name,
                        'instance_id': instance_id
                    },
                    UpdateExpression='SET modeledDatasetId = :val',
                    ExpressionAttributeValues={
                        ':val': modeled_dataset_id
                    }
                )
                logger.info(f"Saved modeledDatasetId to DynamoDB: {modeled_dataset_id}")
            except Exception as e:
                logger.warning(f"Error saving to DynamoDB: {str(e)}")
        
        return {
            'statusCode': 200,
            'body': {
                'message': 'Modeled dataset created successfully',
                'modeledDatasetId': modeled_dataset_id,
                'inferenceJobId': inference_job_id
            }
        }
        
    except Exception as e:
        if e.args and e.args[0].startswith('{'):
            raise e
        error_info = format_error('API_ERROR', f"Error in modeled dataset operation: {str(e)}", 'create_modeled_dataset', 'Check AMC API credentials and schema configuration')
        logger.error(json.dumps(error_info))
        raise ValueError(json.dumps(error_info))
