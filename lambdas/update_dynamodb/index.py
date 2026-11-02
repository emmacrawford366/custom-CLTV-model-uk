import json
import boto3
import logging
from botocore.exceptions import ClientError
import os
from datetime import datetime

# Configure logging
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

def extract_error_info(payload, error_log_content=None, error_log_s3_locations=None, error_log_presigned_urls=None):
    """
    Extract error information from Step Functions payload and enrich with error log data.
    
    This function recursively searches the payload for error information from various
    sources (Step Functions errors, API errors, etc.) and optionally enriches them
    with downloaded error log content and S3 locations.
    
    Args:
        payload (dict): The Step Functions execution payload containing potential errors
        error_log_content (list, optional): List of error log content dictionaries with:
            - filename: Name of the error log file
            - content: Text content of the error log
            - s3_location: S3 path where log is stored
            - original_url: Original pre-signed URL
        error_log_s3_locations (list, optional): List of S3 paths to error logs
        error_log_presigned_urls (list, optional): List of original pre-signed URLs
    
    Returns:
        list: List of error dictionaries, each containing:
            - error_category: Category of the error
            - error_message: Detailed error message
            - lambda_name: Source of the error
            - timestamp: When error occurred
            - suggestion: Recommended action
            - path: Location in payload where error was found
            - error_log_content (optional): Error log content if available
            - error_log_s3_location (optional): S3 location if available
            - error_log_original_url (optional): Original URL if available
    
    Note:
        The function searches for both Step Functions error fields (like 'training_error',
        'map_failure_info') and embedded 'error_info' dictionaries throughout the payload.
    """
    errors = []
    
    # Check for all error field types from Step Functions
    error_fields = ['training_error', 'map_failure_info', 'export_error', 'export_status_error', 
                   'dataset_error', 'upload_error', 'status_error']
    
    for field in error_fields:
        if field in payload:
            error_data = payload[field]
            if error_data.get('Error') and error_data.get('Cause'):
                error_info = format_error(
                    'API_ERROR',
                    error_data.get('Cause'),
                    field.replace('_', ' ').title(),
                    'Check Step Functions console for detailed logs'
                )
                errors.append(error_info)
    
    def find_errors(obj, path=""):
        if isinstance(obj, dict):
            if 'error_info' in obj:
                error_info = obj['error_info']
                if isinstance(error_info, dict):
                    error_info['path'] = path
                    # Add optional error log fields if available
                    if error_log_content:
                        error_info['error_log_content'] = error_log_content[0].get('content') if error_log_content else None
                    if error_log_s3_locations:
                        error_info['error_log_s3_location'] = error_log_s3_locations[0] if error_log_s3_locations else None
                    if error_log_presigned_urls:
                        error_info['error_log_original_url'] = error_log_presigned_urls[0] if error_log_presigned_urls else None
                    errors.append(error_info)
            for key, value in obj.items():
                find_errors(value, f"{path}.{key}" if path else key)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                find_errors(item, f"{path}[{i}]")
    
    find_errors(payload)
    return errors

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ.get('DYNAMODB_TABLE_EXECUTIONS'))

def lambda_handler(event, context):
    """
    Update DynamoDB execution record with pipeline completion status and results.
    
    This function is called at the end of Step Functions execution (success or failure)
    to update the execution record in DynamoDB. It stores comprehensive information about
    the execution including status, results, errors, and downloadedfiles.
    
    Args:
        event (dict): Lambda event object containing:
            - end_status (str): Final status ('COMPLETED' or 'FAILED')
            - Payload (dict): Full Step Functions execution payload with:
                - acm_algorithm_name (str): Algorithm name (required)
                - current_time (str): Execution timestamp (required)
                - workflow_type (str): 'train' or 'inference'
                - training_job_response (dict, optional): Training job results
                - inferenceJob (dict, optional): Inference job information
                - modeledDataset (dict, optional): Dataset creation results
                - datasetUpload (dict, optional): Upload job information
                - uploadStatus (dict, optional): Upload completion status
                - downloadedFiles (dict, optional): Downloaded file information
                - mlDataExportStatus (dict, optional): Export status and URLs
                - error_log_download (dict, optional): Downloaded error logs
        context: Lambda context object (unused)
    
    Returns:
        dict: Response containing:
            - statusCode (int): HTTP status code (200 for success)
            - body (dict): Response body with:
                - message: Success message
                - acm_algorithm_name: Algorithm name
                - execution_timestamp: Execution timestamp
                - updated_status: The final status set
                - trained_model_id: Training job ID (if applicable)
                - trainedModelInferenceId: Inference job ID (if applicable)
                - workflow_type: The workflow type
                - create_modeled_dataset: Dataset creation info (if applicable)
                - create_modeled_dataset_upload: Upload info (if applicable)
                - get_modeled_dataset_upload: Upload status (if applicable)
                - storage_location: S3 location of downloaded files
                - data_type: Data type (for measurement models)
                - Payload: Full execution payload
    
    Raises:
        ValueError: If required fields are missing, with formatted error info
        Exception: If DynamoDB update fails, with formatted error info
        
    Environment Variables:
        DYNAMODB_TABLE_EXECUTIONS (str): Name of DynamoDB table for execution tracking
    
    Note:
        - This function handles both successful and failed executions
        - For failed executions, it extracts and stores detailed error information
        - Error logs from failed jobs are enriched with S3 locations and content
        - The entire execution payload is stored for comprehensive debugging
        - Updates are atomic using DynamoDB UpdateItem with conditional expressions
    """
    try:
        # Log received event
        logger.info("Received event: %s", json.dumps(event, indent=2))
        
        # Validate required fields
        required_fields = ['acm_algorithm_name', 'current_time']
        missing_fields = [field for field in required_fields if field not in event['Payload']]
        
        if missing_fields:
            error_info = format_error('VALIDATION', f"Missing required fields: {', '.join(missing_fields)}", 'update_dynamoDB')
            logger.error(json.dumps(error_info))
            raise ValueError(json.dumps(error_info))
            
        # Extract values from event
        end_status = event['end_status']
        acm_algorithm_name = event['Payload']['acm_algorithm_name']
        current_time = event['Payload']['current_time']
        
        # Extract trainedModelInferenceId if available
        trained_model_inference_id = None
        if 'inferenceJob' in event['Payload']:
            trained_model_inference_id = event['Payload']['inferenceJob'].get('trainedModelInferenceJobId')

        trained_model_id = None
        if 'training_job_response' in event['Payload']:
            trained_model_id = event['Payload']['training_job_response'].get('training_response', {}).get('trainedModelId')
        
        # Extract outputs from Create_modeled_dataset, Create_modeled_dataset_upload, and get_modeled_dataset_upload
        create_modeled_dataset_output = event['Payload'].get('modeledDataset')
        create_modeled_dataset_upload_output = event['Payload'].get('datasetUpload')
        get_modeled_dataset_upload_output = event['Payload'].get('uploadStatus')
        
        # Extract new training completion elements
        storage_location = event['Payload'].get('downloadedFiles', {}).get('downloadResult', {}).get('storage_location')
        data_type = event['Payload'].get('mlDataExport', {}).get('date_type') or event['Payload'].get('mlDataExportStatus', {}).get('mlDataExport', {}).get('dataType')
        downloaded_files = event['Payload'].get('downloadedFiles')
        ml_data_export_status = event['Payload'].get('mlDataExportStatus')
        
        # Extract error log download information
        error_log_downloads = None
        error_log_content = None
        error_log_s3_locations = []
        error_log_presigned_urls = []
        
        # Check if we have downloaded error logs from the download_url lambda
        # First check the actual location where Step Function stores the result
        download_result = None
        if 'error_log_download' in event['Payload'] and 'error_log_download_result' in event['Payload']['error_log_download']:
            download_result = event['Payload']['error_log_download']['error_log_download_result']
            logger.info("Found error log download result in error_log_download.error_log_download_result")
        # Fallback to legacy location for backward compatibility
        elif 'downloadedFiles' in event['Payload'] and isinstance(event['Payload']['downloadedFiles'], dict):
            download_result = event['Payload']['downloadedFiles']
            logger.info("Found error log download result in downloadedFiles (legacy location)")
        
        if download_result and download_result.get('download_type') == 'error_log':
            error_log_downloads = download_result
            downloaded_files_list = download_result.get('downloaded_files', [])
            logger.info(f"Processing {len(downloaded_files_list)} downloaded error log files")
            
            # Extract error content, S3 locations, and original URLs
            for file_info in downloaded_files_list:
                if file_info.get('download_type') == 'error_log' and 'error' not in file_info:
                    # Collect S3 locations
                    s3_location = f"s3://{file_info.get('s3_bucket')}/{file_info.get('s3_key')}"
                    error_log_s3_locations.append(s3_location)
                    
                    # Collect original pre-signed URLs
                    error_log_presigned_urls.append(file_info.get('original_url'))
                    
                    # Collect error content for display
                    if 'error_content' in file_info:
                        if error_log_content is None:
                            error_log_content = []
                        error_log_content.append({
                            'filename': file_info.get('filename'),
                            'content': file_info.get('error_content'),
                            's3_location': s3_location,
                            'original_url': file_info.get('original_url')
                        })
                        logger.info(f"Added error content for file: {file_info.get('filename')}")
                    elif 'error_content_base64' in file_info:
                        if error_log_content is None:
                            error_log_content = []
                        error_log_content.append({
                            'filename': file_info.get('filename'),
                            'content_base64': file_info.get('error_content_base64'),
                            's3_location': s3_location,
                            'original_url': file_info.get('original_url')
                        })
                        logger.info(f"Added base64 error content for file: {file_info.get('filename')}")
        
        # Extract error information from payload and add optional error log fields
        error_list = extract_error_info(event['Payload'], error_log_content, error_log_s3_locations, error_log_presigned_urls)
        
        # Store entire event
        full_event = json.dumps(event)
        
        logger.info(f"Updating status to '{end_status}' for algorithm '{acm_algorithm_name}' "
                   f"executed at '{current_time}'")
        
        # Prepare update expression and attribute values
        update_expression = 'SET #status = :status'
        expression_attribute_values = {':status': end_status}
        expression_attribute_names = {'#status': 'status'}
        
        # Add trainedModelInferenceId to update if available
        if trained_model_inference_id:
            update_expression += ', #trainedModelInferenceId = :trainedModelInferenceId'
            expression_attribute_values[':trainedModelInferenceId'] = trained_model_inference_id
            expression_attribute_names['#trainedModelInferenceId'] = 'trainedModelInferenceId'

        # Add trained_model_id to update if available
        if trained_model_id:
            update_expression += ', #trained_model_id = :trained_model_id'
            expression_attribute_values[':trained_model_id'] = trained_model_id
            expression_attribute_names['#trained_model_id'] = 'trained_model_id'
        
        # Add Create_modeled_dataset output if available
        if create_modeled_dataset_output:
            update_expression += ', #create_modeled_dataset = :create_modeled_dataset'
            expression_attribute_values[':create_modeled_dataset'] = create_modeled_dataset_output
            expression_attribute_names['#create_modeled_dataset'] = 'create_modeled_dataset'
        
        # Add Create_modeled_dataset_upload output if available
        if create_modeled_dataset_upload_output:
            update_expression += ', #create_modeled_dataset_upload = :create_modeled_dataset_upload'
            expression_attribute_values[':create_modeled_dataset_upload'] = create_modeled_dataset_upload_output
            expression_attribute_names['#create_modeled_dataset_upload'] = 'create_modeled_dataset_upload'
        
        # Add get_modeled_dataset_upload output if available
        if get_modeled_dataset_upload_output:
            update_expression += ', #get_modeled_dataset_upload = :get_modeled_dataset_upload'
            expression_attribute_values[':get_modeled_dataset_upload'] = get_modeled_dataset_upload_output
            expression_attribute_names['#get_modeled_dataset_upload'] = 'get_modeled_dataset_upload'
        
        # Add new training completion elements if available
        if storage_location:
            update_expression += ', #storage_location = :storage_location'
            expression_attribute_values[':storage_location'] = storage_location
            expression_attribute_names['#storage_location'] = 'storage_location'
        
        if data_type:
            update_expression += ', #data_type = :data_type'
            expression_attribute_values[':data_type'] = data_type
            expression_attribute_names['#data_type'] = 'data_type'
        
        if downloaded_files:
            update_expression += ', #downloaded_files = :downloaded_files'
            expression_attribute_values[':downloaded_files'] = downloaded_files
            expression_attribute_names['#downloaded_files'] = 'downloaded_files'
        
        if ml_data_export_status:
            update_expression += ', #ml_data_export_status = :ml_data_export_status'
            expression_attribute_values[':ml_data_export_status'] = ml_data_export_status
            expression_attribute_names['#ml_data_export_status'] = 'ml_data_export_status'
        
        # Add full event
        update_expression += ', #full_event = :full_event'
        expression_attribute_values[':full_event'] = full_event
        expression_attribute_names['#full_event'] = 'full_event'
        
        
        # Add error information if present
        if error_list:
            update_expression += ', #error_list = :error_list, #has_errors = :has_errors'
            expression_attribute_values[':error_list'] = error_list
            expression_attribute_values[':has_errors'] = True
            expression_attribute_names['#error_list'] = 'error_list'
            expression_attribute_names['#has_errors'] = 'has_errors'
        else:
            update_expression += ', #has_errors = :has_errors'
            expression_attribute_values[':has_errors'] = False
            expression_attribute_names['#has_errors'] = 'has_errors'
        
        # Update DynamoDB record
        try:
            response = table.update_item(
                Key={
                    'acm_algorithm_name': acm_algorithm_name,
                    'execution_timestamp': current_time
                },
                UpdateExpression=update_expression,
                ExpressionAttributeNames=expression_attribute_names,
                ExpressionAttributeValues=expression_attribute_values,
                ReturnValues='UPDATED_NEW'
            )
            
            logger.info("DynamoDB update successful")
            logger.info(f"Updated attributes: {response.get('Attributes', {})}")
            
            return_body = {
                'message': 'Status updated successfully',
                'acm_algorithm_name': acm_algorithm_name,
                'execution_timestamp': current_time,
                'updated_status': end_status,
                'trained_model_inference_id': trained_model_inference_id,
                'trained_model_id': trained_model_id,
                'workflow_type': event['Payload'].get('workflow_type'),
                'create_modeled_dataset': create_modeled_dataset_output,
                'create_modeled_dataset_upload': create_modeled_dataset_upload_output,
                'get_modeled_dataset_upload': get_modeled_dataset_upload_output,
                'storage_location': storage_location,
                'data_type': data_type,
                'downloaded_files': downloaded_files,
                'ml_data_export_status': ml_data_export_status,
                "Payload": event['Payload']
            }
            
            if trained_model_inference_id:
                return_body['trainedModelInferenceId'] = trained_model_inference_id
            
            return {
                'statusCode': 200,
                'body': return_body
            }
            
        except ClientError as e:
            error_info = format_error('INFRASTRUCTURE', f"DynamoDB update failed: {str(e)}", 'update_dynamoDB')
            logger.error(json.dumps(error_info))
            raise Exception(json.dumps(error_info))
            
    except Exception as e:
        if e.args and e.args[0].startswith('{'):
            raise e
        error_info = format_error('INFRASTRUCTURE', f"Lambda execution failed: {str(e)}", 'update_dynamoDB')
        logger.error(json.dumps(error_info))
        raise Exception(json.dumps(error_info))
