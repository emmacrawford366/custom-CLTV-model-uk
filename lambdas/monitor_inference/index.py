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
    Monitor the status of a Trained Model Inference Job in Amazon Clean Rooms ML (ACM).
    
    This function polls an inference job to check if it has completed, is still in
    progress, or has failed. It's called repeatedly by Step Functions until the job
    reaches a terminal state (ACTIVE or CREATE_FAILED).
    
    Args:
        event (dict): Lambda event object containing:
            - workflow_type (str): Type of workflow, must be 'inference'
            - account_id (str): The AMC account ID for API access
            - instance_id (str): The AMC instance ID
            - inferenceJob.trainedModelInferenceJobId (str): ID of the inference job to monitor
        context: Lambda context object (unused)
    
    Returns:
        dict: Response with status-dependent content:
        
            For ACTIVE (completed successfully):
            - statusCode: 200
            - body:
                - message: Success message
                - trainedModelInferenceJobId: The inference job ID
                - status: 'ACTIVE'
                - workflow_type: 'inference'
                - jobDetails: Dictionary with job statistics including:
                    - name: Job name
                    - approximateRecordCount: Number of records processed
                    - numberOfFiles: Number of output files
                    - completionTime: When job completed
                    - createTime: When job was created
                    - lastModifiedTime: Last update time
            
            For CREATE_IN_PROGRESS (still running):
            - statusCode: 202
            - body:
                - message: In progress message
                - trainedModelInferenceJobId: The inference job ID
                - status: 'CREATE_IN_PROGRESS'
                - workflow_type: 'inference'
            
            For CREATE_FAILED (job failed):
            - statusCode: 400
            - body:
                - message: Failure message with reason
                - trainedModelInferenceJobId: The inference job ID
                - status: 'FAILED'
                - workflow_type: 'inference'
                - failureReason: Detailed failure reason
                - error_info: Formatted error information
    
    Raises:
        ValueError: If validation fails with formatted error info
        Exception: If API call fails or unexpected status received, with formatted error info
        
    Environment Variables:
        SECRETS_NAME (str): Name of AWS Secrets Manager secret containing API credentials
    
    Note:
        This function is designed to be called repeatedly (polling) until the inference
        job reaches a terminal state. Step Functions handles the retry logic.
    """
    try:
        # Log received event
        logger.info("Received event: %s", json.dumps(event, indent=2))
        
        if 'workflow_type' not in event:
            error_info = format_error('VALIDATION', "Missing 'workflow_type' field", 'monitor_inference')
            raise ValueError(json.dumps(error_info))

        if 'account_id' not in event:
            error_info = format_error('VALIDATION', "Missing 'account_id' field", 'monitor_inference')
            raise ValueError(json.dumps(error_info))
        
        if 'instance_id' not in event:
            error_info = format_error('VALIDATION', "Missing 'instance_id' field", 'monitor_inference')
            raise ValueError(json.dumps(error_info))
        
        inference_job_id = event['inferenceJob']['trainedModelInferenceJobId']
        logger.info(f"Inference Job ID: {inference_job_id}")
        
        secret_name = os.environ.get('SECRETS_NAME')
        
        # Initialize handler
        inference_job_handler = acm_api.TrainedModelInferenceJobs(
            secret_name=secret_name,
            ACCOUNT_ID=event['account_id'],
            INSTANCE_ID=event['instance_id']
        )
        
        # Get inference job status
        job_details = inference_job_handler.get_inference_job(inference_job_id)
        logger.info(f"Inference job details: {job_details}")
        status = job_details.get('status')
        status_reason = job_details.get('statusReason')
        
        logger.info(f"Inference job status: {status}")
        
        if status == 'ACTIVE':
            logger.info("Inference job completed successfully")
            return {
                'statusCode': 200,
                'body': {
                    'message': 'Inference job completed successfully',
                    'trainedModelInferenceJobId': inference_job_id,
                    'status': 'ACTIVE',
                    'workflow_type': event['workflow_type'],
                    'jobDetails': {
                        'name': job_details.get('name'),
                        'approximateRecordCount': job_details.get('approximateRecordCount'),
                        'numberOfFiles': job_details.get('numberOfFiles'),
                        'completionTime': job_details.get('completionTime'),
                        'createTime': job_details.get('createTime'),
                        'lastModifiedTime': job_details.get('lastModifiedTime')
                    }
                }
            }
        elif status == 'CREATE_IN_PROGRESS' or status == 'QUEUED':
            logger.info("Inference job still in progress or queued")
            return {
                'statusCode': 202,
                'body': {
                    'message': 'Inference job is still in progress',
                    'trainedModelInferenceJobId': inference_job_id,
                    'status': 'CREATE_IN_PROGRESS',
                    'workflow_type': event['workflow_type']
                }
            }
        elif status == 'CREATE_FAILED':
            logger.error(f"Inference job failed. Error message from API: {status_reason}")
            error_info = format_error('INFERENCE JOB', f"Inference job failed: {status_reason}", 'monitor_inference', 'Check inference parameters and input data')
            return {
                'statusCode': 400,
                'body': {
                    'message': f"Inference job failed. Error message from API: {status_reason}",
                    'trainedModelInferenceJobId': inference_job_id,
                    'status': 'FAILED',
                    'workflow_type': event['workflow_type'],
                    'failureReason': job_details.get('failureReason', 'Unknown failure reason'),
                    'error_info': error_info
                }
            }
        else:
            error_info = format_error('API_ERROR', f"Unexpected inference job status: {status}", 'monitor_inference')
            raise Exception(json.dumps(error_info))
            
    except Exception as e:
        if e.args and e.args[0].startswith('{'):
            raise e
        error_info = format_error('INFRASTRUCTURE', f"Lambda execution failed: {str(e)}", 'monitor_inference')
        logger.error(json.dumps(error_info))
        raise Exception(json.dumps(error_info))
