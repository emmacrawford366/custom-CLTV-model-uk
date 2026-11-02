import json
import boto3
import logging
import requests
import os
import re
from datetime import datetime
from botocore.exceptions import ClientError

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

def extract_error_log_urls(event):
    """
    Extract error log URLs from various locations in the event payload.
    
    This function searches for pre-signed S3 URLs containing error logs from ACM
    training and inference jobs. These URLs are typically embedded in error messages
    when jobs fail.
    
    Args:
        event (dict): Lambda event object that may contain error messages with URLs in:
            - training_job_monitoring.training_job_monitoring_result.error_info.error_message
            - monitoring.inference_monitoring_result.error_info.error_message
    
    Returns:
        list: List of unique error log URLs found, with duplicates removed while
              preserving order. Empty list if no URLs found.
    
    Note:
        URLs must match the pattern: 
        https://errorsummarylog-egress-*.s3.amazonaws.com/*
    """
    error_urls = []
    url_pattern = r'https://errorsummarylog-egress-[^.\s]+\.s3\.amazonaws\.com/[^\s]+'
    
    # Check training job monitoring error messages
    training_error = event.get('training_job_monitoring', {}).get('training_job_monitoring_result', {}).get('error_info', {}).get('error_message', '')
    if training_error:
        urls = re.findall(url_pattern, training_error)
        error_urls.extend(urls)
        logger.info(f"Found {len(urls)} error log URLs in training error message")
    
    # Check inference job monitoring error messages  
    inference_error = event.get('monitoring', {}).get('inference_monitoring_result', {}).get('error_info', {}).get('error_message', '')
    if inference_error:
        urls = re.findall(url_pattern, inference_error)
        error_urls.extend(urls)
        logger.info(f"Found {len(urls)} error log URLs in inference error message")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_urls = []
    for url in error_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    
    return unique_urls

def lambda_handler(event, context):
    """
    Download files from pre-signed URLs and store them in S3 for ML data exports or error logs.
    
    This function serves dual purposes:
    1. Downloads ML data export files from measurement model training jobs
    2. Downloads error logs when training or inference jobs fail
    
    The function automatically detects which type of URLs are present and processes
    accordingly. It organizes files in S3 with a structured path for easy retrieval.
    
    Args:
        event (dict): Lambda event object that may contain:
            For ML Data Exports:
            - mlDataExportStatus.mlDataExport.preSignedS3Uris (list): Pre-signed S3 URLs
            - mlDataExportStatus.mlDataExport.mlDataExportId (str): Export identifier
            - data_type (str): Type of data being exported (e.g., 'MMM')
            
            For Error Logs:
            - training_job_monitoring.training_job_monitoring_result.error_info.error_message (str)
            - monitoring.inference_monitoring_result.error_info.error_message (str)
            
            Common fields:
            - acm_algorithm_name (str): Algorithm name for organizing files
            - instance_id (str): Instance identifier
            - current_time (str): Timestamp for file organization
            
        context: Lambda context object (unused)
    
    Returns:
        dict: Response containing:
            - statusCode (int): HTTP status code (200 for success)
            - body (dict): Response body with:
                - message (str): Summary of download operation
                - download_type (str): Type of files downloaded ('ml_data_export', 'error_log', or 'none')
                - downloaded_files (list): Details of each downloaded file including:
                    - original_url: Source URL
                    - s3_bucket: Destination bucket
                    - s3_key: S3 object key
                    - filename: File name
                    - size_bytes: File size
                    - error_content (str, optional): Error log content for DynamoDB storage
                - storage_location (str): S3 path prefix where files were stored
    
    Raises:
        ValueError: If configuration is missing or download operation fails, with formatted error info
        
    Environment Variables:
        S3_BUCKET_NAME (str): Name of S3 bucket for storing downloaded files
    
    Note:
        - Files are organized in S3 by type, algorithm, instance, and timestamp
        - For error logs, content is included in response for DynamoDB storage
        - Failed downloads don't halt processing; they're logged and marked as failed
        - If no URLs are found, the function passes through gracefully with success status
    """
    try:
        logger.info("=== DOWNLOAD_URL LAMBDA STARTED ===")
        logger.info("Received event keys: %s", list(event.keys()))
        
        # Initialize S3 client
        s3_client = boto3.client('s3')
        bucket_name = os.environ.get('S3_BUCKET_NAME')
        
        if not bucket_name:
            error_info = format_error('CONFIGURATION', 'S3_BUCKET_NAME environment variable not set', 'download_url')
            raise ValueError(json.dumps(error_info))
        
        # Extract common fields
        acm_algorithm_name = event.get('acm_algorithm_name', 'unknown')
        instance_id = event.get('instance_id', 'unknown')
        data_type = event.get('data_type', 'unknown')
        current_time = event.get('current_time', datetime.now().isoformat())
        
        logger.info(f"Context: algorithm={acm_algorithm_name}, instance={instance_id}, data_type={data_type}")
        
        # Check for ML Data Export URLs (existing functionality)
        ml_export_urls = event.get('mlDataExportStatus', {}).get('mlDataExport', {}).get('preSignedS3Uris', [])
        ml_data_export_id = event.get('mlDataExportStatus', {}).get('mlDataExport', {}).get('mlDataExportId', 'unknown')
        
        logger.info(f"ML Data Export check: Found {len(ml_export_urls)} URLs in mlDataExportStatus")
        
        # Check for Error Log URLs (new functionality)
        error_log_urls = extract_error_log_urls(event)
        
        logger.info(f"Error Log check: Found {len(error_log_urls)} URLs in error messages")
        
        # Determine which URLs to process with detailed logging
        urls_to_download = []
        download_type = None
        storage_prefix = None
        
        if ml_export_urls:
            urls_to_download = ml_export_urls
            download_type = 'ml_data_export'
            storage_prefix = f"{data_type}/{acm_algorithm_name}/{instance_id}/{ml_data_export_id}/{current_time}"
            logger.info(f"DECISION: Processing ML data export - {len(ml_export_urls)} URLs")
            logger.info(f"ML Export ID: {ml_data_export_id}")
            logger.info(f"Storage prefix: {storage_prefix}")
        elif error_log_urls:
            urls_to_download = error_log_urls
            download_type = 'error_log'
            # Extract trained model ID for error log organization
            trained_model_id = (
                event.get('training_job_response', {}).get('training_response', {}).get('trainedModelId') or
                event.get('training_job_monitoring', {}).get('training_job_monitoring_result', {}).get('trainedModelId') or
                'unknown'
            )
            storage_prefix = f"error_logs/{acm_algorithm_name}/{instance_id}/{trained_model_id}/{current_time}"
            logger.info(f"DECISION: Processing error logs - {len(error_log_urls)} URLs")
            logger.info(f"Trained Model ID: {trained_model_id}")
            logger.info(f"Storage prefix: {storage_prefix}")
            for i, url in enumerate(error_log_urls, 1):
                logger.info(f"Error log URL {i}: {url[:100]}...")
        else:
            logger.info("DECISION: No URLs found to download - passing through")
            logger.info("Checked locations:")
            logger.info("  - mlDataExportStatus.mlDataExport.preSignedS3Uris: Not found or empty")
            logger.info("  - training_job_monitoring error messages: No error log URLs found")
            logger.info("  - inference monitoring error messages: No error log URLs found")
            return {
                'statusCode': 200,
                'body': {
                    'message': 'No files to download - no URLs found in payload',
                    'download_type': 'none',
                    'downloaded_files': [],
                    'ml_export_urls_checked': len(ml_export_urls),
                    'error_log_urls_checked': len(error_log_urls)
                }
            }
        
        downloaded_files = []
        
        # Download each URL
        for i, url in enumerate(urls_to_download):
            try:
                logger.info(f"Downloading URL {i+1}/{len(urls_to_download)}: {url[:100]}...")
                
                # Download the file
                response = requests.get(url, timeout=300)
                response.raise_for_status()
                
                # Extract filename from URL or create one based on type
                filename = url.split('/')[-1].split('?')[0]
                if not filename or filename == '':
                    if download_type == 'error_log':
                        filename = f"error_summary_{i+1}.txt"
                    else:
                        filename = f"export_file_{i+1}.csv"
                else:
                    # For error logs, ensure .txt extension
                    if download_type == 'error_log' and not filename.endswith('.txt'):
                        filename = f"{filename}.txt"
                
                # Create S3 key with organized structure
                s3_key = f"{storage_prefix}/{filename}"
                
                # Determine content type based on download type
                content_type = 'text/plain' if download_type == 'error_log' else 'text/csv'
                
                # Upload to S3
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=s3_key,
                    Body=response.content,
                    ContentType=content_type
                )
                
                # For error logs, also include the content in the payload for DynamoDB storage
                file_info = {
                    'original_url': url,
                    's3_bucket': bucket_name,
                    's3_key': s3_key,
                    'filename': filename,
                    'size_bytes': len(response.content),
                    'download_type': download_type
                }
                
                # Add error content to payload if it's an error log
                if download_type == 'error_log':
                    try:
                        # Decode content as text for error logs
                        error_content = response.content.decode('utf-8')
                        file_info['error_content'] = error_content
                        logger.info(f"Added error content to payload ({len(error_content)} characters)")
                    except UnicodeDecodeError:
                        # If decoding fails, store as base64
                        import base64
                        file_info['error_content_base64'] = base64.b64encode(response.content).decode('utf-8')
                        logger.info("Error content stored as base64 due to encoding issues")
                
                downloaded_files.append(file_info)
                
                logger.info(f"Successfully downloaded and stored: {s3_key}")
                
            except requests.RequestException as e:
                error_info = format_error('DOWNLOAD_ERROR', f"Failed to download URL {i+1}: {str(e)}", 'download_url')
                logger.error(json.dumps(error_info))
                # Continue with other URLs instead of failing completely
                downloaded_files.append({
                    'original_url': url,
                    'error': str(e),
                    'status': 'failed',
                    'download_type': download_type
                })
            
            except ClientError as e:
                error_info = format_error('S3_ERROR', f"Failed to upload to S3: {str(e)}", 'download_url')
                logger.error(json.dumps(error_info))
                downloaded_files.append({
                    'original_url': url,
                    'error': str(e),
                    'status': 'failed',
                    'download_type': download_type
                })
        
        # Build response based on download type
        successful_downloads = len([f for f in downloaded_files if "error" not in f])
        
        response_body = {
            'message': f'Downloaded {successful_downloads} of {len(urls_to_download)} files',
            'download_type': download_type,
            'downloaded_files': downloaded_files,
            'storage_location': f's3://{bucket_name}/{storage_prefix}/'
        }
        
        # Add type-specific metadata
        if download_type == 'ml_data_export':
            response_body['mlDataExportId'] = ml_data_export_id
        elif download_type == 'error_log':
            response_body['error_log_urls_found'] = len(error_log_urls)
            response_body['original_error_urls'] = error_log_urls
        
        return {
            'statusCode': 200,
            'body': response_body
        }
        
    except Exception as e:
        if e.args and e.args[0].startswith('{'):
            raise e
        error_info = format_error('INFRASTRUCTURE', f"Lambda execution failed: {str(e)}", 'download_url')
        logger.error(json.dumps(error_info))
        raise ValueError(json.dumps(error_info))
