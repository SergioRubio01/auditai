# Copyright 2025-present Sergio García Arrojo. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
from .models import Transferencia, Tarjeta
from .process_images import process_single_batch, image_cache
from .workflowmanager import WorkflowManager
import json

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def process_single_image(
    image_path: str,
    workflow_type: Optional[str] = None,
    use_cache: bool = True,
    priority: int = 2
) -> Dict[str, Any]:
    """
    Enhanced single image processor with intelligent workflow selection and caching.
    
    Args:
        image_path: Path to the image file to process
        workflow_type: Specific workflow to use (auto-detect if None)
        use_cache: Whether to use cached results
        priority: Processing priority (1-4, higher is more important)
        
    Returns:
        Dict containing:
        - processed_content: bytes of processed image
        - detected_type: Detected document type
        - extracted_data: Extracted structured data
        - workflow_results: Results from each workflow
        - success: boolean indicating if processing was successful
        - message: status message
        - metadata: Processing metadata
    """
    start_time = datetime.now()
    processing_id = f"proc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{Path(image_path).stem}"
    
    try:
        # Convert path to Path object and validate
        image_path = Path(image_path).resolve()
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if not image_path.is_file():
            raise ValueError(f"Path is not a file: {image_path}")
        
        # Check file size
        file_size = image_path.stat().st_size
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            raise ValueError(f"File too large ({file_size / 1024 / 1024:.1f}MB): {image_path}")
        
        logger.info(f"Starting processing {processing_id} for: {image_path}")
        
        # Get image directory and filename
        image_dir = image_path.parent
        filename = image_path.name
        
        # Check cache if enabled
        cache_key = workflow_type or "auto"
        if use_cache:
            cached_result = await image_cache.get(filename, cache_key)
            if cached_result:
                logger.info(f"Cache hit for {processing_id}")
                return {
                    **cached_result,
                    "from_cache": True,
                    "processing_id": processing_id
                }
        
        # Process the image
        logger.info(f"Processing {processing_id} with priority {priority}")
        
        if workflow_type:
            # Process with specific workflow
            workflow_manager = WorkflowManager()
            result = await workflow_manager.process_workflow(
                workflow_type=workflow_type,
                image_paths=[filename],
                image_directory=str(image_dir)
            )
        else:
            # Process with all workflows (auto-detect)
            result = await process_single_batch(
                image_paths=[filename],
                image_directory=str(image_dir),
                use_cache=False  # We handle caching at this level
            )
        
        # Analyze results to determine document type
        detected_type = await _detect_document_type(result)
        extracted_data = await _extract_structured_data(result, detected_type)
        
        # Prepare comprehensive response
        processing_duration = (datetime.now() - start_time).total_seconds()
        
        response = {
            "processed_content": image_path.read_bytes(),
            "success": result.get("status") in ["success", "partial"],
            "message": "Successfully processed image" if result.get("status") == "success" else result.get("message", "Processing completed with warnings"),
            "processing_id": processing_id,
            "detected_type": detected_type,
            "extracted_data": extracted_data,
            "workflow_results": result,
            "metadata": {
                "filename": filename,
                "file_size": file_size,
                "processing_duration": processing_duration,
                "workflow_type": workflow_type or "auto",
                "priority": priority,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Cache successful results
        if use_cache and response["success"]:
            await image_cache.set(filename, cache_key, response)
        
        logger.info(f"Completed {processing_id} in {processing_duration:.2f}s")
        return response

    except Exception as e:
        logger.error(f"Error in {processing_id}: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": f"Error processing image: {str(e)}",
            "processing_id": processing_id,
            "error_type": type(e).__name__,
            "metadata": {
                "filename": str(image_path),
                "timestamp": datetime.now().isoformat(),
                "duration": (datetime.now() - start_time).total_seconds()
            }
        }

async def _detect_document_type(workflow_results: Dict[str, Any]) -> str:
    """
    Analyze workflow results to determine the document type.
    """
    # Check each workflow result for successful extraction
    type_scores = {}
    
    for workflow_type in ["facturas", "pagos", "nominas"]:
        if workflow_type in workflow_results:
            result = workflow_results[workflow_type]
            if isinstance(result, dict) and result.get("status") == "success":
                # Score based on extracted data
                score = 0
                
                if workflow_type == "facturas" and result.get("factura"):
                    score = 10
                elif workflow_type == "pagos" and (result.get("transferencia") or result.get("tarjeta")):
                    score = 10
                elif workflow_type == "nominas" and result.get("nomina"):
                    score = 10
                
                type_scores[workflow_type] = score
    
    # Return the type with highest score
    if type_scores:
        return max(type_scores, key=type_scores.get)
    return "unknown"

async def _extract_structured_data(workflow_results: Dict[str, Any], document_type: str) -> Dict[str, Any]:
    """
    Extract structured data based on document type.
    """
    if document_type == "facturas" and "facturas" in workflow_results:
        result = workflow_results["facturas"]
        return {
            "type": "factura",
            "data": result.get("factura", {}),
            "table_data": result.get("tablafacturas", {})
        }
    elif document_type == "pagos" and "pagos" in workflow_results:
        result = workflow_results["pagos"]
        return {
            "type": "pago",
            "transferencia": result.get("transferencia", {}),
            "tarjeta": result.get("tarjeta", {}),
            "table_data": result.get("tablatarjetas", {})
        }
    elif document_type == "nominas" and "nominas" in workflow_results:
        result = workflow_results["nominas"]
        return {
            "type": "nomina",
            "data": result.get("nomina", {}),
            "table_data": result.get("tablanominas", {})
        }
    
    return {"type": "unknown", "data": {}}

# Batch processing with enhanced capabilities
async def process_image_batch(
    image_paths: List[str],
    workflow_type: Optional[str] = None,
    max_concurrent: int = 5
) -> List[Dict[str, Any]]:
    """
    Process multiple images concurrently with rate limiting.
    
    Args:
        image_paths: List of image paths to process
        workflow_type: Specific workflow to use for all images
        max_concurrent: Maximum concurrent operations
        
    Returns:
        List of processing results
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(image_path):
        async with semaphore:
            return await process_single_image(image_path, workflow_type)
    
    tasks = [process_with_semaphore(path) for path in image_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Convert exceptions to error results
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                "success": False,
                "message": str(result),
                "image_path": image_paths[i],
                "error_type": type(result).__name__
            })
        else:
            processed_results.append(result)
    
    return processed_results
