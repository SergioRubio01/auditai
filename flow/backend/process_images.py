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

from typing import List, Dict, Any, Optional
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.messages import HumanMessage
import logging
from .utils import encode_image
from .workflowmanager import WorkflowManager
import uuid
from pathlib import Path
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add at module level
processing_lock = asyncio.Lock()

async def process_single_batch(
    image_paths: List[str],
    image_directory: str,
    excel_filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a single batch of related images.
    
    Args:
        `image_paths`: List of image paths to process
        `image_directory`: Directory containing the images
        `excel_filename`: Optional name of the Excel file to save results
    
    Returns:
        Dict containing processing results and status
    """
    try:
        # Add a semaphore or lock to prevent multiple executions
        async with processing_lock:
            # Process the image only once
            result = await process_single_image_internal(image_paths, image_directory)
            if result.get("status") == "success":
                logger.info(f"Successfully processed {image_paths[0]}")
                return result
            else:
                logger.error(f"Failed to process {image_paths[0]}: {result.get('message')}")
                return result
                
    except Exception as e:
        logger.error(f"Error in process_single_batch: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

async def process_single_image_internal(
    image_paths: List[str],
    image_directory: str,
    workflow_type: str
) -> Dict[str, Any]:
    """
    Process a single image.
    
    Args:
        image_paths: List of image paths to process
        image_directory: Directory containing the images
        workflow_type: Type of workflow to use ('facturas' or 'pagos' )
    
    Returns:
        Dict containing processing results and status
    """
    try:
        # Create message content with all matching images
        message_content = []
        for image_path in image_paths:
            full_path = os.path.join(image_directory, image_path).replace('\\', '/')
            try:
                base64_image = encode_image(full_path)
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {str(e)}")
                continue

        if not message_content:
            return {
                "status": "error",
                "message": "No valid images to process",
                "workflow_type": workflow_type
            }

        # Initialize the results list
        results = []
        
        # Initialize the graph with specified workflow type
        graph = WorkflowManager().get_graph(workflow_type)
        
        # Pass the filename in the state for the workflow to use
        async for event in graph.astream(
            {
                "messages": [HumanMessage(content=message_content)],
                "filename": image_path,
                "workflowtype": workflow_type,
                "factura":"",
                "tablafacturas":"",
                "tablatarjetas":"",
                "transferencia":"",
                "tarjeta":"",
                "nomina":"",
                "tablanominas":""
            },
            config={"recursion_limit": 30}
        ):
            logger.info(f"Processing event in {workflow_type} workflow: {event}")
            results.append(event)
        
        return {
            "status": "success",
            # "messages": results[-1]["supervisor_agent"]["messages"] if results else [],
            "image_paths": image_paths,
            "workflow_type": workflow_type,
            "factura":"",
            "tablafacturas":"",
            "tablatarjetas":"",
            "tarjeta":"",
            "transferencia":"",
            "nomina":"",
            "tablanominas":""
        }
        
    except Exception as e:
        logger.error(f"Error processing image in {workflow_type} workflow: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "image_paths": [image_path],
            "workflow_type": workflow_type
        }

def update_excel(excel_filename: str, processed_results: Dict[str, Any]) -> None:
    """
    Update Excel file with processing results.
    
    Args:
        excel_filename: Name of the Excel file
        processed_results: Results from image processing
    """
    try:
        try:
            excel = pd.read_excel(excel_filename + '.xlsx')
        except FileNotFoundError:
            excel = pd.DataFrame()
            
        if processed_results["status"] == "success":
            start_idx = len(excel) - len(processed_results["image_paths"])
            
            if not excel.empty and start_idx >= 0:
                for i, image_path in enumerate(processed_results["image_paths"]):
                    excel.loc[start_idx + i, 'DOCUMENTO'] = image_path
                    
            excel.to_excel(excel_filename + '.xlsx', index=False)
            logger.info(f"Successfully updated Excel file: {excel_filename}")
    except Exception as e:
        logger.error(f"Error updating Excel file: {str(e)}")

def process_images(
    image_directory: str,
    base_filenames: List[str],
    excel_filename: str,
    max_workers: int = 4
) -> None:
    """
    Process multiple batches of images in parallel.
    
    Args:
        `image_directory`: Directory containing the images
        `base_filenames`: List of base filenames to match
        `excel_filename`: Name of the Excel file to save results
        `max_workers`: Maximum number of parallel workers
    """
    try:
        # Normalize directory path
        image_directory = image_directory.replace('\\', '/')
        
        # Get all files in directory and normalize their paths
        all_files = [f.replace('\\', '/') for f in os.listdir(image_directory)]
        
        # Group images by normalized paths
        image_batches = []
        for filename in base_filenames:
            filename = filename.replace('\\', '/')
            if filename in all_files:
                image_batches.append([filename])
            else:
                logger.warning(f"Image not found: {filename}")
        
        if not image_batches:
            logger.warning("No valid image batches found to process")
            return
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(
                    process_single_batch,
                    batch,
                    image_directory,
                    excel_filename
                ): batch for batch in image_batches
            }
            
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    result = future.result()
                    if result["status"] == "success":
                        update_excel(excel_filename, result)
                    else:
                        logger.error(f"Failed to process batch: {result['message']}")
                except Exception as e:
                    logger.error(f"Error processing batch {batch}: {str(e)}")
                    
    except Exception as e:
        logger.error(f"Error in process_images: {str(e)}")