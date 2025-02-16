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
from typing import Dict, Any
from .models import Transferencia, Tarjeta
from .process_images import process_single_batch

logger = logging.getLogger(__name__)

async def process_single_image(image_path: str) -> Dict[str, Any]:
    """
    Process a single image through the workflow and return the results
    
    Args:
        image_path: Path to the image file to process
        
    Returns:
        Dict containing:
        - processed_content: bytes of processed image
        - data: Transferencia or Tarjeta object if found
        - data_type: "transferencia" or "tarjeta"
        - success: boolean indicating if processing was successful
        - message: status message
    """
    try:
        # Convert path to Path object
        image_path = Path(image_path)
        logger.info(f"Processing image at path: {image_path}")
        
        # Get image directory and filename
        image_dir = image_path.parent
        filename = image_path.name
        logger.info(f"Image directory: {image_dir}, filename: {filename}")
        
        # Process the single image using process_single_batch
        logger.info("Calling process_single_batch...")
        result = await process_single_batch(
            image_paths=[filename],
            image_directory=str(image_dir)
        )
        logger.info(f"process_single_batch result: {result}")
        
        if result["status"] != "success":
            logger.error(f"Process failed with status: {result['status']}")
            return {
                "success": False,
                "message": result.get("message", "Unknown error")
            }
            
        # Get the last message which should contain the extracted data
        messages = result.get("messages", [])
        logger.info(f"Got messages: {messages}")
        
        if not messages:
            logger.warning("No messages returned from workflow")
            return {
                "success": False,
                "message": "No messages returned from workflow"
            }
            
        last_message = messages[-1]
        logger.info(f"Last message: {last_message}")
        
        # If we got FINAL ANSWER, the data has been stored in the database
        if last_message.content == "FINAL ANSWER":
            return {
                "processed_content": image_path.read_bytes(),
                "success": True,
                "message": "Successfully processed image"
            }
        else:
            logger.warning(f"Unexpected workflow end state: {last_message.content}")
            return {
                "success": False,
                "message": "Workflow did not complete successfully"
            }

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        logger.exception("Full traceback:")
        return {
            "success": False,
            "message": f"Error processing image: {str(e)}"
        }
