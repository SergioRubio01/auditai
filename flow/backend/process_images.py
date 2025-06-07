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

from typing import List, Dict, Any, Optional, Tuple
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.messages import HumanMessage, BaseMessage
from .utils import encode_image
from .workflowmanager import WorkflowManager
import uuid
from pathlib import Path
import asyncio
import logging
import hashlib
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
import aiofiles
from collections import defaultdict

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add at module level
processing_lock = asyncio.Lock()

# Cache for processed images
class ImageCache:
    def __init__(self, ttl_seconds: int = 3600):
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.ttl = timedelta(seconds=ttl_seconds)
        self._lock = asyncio.Lock()
    
    def _get_cache_key(self, image_path: str, workflow_type: str) -> str:
        """Generate cache key from image path and workflow type"""
        return f"{workflow_type}:{image_path}"
    
    async def get(self, image_path: str, workflow_type: str) -> Optional[Any]:
        """Get cached result if available and not expired"""
        async with self._lock:
            key = self._get_cache_key(image_path, workflow_type)
            if key in self.cache:
                result, timestamp = self.cache[key]
                if datetime.now() - timestamp < self.ttl:
                    logger.debug(f"Cache hit for {key}")
                    return result
                else:
                    # Remove expired entry
                    del self.cache[key]
        return None
    
    async def set(self, image_path: str, workflow_type: str, result: Any):
        """Cache the result"""
        async with self._lock:
            key = self._get_cache_key(image_path, workflow_type)
            self.cache[key] = (result, datetime.now())
            logger.debug(f"Cached result for {key}")
    
    async def clear_expired(self):
        """Remove expired entries from cache"""
        async with self._lock:
            now = datetime.now()
            expired_keys = [
                key for key, (_, timestamp) in self.cache.items()
                if now - timestamp >= self.ttl
            ]
            for key in expired_keys:
                del self.cache[key]
            if expired_keys:
                logger.info(f"Cleared {len(expired_keys)} expired cache entries")

# Global cache instance
image_cache = ImageCache()

# Batch processing queue for improved throughput
class BatchQueue:
    def __init__(self, max_batch_size: int = 10, max_wait_seconds: float = 2.0):
        self.queue: List[Dict[str, Any]] = []
        self.max_batch_size = max_batch_size
        self.max_wait_seconds = max_wait_seconds
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(lock=self._lock)
        self._last_flush = asyncio.get_event_loop().time()
    
    async def add(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Add item to queue and return batch if ready"""
        async with self._condition:
            self.queue.append(item)
            
            # Check if batch is ready
            current_time = asyncio.get_event_loop().time()
            time_since_last_flush = current_time - self._last_flush
            
            if len(self.queue) >= self.max_batch_size or time_since_last_flush >= self.max_wait_seconds:
                batch = self.queue[:]
                self.queue.clear()
                self._last_flush = current_time
                return batch
            
            return []
    
    async def flush(self) -> List[Dict[str, Any]]:
        """Force flush the queue"""
        async with self._condition:
            batch = self.queue[:]
            self.queue.clear()
            self._last_flush = asyncio.get_event_loop().time()
            return batch

async def process_single_batch(
    image_paths: List[str],
    image_directory: str,
    excel_filename: Optional[str] = None,
    use_cache: bool = True
) -> Dict[str, Any]:
    """Process a batch of images with caching and improved error handling"""
    try:
        # Check cache first if enabled
        if use_cache and len(image_paths) == 1:
            cached_result = await image_cache.get(image_paths[0], "auto")
            if cached_result:
                logger.info(f"Using cached result for {image_paths[0]}")
                return cached_result
        
        async with processing_lock:
            # Process through all workflows
            workflow_manager = WorkflowManager()
            result = await workflow_manager.process_all_workflows(
                image_paths=image_paths,
                image_directory=image_directory
            )
            
            # Cache successful single-image results
            if use_cache and len(image_paths) == 1 and any(
                wf_result.get("status") == "success" 
                for wf_result in result.values() 
                if isinstance(wf_result, dict)
            ):
                await image_cache.set(image_paths[0], "auto", result)
            
            return result
                
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "image_paths": image_paths
        }

async def process_single_image_internal(
    image_paths: List[str],
    image_directory: str,
    workflow_type: str
) -> Dict[str, Any]:
    """Process images through a specific workflow with enhanced error handling and state management"""
    start_time = datetime.now()
    workflow_id = f"{workflow_type}_{uuid.uuid4().hex[:8]}"
    
    try:
        # Validate inputs
        if not image_paths:
            raise ValueError("No image paths provided")
        if not os.path.exists(image_directory):
            raise ValueError(f"Image directory not found: {image_directory}")
        
        # Prepare message content with validation
        message_content = []
        failed_images = []
        
        for image_path in image_paths:
            full_path = os.path.join(image_directory, image_path).replace('\\', '/')
            
            # Validate image exists and is readable
            if not os.path.exists(full_path):
                logger.warning(f"Image not found: {full_path}")
                failed_images.append(image_path)
                continue
                
            try:
                # Check file size (limit to 10MB)
                file_size = os.path.getsize(full_path)
                if file_size > 10 * 1024 * 1024:
                    logger.warning(f"Image too large ({file_size} bytes): {full_path}")
                    failed_images.append(image_path)
                    continue
                
                base64_image = encode_image(full_path)
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })
                logger.debug(f"Successfully encoded image: {image_path}")
                
            except Exception as e:
                logger.error(f"Failed to encode image {image_path}: {str(e)}")
                failed_images.append(image_path)
                continue

        if not message_content:
            return {
                "status": "error",
                "message": "No valid images to process",
                "workflow_type": workflow_type,
                "workflow_id": workflow_id,
                "failed_images": failed_images
            }

        # Initialize state with comprehensive metadata
        initial_state = {
            "messages": [HumanMessage(content=message_content)],
            "filename": image_paths[0],
            "all_filenames": image_paths,
            "workflowtype": workflow_type,
            "workflow_id": workflow_id,
            "start_time": start_time.isoformat(),
            "factura": "",
            "tablafacturas": "",
            "tablatarjetas": "",
            "transferencia": "",
            "tarjeta": "",
            "nomina": "",
            "tablanominas": "",
            "metadata": {
                "image_count": len(message_content),
                "failed_count": len(failed_images),
                "source_directory": image_directory
            }
        }
        
        # Get workflow graph
        workflow_manager = WorkflowManager()
        graph = workflow_manager.get_graph(workflow_type)
        
        # Execute workflow with enhanced configuration
        results = []
        last_state = None
        error_count = 0
        
        logger.info(f"Starting workflow {workflow_id} for {len(message_content)} images")
        
        async for event in graph.astream(
            initial_state,
            config={
                "recursion_limit": 30,
                "configurable": {
                    "thread_id": workflow_id,
                    "checkpoint_ns": workflow_type
                }
            }
        ):
            results.append(event)
            last_state = event
            
            # Log progress every 10 events
            if len(results) % 10 == 0:
                logger.debug(f"Workflow {workflow_id} progress: {len(results)} events processed")
            
            # Check for errors in event
            if isinstance(event, dict) and event.get("error"):
                error_count += 1
                logger.warning(f"Error in workflow {workflow_id}: {event.get('error')}")
        
        # Extract final results from last state
        final_results = {
            "status": "success" if error_count == 0 else "partial",
            "workflow_id": workflow_id,
            "image_paths": image_paths,
            "workflow_type": workflow_type,
            "duration": (datetime.now() - start_time).total_seconds(),
            "events_processed": len(results),
            "error_count": error_count,
            "failed_images": failed_images
        }
        
        # Extract workflow-specific results from last state
        if last_state and isinstance(last_state, dict):
            for key in ["factura", "tablafacturas", "tablatarjetas", "transferencia", 
                       "tarjeta", "nomina", "tablanominas"]:
                if key in last_state:
                    final_results[key] = last_state[key]
        
        logger.info(f"Workflow {workflow_id} completed with status: {final_results['status']}")
        return final_results
        
    except Exception as e:
        logger.error(f"Workflow {workflow_id} failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "workflow_id": workflow_id,
            "image_paths": image_paths,
            "workflow_type": workflow_type,
            "duration": (datetime.now() - start_time).total_seconds()
        }

def update_excel(excel_filename: str, processed_results: Dict[str, Any]) -> None:
    """
    Synchronous version of update_excel for backward compatibility.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(update_excel_async(excel_filename, processed_results))
    finally:
        loop.close()

async def process_images_async(
    image_directory: str,
    base_filenames: List[str],
    excel_filename: str,
    max_concurrent: int = 5,
    batch_size: int = 10
) -> Dict[str, Any]:
    """
    Process multiple batches of images asynchronously with improved batching and concurrency.
    
    Args:
        image_directory: Directory containing the images
        base_filenames: List of base filenames to match
        excel_filename: Name of the Excel file to save results
        max_concurrent: Maximum concurrent operations
        batch_size: Size of batches for processing
        
    Returns:
        Summary of processing results
    """
    start_time = datetime.now()
    
    try:
        # Normalize directory path
        image_directory = Path(image_directory).resolve()
        
        # Validate directory exists
        if not image_directory.exists():
            raise ValueError(f"Image directory not found: {image_directory}")
        
        # Get all files in directory
        all_files = {f.name for f in image_directory.iterdir() if f.is_file()}
        
        # Validate and group images into optimal batches
        valid_images = []
        missing_images = []
        
        for filename in base_filenames:
            if filename in all_files:
                valid_images.append(filename)
            else:
                logger.warning(f"Image not found: {filename}")
                missing_images.append(filename)
        
        if not valid_images:
            logger.warning("No valid images found to process")
            return {
                "status": "error",
                "message": "No valid images found",
                "missing_images": missing_images
            }
        
        # Create optimized batches
        batches = [valid_images[i:i + batch_size] for i in range(0, len(valid_images), batch_size)]
        logger.info(f"Processing {len(valid_images)} images in {len(batches)} batches")
        
        # Process batches concurrently with semaphore for rate limiting
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_batch_with_semaphore(batch):
            async with semaphore:
                return await process_single_batch(
                    batch,
                    str(image_directory),
                    excel_filename
                )
        
        # Create tasks for all batches
        tasks = [process_batch_with_semaphore(batch) for batch in batches]
        
        # Process with progress tracking
        results = []
        successful_count = 0
        failed_count = 0
        
        for i, task in enumerate(asyncio.as_completed(tasks)):
            try:
                result = await task
                results.append(result)
                
                if result.get("status") in ["success", "partial"]:
                    successful_count += 1
                    await update_excel_async(excel_filename, result)
                else:
                    failed_count += 1
                    
                # Log progress
                progress = (i + 1) / len(tasks) * 100
                logger.info(f"Processing progress: {progress:.1f}% ({i + 1}/{len(tasks)} batches)")
                
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                failed_count += 1
                results.append({
                    "status": "error",
                    "message": str(e)
                })
        
        # Clear expired cache entries
        await image_cache.clear_expired()
        
        # Generate summary
        duration = (datetime.now() - start_time).total_seconds()
        
        summary = {
            "status": "success" if failed_count == 0 else "partial",
            "total_images": len(valid_images),
            "total_batches": len(batches),
            "successful_batches": successful_count,
            "failed_batches": failed_count,
            "missing_images": missing_images,
            "duration": duration,
            "images_per_second": len(valid_images) / duration if duration > 0 else 0,
            "results": results
        }
        
        logger.info(f"Processing completed: {successful_count}/{len(batches)} batches successful in {duration:.2f}s")
        return summary
        
    except Exception as e:
        logger.error(f"Error in process_images_async: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "duration": (datetime.now() - start_time).total_seconds()
        }

# Keep the synchronous version for backward compatibility
def process_images(
    image_directory: str,
    base_filenames: List[str],
    excel_filename: str,
    max_workers: int = 4
) -> None:
    """
    Synchronous wrapper for backward compatibility.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            process_images_async(
                image_directory,
                base_filenames,
                excel_filename,
                max_concurrent=max_workers
            )
        )
        if result["status"] == "error":
            logger.error(f"Processing failed: {result['message']}")
    finally:
        loop.close()

async def update_excel_async(excel_filename: str, processed_results: Dict[str, Any]) -> None:
    """
    Asynchronously update Excel file with processing results.
    """
    try:
        # Use aiofiles for async file operations
        excel_path = Path(excel_filename + '.xlsx')
        
        # Read existing data or create new
        if excel_path.exists():
            # For now, use sync pandas read (could be improved with async library)
            excel = await asyncio.get_event_loop().run_in_executor(
                None, pd.read_excel, excel_path
            )
        else:
            excel = pd.DataFrame()
        
        # Update with new results
        if processed_results.get("status") in ["success", "partial"]:
            new_rows = []
            
            for image_path in processed_results.get("image_paths", []):
                row_data = {
                    "DOCUMENTO": image_path,
                    "WORKFLOW_ID": processed_results.get("workflow_id"),
                    "STATUS": processed_results.get("status"),
                    "PROCESSED_AT": datetime.now().isoformat()
                }
                
                # Add workflow-specific results
                for key in ["factura", "transferencia", "tarjeta", "nomina"]:
                    if key in processed_results:
                        row_data[key.upper()] = json.dumps(processed_results[key])
                
                new_rows.append(row_data)
            
            # Append new rows
            if new_rows:
                new_df = pd.DataFrame(new_rows)
                excel = pd.concat([excel, new_df], ignore_index=True)
        
        # Save asynchronously
        await asyncio.get_event_loop().run_in_executor(
            None, excel.to_excel, excel_path, index=False
        )
        
        logger.info(f"Successfully updated Excel file: {excel_filename}")
        
    except Exception as e:
        logger.error(f"Error updating Excel file: {str(e)}")