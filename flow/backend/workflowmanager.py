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

from .models import AgentState
from .utils import workflow_pagos, workflow_facturas, workflow_nominas
import asyncio
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import logging
import time
from datetime import datetime
import random
from enum import Enum
from dataclasses import dataclass
import json

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class RetryPolicy:
    """Retry policy configuration"""
    max_retries: int = 3
    backoff_factor: float = 2.0
    max_backoff: float = 60.0
    jitter: bool = True
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and optional jitter"""
        delay = min(self.backoff_factor ** attempt, self.max_backoff)
        if self.jitter:
            delay += random.uniform(0, 0.1 * delay)
        return delay

@dataclass
class WorkflowMetrics:
    """Metrics for workflow execution"""
    start_time: float
    end_time: Optional[float] = None
    status: WorkflowStatus = WorkflowStatus.PENDING
    retry_count: int = 0
    error_message: Optional[str] = None
    
    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

class WorkflowManager:
    """Enhanced manager class to handle different types of workflows with improved orchestration."""
    
    def __init__(self, retry_policy: Optional[RetryPolicy] = None, enable_metrics: bool = True):
        self.workflows = {
            'facturas': workflow_facturas,
            'pagos': workflow_pagos,
            'nominas': workflow_nominas
        }
        self.retry_policy = retry_policy or RetryPolicy()
        self.enable_metrics = enable_metrics
        self.metrics_history: Dict[str, List[WorkflowMetrics]] = {}
        self._semaphore = asyncio.Semaphore(10)  # Limit concurrent workflows
        logger.info("WorkflowManager initialized with enhanced orchestration")
    
    def get_graph(self, workflow_type: str, checkpointer: str = None):
        """
        Get the appropriate graph based on type with validation
        
        Args:
            workflow_type (str): Type of workflow ('facturas' or 'pagos' or 'nominas')
            checkpointer (str, optional): Checkpointer to use for the workflow
        Returns:
            Compiled state graph into a CompiledGraph object
        """
        workflow = self.workflows.get(workflow_type.lower())
        if workflow is None:
            logger.error(f"Unsupported workflow type requested: {workflow_type}")
            raise ValueError(f"Unsupported workflow type: {workflow_type}. "
                           f"Supported types are: {list(self.workflows.keys())}")
        
        logger.debug(f"Compiling workflow graph for type: {workflow_type}")
        if checkpointer:
            return workflow.compile(checkpointer=checkpointer)
        return workflow.compile()
    
    def invoke(self, workflow_type: str, state: AgentState):
        """
        Invoke the specified workflow with given configuration.
        
        Args:
            workflow_type (str): Type of workflow ('facturas' or 'pagos' or 'nominas')
            config (dict): Configuration/input for the workflow
            
        Returns:
            Workflow execution result
        """
        workflow = self.get_workflow(workflow_type)
        return workflow.invoke(state)

    async def process_workflow(
        self,
        workflow_type: str,
        image_paths: List[str],
        image_directory: str
    ) -> Dict[str, Any]:
        """
        Process images through a specific workflow with retry logic and metrics.
        
        Args:
            workflow_type: Type of workflow ('facturas' or 'pagos' or 'nominas')
            image_paths: List of image paths to process
            image_directory: Directory containing the images
            
        Returns:
            Dict containing processing results
        """
        from .process_images import process_single_image_internal
        
        metrics = WorkflowMetrics(start_time=time.time())
        workflow_id = f"{workflow_type}_{int(time.time() * 1000)}"
        
        async with self._semaphore:  # Rate limiting
            for attempt in range(self.retry_policy.max_retries + 1):
                try:
                    metrics.status = WorkflowStatus.RUNNING if attempt == 0 else WorkflowStatus.RETRYING
                    metrics.retry_count = attempt
                    
                    logger.info(f"Processing workflow {workflow_id} (attempt {attempt + 1}/{self.retry_policy.max_retries + 1})")
                    
                    result = await process_single_image_internal(
                        image_paths=image_paths,
                        image_directory=image_directory,
                        workflow_type=workflow_type
                    )
                    
                    metrics.status = WorkflowStatus.COMPLETED
                    metrics.end_time = time.time()
                    
                    # Store metrics
                    if self.enable_metrics:
                        self._store_metrics(workflow_type, metrics)
                    
                    logger.info(f"Workflow {workflow_id} completed successfully in {metrics.duration:.2f}s")
                    return {
                        **result,
                        "workflow_id": workflow_id,
                        "duration": metrics.duration,
                        "retry_count": metrics.retry_count
                    }
                    
                except Exception as e:
                    metrics.error_message = str(e)
                    logger.error(f"Workflow {workflow_id} failed on attempt {attempt + 1}: {str(e)}")
                    
                    if attempt < self.retry_policy.max_retries:
                        delay = self.retry_policy.calculate_delay(attempt)
                        logger.info(f"Retrying workflow {workflow_id} in {delay:.2f}s...")
                        await asyncio.sleep(delay)
                    else:
                        metrics.status = WorkflowStatus.FAILED
                        metrics.end_time = time.time()
                        
                        if self.enable_metrics:
                            self._store_metrics(workflow_type, metrics)
                        
                        return {
                            "status": "error",
                            "message": str(e),
                            "workflow_type": workflow_type,
                            "workflow_id": workflow_id,
                            "retry_count": metrics.retry_count
                        }

    async def process_all_workflows(
        self,
        image_paths: List[str],
        image_directory: str
    ) -> Dict[str, Any]:
        """
        Process images through all available workflows concurrently with improved error handling.
        
        Args:
            image_paths: List of image paths to process
            image_directory: Directory containing the images
            
        Returns:
            Dict containing results from all workflows
        """
        start_time = time.time()
        logger.info(f"Starting parallel processing of {len(self.workflows)} workflows")
        
        tasks = []
        for workflow_type in self.workflows.keys():
            task = self.process_workflow(
                workflow_type=workflow_type,
                image_paths=image_paths,
                image_directory=image_directory
            )
            tasks.append((workflow_type, task))
        
        # Execute with proper exception handling
        results = {}
        workflow_tasks = [(wf_type, asyncio.create_task(task)) for wf_type, task in tasks]
        
        for workflow_type, task in workflow_tasks:
            try:
                result = await task
                results[workflow_type] = result
            except Exception as e:
                logger.error(f"Failed to process workflow {workflow_type}: {str(e)}")
                results[workflow_type] = {
                    "status": "error",
                    "message": str(e),
                    "workflow_type": workflow_type
                }
        
        total_duration = time.time() - start_time
        logger.info(f"Completed parallel processing in {total_duration:.2f}s")
        
        return {
            **results,
            "total_duration": total_duration,
            "processed_at": datetime.now().isoformat()
        }
    
    def _store_metrics(self, workflow_type: str, metrics: WorkflowMetrics):
        """Store metrics for analysis"""
        if workflow_type not in self.metrics_history:
            self.metrics_history[workflow_type] = []
        self.metrics_history[workflow_type].append(metrics)
        
        # Keep only last 1000 metrics per workflow
        if len(self.metrics_history[workflow_type]) > 1000:
            self.metrics_history[workflow_type] = self.metrics_history[workflow_type][-1000:]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of workflow metrics"""
        summary = {}
        
        for workflow_type, metrics_list in self.metrics_history.items():
            if not metrics_list:
                continue
                
            completed = [m for m in metrics_list if m.status == WorkflowStatus.COMPLETED]
            failed = [m for m in metrics_list if m.status == WorkflowStatus.FAILED]
            
            durations = [m.duration for m in completed if m.duration]
            
            summary[workflow_type] = {
                "total_executions": len(metrics_list),
                "successful": len(completed),
                "failed": len(failed),
                "avg_duration": sum(durations) / len(durations) if durations else 0,
                "min_duration": min(durations) if durations else 0,
                "max_duration": max(durations) if durations else 0,
                "avg_retry_count": sum(m.retry_count for m in metrics_list) / len(metrics_list)
            }
        
        return summary