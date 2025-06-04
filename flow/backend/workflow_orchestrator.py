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

"""
Enhanced Workflow Orchestrator for AutoAudit
This module provides advanced orchestration capabilities for LangGraph workflows
with improved error handling, observability, and performance.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Set, Callable, TypeVar, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import json
from functools import wraps
import traceback

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from .models import AgentState
from .utils import workflow_pagos, workflow_facturas, workflow_nominas
import weave

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar('T')
WorkflowResult = Dict[str, Any]

class WorkflowStatus(Enum):
    """Enum for workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"

class WorkflowPriority(Enum):
    """Priority levels for workflow execution"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class WorkflowContext:
    """Context for workflow execution with metadata"""
    workflow_id: str
    workflow_type: str
    priority: WorkflowPriority = WorkflowPriority.MEDIUM
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_history: List[Dict[str, Any]] = field(default_factory=list)
    checkpoints: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate execution duration"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    def add_error(self, error: Exception, context: str = ""):
        """Add error to history with context"""
        self.error_history.append({
            "timestamp": datetime.now().isoformat(),
            "error": str(error),
            "type": type(error).__name__,
            "context": context,
            "traceback": traceback.format_exc()
        })

class WorkflowMetrics:
    """Collect and track workflow metrics"""
    def __init__(self):
        self.executions: Dict[str, List[float]] = {}
        self.errors: Dict[str, int] = {}
        self.success_count: Dict[str, int] = {}
        
    def record_execution(self, workflow_type: str, duration: float):
        """Record successful execution"""
        if workflow_type not in self.executions:
            self.executions[workflow_type] = []
            self.success_count[workflow_type] = 0
        self.executions[workflow_type].append(duration)
        self.success_count[workflow_type] += 1
        
    def record_error(self, workflow_type: str):
        """Record workflow error"""
        if workflow_type not in self.errors:
            self.errors[workflow_type] = 0
        self.errors[workflow_type] += 1
        
    def get_stats(self, workflow_type: str) -> Dict[str, Any]:
        """Get statistics for a workflow type"""
        executions = self.executions.get(workflow_type, [])
        return {
            "total_executions": len(executions),
            "success_count": self.success_count.get(workflow_type, 0),
            "error_count": self.errors.get(workflow_type, 0),
            "avg_duration": sum(executions) / len(executions) if executions else 0,
            "min_duration": min(executions) if executions else 0,
            "max_duration": max(executions) if executions else 0
        }

class EnhancedWorkflowManager:
    """Enhanced workflow manager with advanced orchestration capabilities"""
    
    def __init__(self, 
                 checkpointer_type: str = "memory",
                 postgres_conn_string: Optional[str] = None,
                 enable_metrics: bool = True,
                 enable_weave: bool = True):
        """
        Initialize the enhanced workflow manager
        
        Args:
            checkpointer_type: Type of checkpointer ("memory" or "postgres")
            postgres_conn_string: Connection string for PostgreSQL checkpointer
            enable_metrics: Enable metrics collection
            enable_weave: Enable Weave tracing
        """
        self.workflows = {
            'facturas': workflow_facturas,
            'pagos': workflow_pagos,
            'nominas': workflow_nominas
        }
        
        # Initialize checkpointer
        if checkpointer_type == "postgres" and postgres_conn_string:
            self.checkpointer = PostgresSaver.from_conn_string(postgres_conn_string)
        else:
            self.checkpointer = MemorySaver()
            
        # Initialize metrics
        self.metrics = WorkflowMetrics() if enable_metrics else None
        
        # Initialize Weave if enabled
        if enable_weave:
            weave.init("autoaudit-workflows")
            
        # Workflow registry for dynamic registration
        self.workflow_registry: Dict[str, StateGraph] = {}
        
        # Active contexts for tracking
        self.active_contexts: Dict[str, WorkflowContext] = {}
        
        # Rate limiting
        self.rate_limiter = asyncio.Semaphore(10)  # Max 10 concurrent workflows
        
        logger.info(f"Enhanced Workflow Manager initialized with {checkpointer_type} checkpointer")
    
    def register_workflow(self, name: str, workflow: StateGraph):
        """Register a new workflow dynamically"""
        self.workflow_registry[name] = workflow
        logger.info(f"Registered new workflow: {name}")
    
    @weave.op()
    async def execute_with_retry(self, 
                                context: WorkflowContext,
                                state: AgentState,
                                config: Optional[Dict[str, Any]] = None) -> WorkflowResult:
        """
        Execute workflow with automatic retry logic
        
        Args:
            context: Workflow execution context
            state: Initial agent state
            config: Workflow configuration
            
        Returns:
            Workflow execution result
        """
        while context.retry_count <= context.max_retries:
            try:
                context.status = WorkflowStatus.RUNNING if context.retry_count == 0 else WorkflowStatus.RETRYING
                result = await self._execute_workflow(context, state, config)
                context.status = WorkflowStatus.COMPLETED
                return result
                
            except Exception as e:
                context.add_error(e, f"Retry {context.retry_count}/{context.max_retries}")
                logger.error(f"Workflow {context.workflow_id} failed: {str(e)}")
                
                if context.retry_count < context.max_retries:
                    context.retry_count += 1
                    wait_time = min(2 ** context.retry_count, 30)  # Exponential backoff
                    logger.info(f"Retrying workflow {context.workflow_id} in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    context.status = WorkflowStatus.FAILED
                    raise
                    
        context.status = WorkflowStatus.FAILED
        raise Exception(f"Workflow {context.workflow_id} failed after {context.max_retries} retries")
    
    async def _execute_workflow(self,
                              context: WorkflowContext,
                              state: AgentState,
                              config: Optional[Dict[str, Any]] = None) -> WorkflowResult:
        """Internal workflow execution with monitoring"""
        async with self.rate_limiter:
            context.start_time = time.time()
            
            try:
                # Get workflow graph
                graph = self.get_graph(context.workflow_type, use_checkpointer=True)
                
                # Configure execution
                exec_config = {
                    "recursion_limit": config.get("recursion_limit", 30) if config else 30,
                    "configurable": {
                        "thread_id": context.workflow_id,
                        "checkpoint_ns": context.workflow_type
                    }
                }
                
                # Execute workflow with streaming
                results = []
                checkpoints_created = 0
                
                async for event in graph.astream(state, config=exec_config):
                    results.append(event)
                    
                    # Create checkpoint every 5 events
                    if len(results) % 5 == 0:
                        checkpoint_id = f"{context.workflow_id}_cp_{checkpoints_created}"
                        context.checkpoints.append(checkpoint_id)
                        checkpoints_created += 1
                        logger.debug(f"Created checkpoint: {checkpoint_id}")
                
                context.end_time = time.time()
                
                # Record metrics
                if self.metrics:
                    self.metrics.record_execution(context.workflow_type, context.duration)
                
                return {
                    "status": "success",
                    "workflow_id": context.workflow_id,
                    "workflow_type": context.workflow_type,
                    "duration": context.duration,
                    "results": results,
                    "checkpoints": context.checkpoints
                }
                
            except Exception as e:
                context.end_time = time.time()
                if self.metrics:
                    self.metrics.record_error(context.workflow_type)
                raise
    
    def get_graph(self, workflow_type: str, use_checkpointer: bool = True):
        """Get compiled workflow graph with optional checkpointer"""
        # Check registry first
        workflow = self.workflow_registry.get(workflow_type) or self.workflows.get(workflow_type.lower())
        
        if workflow is None:
            raise ValueError(f"Unsupported workflow type: {workflow_type}")
            
        if use_checkpointer:
            return workflow.compile(checkpointer=self.checkpointer)
        return workflow.compile()
    
    async def execute_parallel_workflows(self,
                                       workflows: List[Dict[str, Any]],
                                       max_concurrent: int = 5) -> List[WorkflowResult]:
        """
        Execute multiple workflows in parallel with concurrency control
        
        Args:
            workflows: List of workflow configurations
            max_concurrent: Maximum concurrent executions
            
        Returns:
            List of workflow results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_semaphore(workflow_config):
            async with semaphore:
                context = WorkflowContext(
                    workflow_id=workflow_config.get("id", str(time.time())),
                    workflow_type=workflow_config["type"],
                    priority=WorkflowPriority(workflow_config.get("priority", 2)),
                    metadata=workflow_config.get("metadata", {})
                )
                
                self.active_contexts[context.workflow_id] = context
                
                try:
                    result = await self.execute_with_retry(
                        context=context,
                        state=workflow_config["state"],
                        config=workflow_config.get("config")
                    )
                    return result
                finally:
                    del self.active_contexts[context.workflow_id]
        
        # Sort by priority
        sorted_workflows = sorted(workflows, key=lambda w: w.get("priority", 2), reverse=True)
        
        # Execute workflows
        tasks = [execute_with_semaphore(wf) for wf in sorted_workflows]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    "status": "error",
                    "error": str(result)
                })
            else:
                processed_results.append(result)
                
        return processed_results
    
    async def execute_workflow_chain(self,
                                   chain: List[Dict[str, Any]],
                                   initial_state: AgentState) -> List[WorkflowResult]:
        """
        Execute a chain of workflows where output of one feeds into the next
        
        Args:
            chain: List of workflow configurations in order
            initial_state: Initial state for the first workflow
            
        Returns:
            List of results from each workflow in the chain
        """
        results = []
        current_state = initial_state
        
        for i, workflow_config in enumerate(chain):
            context = WorkflowContext(
                workflow_id=f"chain_{time.time()}_{i}",
                workflow_type=workflow_config["type"],
                metadata={"chain_position": i, "chain_length": len(chain)}
            )
            
            try:
                # Execute workflow
                result = await self.execute_with_retry(
                    context=context,
                    state=current_state,
                    config=workflow_config.get("config")
                )
                
                results.append(result)
                
                # Extract state for next workflow
                if i < len(chain) - 1 and "state_transformer" in workflow_config:
                    current_state = workflow_config["state_transformer"](result, current_state)
                    
            except Exception as e:
                logger.error(f"Chain execution failed at position {i}: {str(e)}")
                results.append({
                    "status": "error",
                    "error": str(e),
                    "chain_position": i
                })
                break
                
        return results
    
    def get_active_workflows(self) -> List[Dict[str, Any]]:
        """Get information about currently active workflows"""
        return [
            {
                "workflow_id": ctx.workflow_id,
                "workflow_type": ctx.workflow_type,
                "status": ctx.status.value,
                "priority": ctx.priority.value,
                "retry_count": ctx.retry_count,
                "duration": time.time() - ctx.start_time if ctx.start_time else 0
            }
            for ctx in self.active_contexts.values()
        ]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        if not self.metrics:
            return {"metrics_enabled": False}
            
        summary = {"metrics_enabled": True, "workflows": {}}
        
        for workflow_type in self.workflows.keys():
            summary["workflows"][workflow_type] = self.metrics.get_stats(workflow_type)
            
        return summary
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel an active workflow"""
        if workflow_id in self.active_contexts:
            context = self.active_contexts[workflow_id]
            context.status = WorkflowStatus.CANCELLED
            logger.info(f"Cancelled workflow: {workflow_id}")
            return True
        return False

# Decorator for workflow monitoring
def monitor_workflow(workflow_type: str):
    """Decorator to monitor workflow execution"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            context = {
                "workflow_type": workflow_type,
                "start_time": datetime.now().isoformat(),
                "args": str(args)[:100],  # Truncate for logging
                "kwargs": str(kwargs)[:100]
            }
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.info(f"Workflow {workflow_type} completed in {duration:.2f}s")
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Workflow {workflow_type} failed after {duration:.2f}s: {str(e)}")
                raise
                
        return wrapper
    return decorator

# Example usage of the enhanced orchestrator
if __name__ == "__main__":
    async def example_usage():
        # Initialize manager
        manager = EnhancedWorkflowManager(checkpointer_type="memory")
        
        # Example: Execute single workflow with retry
        context = WorkflowContext(
            workflow_id="test_001",
            workflow_type="facturas",
            priority=WorkflowPriority.HIGH
        )
        
        state = AgentState(
            messages=[HumanMessage(content="Process this invoice")],
            filename="invoice_001.pdf"
        )
        
        result = await manager.execute_with_retry(context, state)
        print(f"Result: {result}")
        
        # Example: Execute parallel workflows
        workflows = [
            {
                "id": f"parallel_{i}",
                "type": "facturas",
                "state": state,
                "priority": 2
            }
            for i in range(3)
        ]
        
        results = await manager.execute_parallel_workflows(workflows)
        print(f"Parallel results: {results}")
        
        # Get metrics
        metrics = manager.get_metrics_summary()
        print(f"Metrics: {metrics}")
    
    # Run example
    asyncio.run(example_usage())