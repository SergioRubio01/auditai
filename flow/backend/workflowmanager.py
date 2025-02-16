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
from typing import Dict, Any, List
from pathlib import Path

class WorkflowManager:
    """Manager class to handle different types of workflows."""
    
    def __init__(self):
        self.workflows = {
            'facturas': workflow_facturas,
            'pagos': workflow_pagos,
            'nominas': workflow_nominas
        }
    
    def get_graph(self, workflow_type: str, checkpointer: str = None):
        """
        Get the appropiate graph based on type
        
        Args:
            workflow_type (str): Type of workflow ('facturas' or 'pagos' or 'nominas')
            checkpointer (str, optional): Checkpointer to use for the workflow
        Returns:
            Compiled state graph into a CompiledGraph object
        """
        workflow = self.workflows.get(workflow_type.lower())
        if workflow is None:
            raise ValueError(f"Unsupported workflow type: {workflow_type}. "
                           f"Supported types are: {list(self.workflows.keys())}")
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
        Process images through a specific workflow.
        
        Args:
            workflow_type: Type of workflow ('facturas' or 'pagos' or 'nominas')
            image_paths: List of image paths to process
            image_directory: Directory containing the images
            
        Returns:
            Dict containing processing results
        """
        from .process_images import process_single_image_internal
        
        try:
            result = await process_single_image_internal(
                image_paths=image_paths,
                image_directory=image_directory,
                workflow_type=workflow_type
            )
            return result
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "workflow_type": workflow_type
            }

    async def process_all_workflows(
        self,
        image_paths: List[str],
        image_directory: str
    ) -> Dict[str, Any]:
        """
        Process images through all available workflows concurrently.
        
        Args:
            image_paths: List of image paths to process
            image_directory: Directory containing the images
            
        Returns:
            Dict containing results from all workflows
        """
        tasks = []
        for workflow_type in self.workflows.keys():
            task = self.process_workflow(
                workflow_type=workflow_type,
                image_paths=image_paths,
                image_directory=image_directory
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        return {
            "facturas": next((r for r in results if r.get("workflow_type") == "facturas"), None),
            "pagos": next((r for r in results if r.get("workflow_type") == "pagos"), None),
            "nominas": next((r for r in results if r.get("workflow_type") == "nominas"), None)
        }