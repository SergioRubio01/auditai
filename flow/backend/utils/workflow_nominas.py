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

import functools
import logging
from typing import Dict, Any, List
from datetime import datetime
import asyncio
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from ..models import AgentState
from ..routes import router
from ..utils import agent_node, nomina_post, generate_textract, nomina_table_post
from ..utils.create_agent import create_agent
from ..utils.llm import llm14, llm13, llm15, llm4, llm3, llm16

# Configure logging
logger = logging.getLogger(__name__)

# Error handling wrapper for nodes
def error_handling_node(func):
    """Decorator to add error handling to workflow nodes"""
    @functools.wraps(func)
    async def wrapper(state: AgentState) -> AgentState:
        node_name = func.__name__
        start_time = datetime.now()
        
        try:
            logger.info(f"Executing node: {node_name}")
            result = await func(state)
            
            # Add execution metadata
            if "metadata" not in result:
                result["metadata"] = {}
            
            result["metadata"][f"{node_name}_duration"] = (
                datetime.now() - start_time
            ).total_seconds()
            result["metadata"][f"{node_name}_success"] = True
            
            logger.info(f"Node {node_name} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in node {node_name}: {str(e)}", exc_info=True)
            
            # Add error to state
            error_message = AIMessage(
                content=f"Error in {node_name}: {str(e)}",
                additional_kwargs={"error": True, "node": node_name}
            )
            
            state["messages"].append(error_message)
            
            if "metadata" not in state:
                state["metadata"] = {}
            
            state["metadata"][f"{node_name}_error"] = str(e)
            state["metadata"][f"{node_name}_duration"] = (
                datetime.now() - start_time
            ).total_seconds()
            state["metadata"][f"{node_name}_success"] = False
            
            return state
    
    return wrapper

# Enhanced agent with validation and retry capabilities
table_format_agent = create_agent(
    llm14,
    tools=[],
    system_message="""You are a helpful assistant that formats tables with strict validation.
        You receive DESCRIPCION, DEVENGOS, and DEDUCCIONES fields.
        
        Expected output format:
        {"fila 1": {"DESCRIPCION": "101 -SALARIO MES", "DEVENGOS": "2717,94", "DEDUCCIONES": ""}, "fila 2": {"DESCRIPCION": "150-ANTIGUEDAD", "DEVENGOS": "146,82", "DEDUCCIONES": ""}, "fila 3": {"DESCRIPCION": "203-COMPLEMENTO", "DEVENGOS": "900,00", "DEDUCCIONES": ""}, "fila 4": {"DESCRIPCION": "205-GRATIFICACION", "DEVENGOS": "400,00", "DEDUCCIONES": ""}}

        STRICT RULES:
        1. For DESCRIPCION: Keep the original description exactly as provided
        2. For DEVENGOS: Must be a numeric string (e.g., "2717,94"), remove currency symbols
        3. For DEDUCCIONES: Must be a numeric string or empty string
        4. If a row has DEDUCCIONES, then DEVENGOS must be "0" or empty
        5. If a row has DEVENGOS, then DEDUCCIONES must be "0" or empty
        6. Remove any row with neither DEVENGOS nor DEDUCCIONES
        7. Validate all numeric values are properly formatted
        
        IMPORTANT: Return ONLY valid JSON without any additional text or explanations.
    """
)

retrieval_nominas_agent = create_agent(
    llm16,
    tools=[],
    system_message="""You are an expert payroll data extractor with validation capabilities.
    
    Extract the following fields from payroll documents:
    - CIF: Company tax ID (starts with a letter, format: X########)
    - TRABAJADOR: Employee name
    - NAF: Social Security affiliation number
    - NIF: Employee tax ID
    - CATEGORIA: Job category/position
    - ANTIGUEDAD: Seniority date
    - CONTRATO: Contract type
    - TOTAL_DEVENGOS: Total earnings
    - TOTAL_DEDUCCIONES: Total deductions
    - LIQUIDO_A_PERCIBIR: Net pay
    
    VALIDATION RULES:
    - CIF must start with a letter and have 9 characters
    - Monetary values must be numeric
    - Dates must be in DD/MM/YYYY format
    - All required fields must be present
    
    Return data in structured JSON format with field validation.
   """
)

# Enhanced node functions with validation
@error_handling_node
async def enhanced_textract_node(state: AgentState) -> AgentState:
    """Enhanced Textract node with retry and validation"""
    max_retries = 3
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            # Call original textract function
            result = await generate_textract(state)
            
            # Validate textract output
            if "messages" in result and result["messages"]:
                last_message = result["messages"][-1]
                if hasattr(last_message, "content") and last_message.content:
                    logger.info("Textract extraction successful")
                    return result
            
            if attempt < max_retries - 1:
                logger.warning(f"Textract attempt {attempt + 1} failed, retrying...")
                await asyncio.sleep(retry_delay * (attempt + 1))
            
        except Exception as e:
            logger.error(f"Textract error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (attempt + 1))
            else:
                raise
    
    raise Exception("Textract failed after all retries")

@error_handling_node
async def enhanced_table_format_node(state: AgentState) -> AgentState:
    """Enhanced table formatting with validation"""
    # Run the agent
    result = await agent_node(table_format_agent, "table_format_agent")(state)
    
    # Validate table format
    if "messages" in result and result["messages"]:
        last_message = result["messages"][-1]
        if hasattr(last_message, "content"):
            try:
                # Attempt to parse JSON to validate format
                import json
                table_data = json.loads(last_message.content)
                
                # Validate each row
                for row_key, row_data in table_data.items():
                    if not isinstance(row_data, dict):
                        raise ValueError(f"Invalid row format: {row_key}")
                    
                    required_fields = ["DESCRIPCION", "DEVENGOS", "DEDUCCIONES"]
                    for field in required_fields:
                        if field not in row_data:
                            raise ValueError(f"Missing field {field} in row {row_key}")
                
                logger.info("Table format validation successful")
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in table format: {str(e)}")
                # Add error message to state
                error_msg = AIMessage(
                    content="Table formatting failed: Invalid JSON format",
                    additional_kwargs={"error": True, "validation_error": str(e)}
                )
                result["messages"].append(error_msg)
            
            except ValueError as e:
                logger.error(f"Table validation error: {str(e)}")
                error_msg = AIMessage(
                    content=f"Table validation failed: {str(e)}",
                    additional_kwargs={"error": True, "validation_error": str(e)}
                )
                result["messages"].append(error_msg)
    
    return result

@error_handling_node
async def enhanced_nomina_retrieval_node(state: AgentState) -> AgentState:
    """Enhanced nomina retrieval with data validation"""
    # Run the agent
    result = await agent_node(retrieval_nominas_agent, "retrieval_nominas_agent")(state)
    
    # Validate extracted data
    if "nomina" in result and result["nomina"]:
        nomina_data = result["nomina"]
        
        # Validate CIF format
        if "CIF" in nomina_data:
            cif = nomina_data["CIF"]
            if not (isinstance(cif, str) and len(cif) == 9 and cif[0].isalpha()):
                logger.warning(f"Invalid CIF format: {cif}")
                nomina_data["CIF_validation"] = "invalid"
        
        # Validate numeric fields
        numeric_fields = [
            "TOTAL_DEVENGOS", "TOTAL_DEDUCCIONES", "LIQUIDO_A_PERCIBIR",
            "TOTAL_SEG_SOCIAL", "TOTAL_RETENCIONES"
        ]
        
        for field in numeric_fields:
            if field in nomina_data:
                try:
                    # Try to convert to float to validate
                    value = str(nomina_data[field]).replace(",", ".")
                    float(value)
                except ValueError:
                    logger.warning(f"Invalid numeric value for {field}: {nomina_data[field]}")
                    nomina_data[f"{field}_validation"] = "invalid"
        
        result["nomina"] = nomina_data
    
    return result

# Enhanced workflow with better error handling and monitoring
workflow_nominas = StateGraph(AgentState)

# Add entry node for initialization and validation
async def entry_node(state: AgentState) -> AgentState:
    """Entry point with state initialization and validation"""
    logger.info(f"Starting nominas workflow for file: {state.get('filename', 'unknown')}")
    
    # Initialize metadata if not present
    if "metadata" not in state:
        state["metadata"] = {}
    
    state["metadata"]["workflow_start"] = datetime.now().isoformat()
    state["metadata"]["workflow_type"] = "nominas"
    
    # Validate required fields
    required_fields = ["messages", "filename"]
    for field in required_fields:
        if field not in state or not state[field]:
            raise ValueError(f"Missing required field: {field}")
    
    return state

# Add monitoring node
async def monitoring_node(state: AgentState) -> AgentState:
    """Node for monitoring and metrics collection"""
    if "metadata" in state:
        # Calculate workflow duration
        if "workflow_start" in state["metadata"]:
            start_time = datetime.fromisoformat(state["metadata"]["workflow_start"])
            duration = (datetime.now() - start_time).total_seconds()
            state["metadata"]["workflow_duration"] = duration
        
        # Count successful nodes
        successful_nodes = sum(
            1 for key in state["metadata"] 
            if key.endswith("_success") and state["metadata"][key]
        )
        state["metadata"]["successful_nodes"] = successful_nodes
        
        # Log summary
        logger.info(
            f"Workflow completed: duration={state['metadata'].get('workflow_duration', 0):.2f}s, "
            f"successful_nodes={successful_nodes}"
        )
    
    return state

# Add nodes with enhanced versions
workflow_nominas.add_node("entry", entry_node)
workflow_nominas.add_node("nomina_post", error_handling_node(nomina_post))
workflow_nominas.add_node("retrieval_nominas_agent", enhanced_nomina_retrieval_node)
workflow_nominas.add_node("textract_agent", enhanced_textract_node)
workflow_nominas.add_node("table_format_agent", enhanced_table_format_node)
workflow_nominas.add_node("table_upload", error_handling_node(nomina_table_post))
workflow_nominas.add_node("monitoring", monitoring_node)

# Tools node remains the same
tools_nominas = []
tools_node_nominas = ToolNode(tools_nominas)
workflow_nominas.add_node("tools", tools_node_nominas)

# Enhanced router with error handling
def enhanced_router(state: AgentState) -> str:
    """Enhanced router with error detection and recovery"""
    # Check for errors in the last message
    if state.get("messages"):
        last_message = state["messages"][-1]
        if hasattr(last_message, "additional_kwargs") and last_message.additional_kwargs.get("error"):
            logger.warning("Error detected in workflow, attempting recovery")
            
            # Determine recovery strategy based on error context
            error_node = last_message.additional_kwargs.get("node")
            if error_node == "textract_agent":
                # If textract failed, try retrieval agent instead
                return "retrieval_nominas_agent"
            elif error_node in ["table_format_agent", "table_upload"]:
                # Skip table processing if it's failing
                return "monitoring"
    
    # Use original router for normal flow
    return router(state)

# Add edges with enhanced flow
workflow_nominas.add_edge(START, "entry")
workflow_nominas.add_edge("entry", "textract_agent")

# Parallel processing branch
workflow_nominas.add_conditional_edges("textract_agent", enhanced_router, {
    "continue": "table_format_agent",
    "retrieval_nominas_agent": "retrieval_nominas_agent",
    "tools": "tools",
    "monitoring": "monitoring"
})

workflow_nominas.add_conditional_edges("table_format_agent", enhanced_router, {
    "continue": "table_upload",
    "tools": "tools",
    "monitoring": "monitoring"
})

workflow_nominas.add_conditional_edges("table_upload", enhanced_router, {
    "continue": "retrieval_nominas_agent",
    "monitoring": "monitoring",
    END: END
})

workflow_nominas.add_conditional_edges("retrieval_nominas_agent", enhanced_router, {
    "continue": "nomina_post",
    "tools": "tools",
    "monitoring": "monitoring"
})

workflow_nominas.add_conditional_edges("nomina_post", enhanced_router, {
    "continue": "monitoring",
    "retrieval_nominas_agent": "retrieval_nominas_agent",
    END: END
})

# Monitoring always goes to END
workflow_nominas.add_edge("monitoring", END)

# Tools routing remains the same but with error handling
workflow_nominas.add_conditional_edges("tools", lambda x: x.get("sender", "monitoring"), {
    "textract_agent": "textract_agent",
    "retrieval_nominas_agent": "retrieval_nominas_agent",
    "table_format_agent": "table_format_agent",
    "monitoring": "monitoring"  # Default to monitoring if sender unknown
})