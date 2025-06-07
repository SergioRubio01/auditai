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
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from ..models import AgentState
from ..routes import router
from ..utils import agent_node, factura_post, generate_textract, factura_table_post
from ..utils.create_agent import create_agent
from ..utils.llm import llm10, llm11

table_format_agent = create_agent(
    llm11,
    tools=[],
    system_message="""You are a helpful assistant that formats tables extracted from the textract_agent. 
    
    You will receive a JSON with rows containing CONCEPTO, UNIDADES, and IMPORTE fields. Your task is to validate and return the same JSON structure, ensuring all required fields are present.

    Output example:
    {
        "rows": [
            {
                "CONCEPTO": "Diseño y desarrollo de plan de contenidos",
                "UNIDADES": "",
                "IMPORTE": "521,78"
            }
        ]
    }

    1. For CONCEPTO: Use the full description, combining related fields if split
    2. For UNIDADES: Keep as is, use empty string if not present
    3. For IMPORTE: Ensure it's a numeric string, remove currency symbols if present
    """
)

retrieval_facturas_agent = create_agent(
    llm10,
    tools=[],
    system_message="""You are a helpful assistant that extracts data from images of payrolls and provides the data in a structured format.
    All available information is in the image.
    
    - CIF_CLIENTE: The CIF name of the client (usually labelled after the word CIF. Its format usually contains a letter and 8 numbers)
    - CLIENTE: The name of the client the CIF_CLIENTE is referring to
    - NUMERO_FACTURA: The number of the payroll
    - FECHA_FACTURA: The date of the payroll
    - PROVEEDOR: The name of the provider the CIF_PROVEEDOR is referring to
    - BASE_IMPONIBLE: str
    - CIF_PROVEEDOR: The CIF name of the provider (usually labelled after the word CIF. Its format usually contains a letter and 8 numbers)
    - IRPF: The value of tax money close to the word IRPF (Impuesto sobre la Renta de las Personas Físicas)
    - IVA: The value of money close to the word IVA (Impuesto del Valor Añadido)
    - TOTAL_FACTURA: The value of money that accounts for all payments of the payroll (sometimes it is labelled under the word TOTAL)
        
    Expected output:
    { 
        "CIF_CLIENTE": "<text>",
        "CLIENTE": "<text>",
        "NUMERO_FACTURA": "<text>",
        "FECHA_FACTURA": "<text>",
        "PROVEEDOR": "<text>",
        "BASE_IMPONIBLE": "<text>",
        "CIF_PROVEEDOR": "<text>",
        "IRPF": "<text>",
        "IVA": "<text>",
        "TOTAL_FACTURA": "<text>",
    }
    
    Note: FECHA FACTURA is the date and it is usually given as one of the following: FECHA, DATA, CONTABLE, F. Valor... although preference is set to something that contains the word 'FECHA'/'DATE.
    Note: Be careful when searching for BASE_IMPONIBLE. BASE_IMPONIBLE is different to 'Exento'. You MUST look for the value closer to B. Imponible, BASE IMPONIBLE, Importe, or Base.
    IMPORTANT: You must always write the date in the format DD/MM/YYYY (e.g. 0702 is translated to 07/02, and 12-04-24 is translated to 12/04/2024).
    """
)

# workflow_facturas
workflow_facturas = StateGraph(AgentState)

# Add nodes
workflow_facturas.add_node("factura_post", functools.partial(factura_post))
workflow_facturas.add_node("retrieval_facturas_agent", functools.partial(agent_node, agent=retrieval_facturas_agent, name="retrieval_facturas_agent"))
workflow_facturas.add_node("textract_agent", functools.partial(generate_textract))
workflow_facturas.add_node("table_format_agent", functools.partial(agent_node, agent=table_format_agent, name="table_format_agent"))
workflow_facturas.add_node("table_upload", functools.partial(factura_table_post))
tools_facturas = []
tools_node_facturas = ToolNode(tools_facturas)
workflow_facturas.add_node("tools", tools_node_facturas)

# Add edges - Sequential flow
workflow_facturas.add_edge(START, "textract_agent")

workflow_facturas.add_conditional_edges("textract_agent", router, {
    "continue": "table_format_agent",
    "tools": "tools"
})

workflow_facturas.add_conditional_edges("table_format_agent", router, {
    "continue": "table_upload",
    "tools": "tools",
})

workflow_facturas.add_conditional_edges("table_upload", router, {
    "continue": "table_format_agent",
    END: "retrieval_facturas_agent"
})

workflow_facturas.add_conditional_edges("retrieval_facturas_agent", router, {
    "continue": "factura_post",
    "tools": "tools"
})

workflow_facturas.add_conditional_edges("factura_post", router, {
    "continue": "retrieval_facturas_agent",
    END: END,
})

workflow_facturas.add_conditional_edges("tools", lambda x: x["sender"], {
    "textract_agent": "textract_agent",
    "retrieval_facturas_agent": "retrieval_facturas_agent",
    "table_format_agent": "table_format_agent"
})