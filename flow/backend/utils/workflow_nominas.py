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
from ..utils import agent_node, nomina_post, generate_textract, nomina_table_post
from ..utils.create_agent import create_agent
from ..utils.llm import llm14, llm13

table_format_agent = create_agent(
    llm14,
    tools=[],
    system_message="""You are a helpful assistant that formats tables extracted from the textract_agent. 
    
    You will receive a JSON with rows containing CONCEPTO, UNIDADES, and IMPORTE fields. Your task is to validate and return the same JSON structure, ensuring all required fields are present.

    Output example:
    {
        "rows": [
            {
                "DESCRIPCION": "Diseño y desarrollo de plan de contenidos",
                "IMPORTE_UNIDAD": "",
                "UNIDAD": "521,78",
                "DEVENGOS": "23,8",
                "DEDUCCIONES": "12,3"
            }
        ]
    }

    1. For DESCRIPCION: Use the full description, combining related fields if split
    2. For IMPORTE_UNIDAD: Ensure it's a numeric string, remove currency symbols if present
    3. For UNIDAD: Keep as is, use empty string if not present
    4. For DEVENGOS: Ensure it's a numeric string, remove currency symbols if present
    5. For DEDUCCIONES: Ensure it's a numeric string, remove currency symbols if present
    """
)

retrieval_nominas_agent = create_agent(
    llm13,
    tools=[],
    system_message="""You are a helpful assistant that extracts data from images of payrolls and provides the data in a structured format.
    All available information is in the image.
    
    - MES: The month of the payroll
    - FECHA_INICIO: The start date of the payroll
    - FECHA_FIN: The end date of the payroll
    - CIF: The CIF name of the client (usually labelled after the word CIF. Its format usually contains a letter and 8 numbers)
    - TRABAJADOR: The name of the worker the CIF_TRABAJADOR is referring to
    - NAF: The number of the worker
    - NIF: The NIF of the worker
    - CATEGORIA: The category of the worker
    - ANTIGUEDAD: The years of experience of the worker
    - CONTRATO: The type of contract of the worker
    - TOTAL_DEVENGOS: The total of the devengos of the worker
    - TOTAL_DEDUCCIONES: The total of the deducciones of the worker
    - ABSENTISMOS: The total of the absentismos of the worker
    - BC_TEORICA: The base of the worker
    - PRORRATA: The prorata of the worker
    - BC_CON_COMPLEMENTOS: The base of the worker with the complementos
    - TOTAL_SEG_SOCIAL: The total of the social security of the worker
    - BONIFICACIONES_SS_TRABAJADOR: The bonifications of the social security of the worker
    - TOTAL_RETENCIONES: The total of the retentions of the worker
    - TOTAL_RETENCIONES_SS: The total of the retentions of the social security of the worker
    - LIQUIDO_A_PERCIBIR: The total of the liquid to be received of the worker
    - A_ABONAR: The total of the money to be paid of the worker
    - TOTAL_CUOTA_EMPRESARIAL: The total of the company's social security quota
     
    Expected output:
    { 
        "MES": "<text>",
        "FECHA_INICIO": "<text>",
        "FECHA_FIN": "<text>",
        "CIF": "<text>",
        "TRABAJADOR": "<text>",
        "NAF": "<text>",
        "NIF": "<text>",
        "CATEGORIA": "<text>",
        "ANTIGUEDAD": "<text>",
        "CONTRATO": "<text>",
        "TOTAL_DEVENGOS": "<text>",
        "TOTAL_DEDUCCIONES": "<text>",
        "ABSENTISMOS": "<text>",
        "BC_TEORICA": "<text>",
        "PRORRATA": "<text>",
        "BC_CON_COMPLEMENTOS": "<text>",
        "TOTAL_SEG_SOCIAL": "<text>",
        "BONIFICACIONES_SS_TRABAJADOR": "<text>",
        "TOTAL_RETENCIONES": "<text>",
        "TOTAL_RETENCIONES_SS": "<text>",
        "LIQUIDO_A_PERCIBIR": "<text>",
        "A_ABONAR": "<text>",
        "TOTAL_CUOTA_EMPRESARIAL": "<text>",
    }
   """
)

# workflow_nominas
workflow_nominas = StateGraph(AgentState)

# Add nodes
workflow_nominas.add_node("nomina_post", functools.partial(nomina_post))
workflow_nominas.add_node("retrieval_nominas_agent", functools.partial(agent_node, agent=retrieval_nominas_agent, name="retrieval_nominas_agent"))
workflow_nominas.add_node("textract_agent", functools.partial(generate_textract))
workflow_nominas.add_node("table_format_agent", functools.partial(agent_node, agent=table_format_agent, name="table_format_agent"))
workflow_nominas.add_node("table_upload", functools.partial(nomina_table_post))
tools_nominas = []
tools_node_nominas = ToolNode(tools_nominas)
workflow_nominas.add_node("tools", tools_node_nominas)

# Add edges - Sequential flow
workflow_nominas.add_edge(START, "textract_agent")

workflow_nominas.add_conditional_edges("textract_agent", router, {
    "continue": "table_format_agent",
    "tools": "tools"
})

workflow_nominas.add_conditional_edges("table_format_agent", router, {
    "continue": "table_upload",
    "tools": "tools",
})

workflow_nominas.add_conditional_edges("table_upload", router, {
    "continue": "table_format_agent",
    END: "retrieval_nominas_agent"
})

workflow_nominas.add_conditional_edges("retrieval_nominas_agent", router, {
    "continue": "nomina_post",
    "tools": "tools"
})

workflow_nominas.add_conditional_edges("nomina_post", router, {
    "continue": "retrieval_nominas_agent",
    END: END,
})

workflow_nominas.add_conditional_edges("tools", lambda x: x["sender"], {
    "textract_agent": "textract_agent",
    "retrieval_nominas_agent": "retrieval_nominas_agent",
    "table_format_agent": "table_format_agent"
})