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
from ..utils.llm import llm14, llm13, llm15, llm4, llm3, llm16

table_format_agent = create_agent(
    llm14,
    tools=[],
    system_message="""You are a helpful assistant that formats tables. 
        You receive DESCRIPCION, DEVENGOS, and DEDUCCIONES fields.
        
        Expected output:
        {"fila 1": {"DESCRIPCION": "101 -SALARIO MES", "DEVENGOS": "2717,94", "DEDUCCIONES": ""}, "fila 2": {"DESCRIPCION": "150-ANTIGUEDAD", "DEVENGOS": "146,82", "DEDUCCIONES": ""}, "fila 3": {"DESCRIPCION": "203-COMPLEMENTO", "DEVENGOS": "900,00", "DEDUCCIONES": ""}, "fila 4": {"DESCRIPCION": "205-GRATIFICACION", "DEVENGOS": "400,00", "DEDUCCIONES": ""}}

        1. For DESCRIPCION: Do NOT combine the fields, just keep the original description.
        2. For DEVENGOS: Ensure it's a numeric string, remove currency symbols if present
        3. For DEDUCCIONES: Ensure it's a numeric string, remove currency symbols if present
        4. If a row has DEDUCCIONES, then DEVENGOS should be 0 and vice versa.
        5. If a row has no DEVENGOS nor DEDUCCIONES, then it should be removed.
    """
)

retrieval_nominas_agent = create_agent(
    llm16,
    tools=[],
    system_message="""You are a helpful assistant that extracts data from images of payrolls and provides the data in a structured format.
    CIF is generally present as C.I.F. or CIF, and starts with a letter, not a number.
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
    END: END
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