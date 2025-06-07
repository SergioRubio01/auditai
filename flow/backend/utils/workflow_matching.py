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
from ..models import MatchState
from ..routes import router
from ..utils import agent_node, transferencia_post, tarjeta_post, generate_textract, pago_table_post
from ..utils.create_agent import create_agent
from ..utils.llm import llm4

supervisor_agent = create_agent(
    llm4,
    tools=[],
    system_message="""
    """
)

# workflow_facturas
workflow_pagos = StateGraph(MatchState)

# Add nodes
workflow_pagos.add_node("supervisor_agent", functools.partial(agent_node, agent=supervisor_agent, name="supervisor_agent"))
tools_pagos = [transferencia_post, tarjeta_post, generate_textract, pago_table_post]
tools_node_pagos = ToolNode(tools_pagos)
workflow_pagos.add_node("tools", tools_node_pagos)

# Add edges
workflow_pagos.add_edge(START, "documenttype_agent")
workflow_pagos.add_conditional_edges("documenttype_agent", router, {
    "continue": "documenttype_agent",
    "tools": "tools",
})
workflow_pagos.add_conditional_edges("tools", lambda x: x["sender"], {
    "supervisor_agent": "supervisor_agent"
})