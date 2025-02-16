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

from ..models import AgentState
from langgraph.graph import END

def router(state: AgentState) -> str:
    """Route to the appropriate next step in the workflow."""
    last_message = state["messages"][-1]
    sender = state["sender"]
    
    # If there are tool calls, route to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # Safety check - if we've been through supervisor multiple times, end
    FINAL_AGENTS = {
        "factura_post",
        "final_agent",
        "table_upload",
        "nomina_post",
        "transferencia_post",
        "tarjeta_post",
        "textract_agent",
        "table_upload_agent"
    }
    FINAL_ANSWER_MARKER = "FINAL ANSWER"
    

    if sender in FINAL_AGENTS and FINAL_ANSWER_MARKER in last_message.content:
        return END
    
    if sender == "documenttype_agent":
        if last_message.content in ["Orden de transferencia", "Transferencia emitida", "Adeudo por transferencia", "Orden de pago", "Detalle movimiento", "Certificado bancario"]:
            return "case1"
        if last_message.content in ["Tarjeta de credito", "Extracto movimiento", "Arqueo de caja"]:
            return "case2"
        else:
            return "Document type not recognized - must be one of: 'Orden de transferencia', 'Transferencia emitida', 'Adeudo por transferencia', 'Orden de pago', 'Detalle movimiento', 'Certificado bancario', 'Tarjeta de credito', 'Extracto movimiento', 'Arqueo de caja'"

        
    return "continue"