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
from ..utils import agent_node, transferencia_post, tarjeta_post, generate_textract, pago_table_post
from ..utils.create_agent import create_agent
from ..utils.llm import llm4, llm5, llm7, llm9, llm12

documenttype_agent = create_agent(
    llm5,
    tools=[],
    system_message="""You are a helpful assistant that extracts TIPO from images of invoices and provides the type of image based on the information from the document.
    There are several possible names:
    - Orden de transferencia
    - Transferencia emitida
    - Adeudo por transferencia
    - Orden de pago
    - Detalle movimiento
    - Certificado bancario
    - Extracto movimiento
    - Tarjeta de credito
    - Arqueo de caja
    
    In case you are not sure, write CONTINUE. Sometimes the message can contain more than one image. In this case, you shall go image by image and write the TIPO for the first image that contains one of the 8 possible names.
    """
)

retrieval_tarjetas_agent = create_agent(
    llm9,
    tools=[],
    system_message="""You are a helpful assistant that extracts information from credit card statements and bank movements.
    You must extract:
    - TIPO (document type)
    - FECHA_VALOR (value date)
    - ORDENANTE (ordering party)
    - BENEFICIARIO (beneficiary)
    - CONCEPTO (concept/description)
    - IMPORTE (amount)
    
    Expected output:
            
    {
        "TIPO": "<text>",
        "FECHA_VALOR": "<text>",
        "ORDENANTE": "<text>",
        "BENEFICIARIO": "<text>",
        "CONCEPTO": "<text>",
        "IMPORTE": "<text>"
    }

    Format your response as a JSON string with these exact field names.
    Dates should be in DD/MM/YYYY format.
    Remove any currency symbols from amounts.
    """
)

retrieval_transferencias_agent = create_agent(
    llm7,
    tools=[],
    system_message="""You are a helpful assistant that extracts information from bank transfer documents.
    You must extract:
    - TIPO (document type)
    - FECHA VALOR (value date)
    - ORDENANTE (ordering party)
    - BENEFICIARIO (beneficiary)
    - CONCEPTO (concept/description)
    - IMPORTE (amount)
    
    Expected output:
            
    {"TIPO":"<text>","FECHA VALOR":"<text>","ORDENANTE":"<text>","BENEFICIARIO":"<text>","CONCEPTO":"<text>","IMPORTE":"<text>"}

    Format your response as a JSON string with these exact field names.
    """
)

table_format_agent = create_agent(
    llm12,
    tools=[],
    system_message="""You are a helpful assistant that saves tables extracted from the textract_agent into the database.
    
    Output example:
    {
        "rows": [
            {
                "CONCEPTO": "Diseño y desarrollo de plan de contenidos",
                "FECHA_VALOR": "DD/MM/YYYY",
                "IMPORTE": "521,78"
            }
        ]
    }
    
    You must return the exact same JSON structure you receive, only making modifications if needed to ensure data quality:

    1. For CONCEPTO: Use the full description, combining related fields if split
    2. For FECHA_VALOR: Format dates as DD/MM/YYYY
    3. For IMPORTE: Ensure it's a numeric string, remove currency symbols if present

    IMPORTANT: Always return a valid JSON with the exact same structure as received. Do not add commentary or additional text.
    Just write FINAL ANSWER when the table is uploaded
    """
)

# workflow_facturas
workflow_pagos = StateGraph(AgentState)

# Add nodes
workflow_pagos.add_node("transferencia_post", functools.partial(transferencia_post))
workflow_pagos.add_node("tarjeta_post", functools.partial(tarjeta_post))
workflow_pagos.add_node("documenttype_agent", functools.partial(agent_node, agent=documenttype_agent, name="documenttype_agent"))
workflow_pagos.add_node("textract_agent", functools.partial(generate_textract))
workflow_pagos.add_node("table_format_agent", functools.partial(agent_node, agent=table_format_agent, name="table_format_agent"))
workflow_pagos.add_node("table_upload_agent", functools.partial(pago_table_post))
workflow_pagos.add_node("retrieval_tarjetas_agent", functools.partial(agent_node, agent=retrieval_tarjetas_agent, name="retrieval_tarjetas_agent"))
workflow_pagos.add_node("retrieval_transferencias_agent", functools.partial(agent_node, agent=retrieval_transferencias_agent, name="retrieval_transferencias_agent"))
tools_pagos = []
tools_node_pagos = ToolNode(tools_pagos)
workflow_pagos.add_node("tools", tools_node_pagos)

# Add edges
workflow_pagos.add_edge(START, "documenttype_agent")
workflow_pagos.add_conditional_edges("documenttype_agent", router, {
    "case1": "retrieval_transferencias_agent",
    "case2": "textract_agent",
    "continue": "documenttype_agent",
    "tools": "tools",
})
workflow_pagos.add_conditional_edges("retrieval_transferencias_agent", router, {
    "continue": "transferencia_post",
    "tools": "tools",
})
workflow_pagos.add_conditional_edges("transferencia_post", router, {
    "continue": "retrieval_transferencias_agent",
    "tools": "tools",
    END: END
})
workflow_pagos.add_conditional_edges("textract_agent", router, {
    "continue": "table_format_agent",
    "tools": "tools"
})
workflow_pagos.add_conditional_edges("table_format_agent", router, {
    "continue": "table_upload_agent",
    "tools": "tools",
})
workflow_pagos.add_conditional_edges("table_upload_agent", router, {
    "continue": "table_format_agent",
    "tools": "tools",
    END: "retrieval_tarjetas_agent"
})
workflow_pagos.add_conditional_edges("retrieval_tarjetas_agent", router, {
    "continue": "tarjeta_post",
    "tools": "tools"
})
workflow_pagos.add_conditional_edges("tarjeta_post", router, {
    "continue": "retrieval_tarjetas_agent",
    "tools": "tools",
    END: END
})

workflow_pagos.add_conditional_edges("tools", lambda x: x["sender"], {
    "retrieval_transferencias_agent": "retrieval_transferencias_agent",
    "retrieval_tarjetas_agent": "retrieval_tarjetas_agent",
    "documenttype_agent":"documenttype_agent",
    "table_format_agent": "table_format_agent",
})