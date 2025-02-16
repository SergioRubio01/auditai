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
# 
from langchain_core.messages import ToolMessage, AIMessage
import json
from ..models import DocType, Transferencia, TablaFacturas, FacturaRow, TarjetaRow, TablaTarjetas, TablaNominas, NominaRow
import logging

logger = logging.getLogger(__name__)

def agent_node(state, agent, name):
    # Store the original filename before agent invocation
    original_filename = state.get('filename')
    workflowtype = state.get('workflowtype')
    
    # Invoke agent
    result = agent.invoke(state)
    
    if isinstance(result, AIMessage) and hasattr(result, 'tool_calls'):
        # Fix filename in tool calls if present
        for tool_call in result.tool_calls:
            # Handle both dictionary and object-style tool calls
            if isinstance(tool_call, dict):
                if 'args' in tool_call and isinstance(tool_call['args'], dict) and 'state' in tool_call['args']:
                    tool_call['args']['state']['filename'] = original_filename
                    tool_call['args']['state']['workflowtype'] = workflowtype
            else:
                if hasattr(tool_call, 'args') and isinstance(tool_call.args, dict) and 'state' in tool_call.args:
                    tool_call.args['state']['filename'] = original_filename
                    tool_call.args['state']['workflowtype'] = workflowtype
    
    if isinstance(result, ToolMessage):
        result = AIMessage(content=result.content)
    elif isinstance(result, DocType):
        result = AIMessage(content=result.TIPO)
        new_state = state.copy()
        new_state.update({
            "messages": [result],
            "sender": name,
            "filename": original_filename,
            "workflowtype": workflowtype,
        })
    elif isinstance(result, (Transferencia)):
        formatted_response = {
            "TIPO": result.TIPO,
            "FECHA_VALOR": result.FECHA_VALOR,
            "ORDENANTE": result.ORDENANTE,
            "BENEFICIARIO": result.BENEFICIARIO,
            "CONCEPTO": result.CONCEPTO,
            "IMPORTE": result.IMPORTE
        }
        result = AIMessage(content=json.dumps(formatted_response))
        new_state = state.copy()
        new_state.update({
            "messages": [],
            "sender": name,
            "filename": original_filename,
            "workflowtype": workflowtype,
            "transferencia": result.content
        })
        print(new_state)
    elif isinstance(result, (TarjetaRow)):
        formatted_response = {
            "TIPO": result.TIPO,
            "FECHA_VALOR": result.FECHA_VALOR,
            "ORDENANTE": result.ORDENANTE,
            "BENEFICIARIO": result.BENEFICIARIO,
            "CONCEPTO": result.CONCEPTO,
            "IMPORTE": result.IMPORTE
        }
        result = AIMessage(content=json.dumps(formatted_response))
        new_state = state.copy()
        new_state.update({
            "messages": [],
            "sender": name,
            "filename": original_filename,
            "workflowtype": workflowtype,
            "tarjeta": result.content
        })
    elif isinstance(result, FacturaRow):
        # Convert the list structure to the dictionary structure expected by the Excel tool
        formatted_response = {
            "CIF_CLIENTE": result.CIF_CLIENTE,
            "CLIENTE": result.CLIENTE,
            "NUMERO_FACTURA": result.NUMERO_FACTURA,
            "FECHA_FACTURA": result.FECHA_FACTURA,
            "PROVEEDOR": result.PROVEEDOR,
            "BASE_IMPONIBLE": result.BASE_IMPONIBLE,
            "CIF_PROVEEDOR": result.CIF_PROVEEDOR,
            "IRPF": result.IRPF,
            "IVA": result.IVA,
            "TOTAL_FACTURA": result.TOTAL_FACTURA
        }
        result = AIMessage(content=json.dumps(formatted_response))
        new_state = state.copy()
        new_state.update({
            "messages": [],
            "sender": name,
            "filename": original_filename,
            "workflowtype": workflowtype,
            "factura": result.content
        })
    elif isinstance(result, NominaRow):
        # Convert the list structure to the dictionary structure expected by the Excel tool
        formatted_response = {
            "MES": result.MES,
            "FECHA_INICIO": result.FECHA_INICIO,
            "FECHA_FIN": result.FECHA_FIN,
            "CIF": result.CIF,
            "TRABAJADOR": result.TRABAJADOR,
            "NAF": result.NAF,
            "NIF": result.NIF,
            "CATEGORIA": result.CATEGORIA,
            "ANTIGUEDAD": result.ANTIGUEDAD,
            "CONTRATO": result.CONTRATO,
            "TOTAL_DEVENGOS": result.TOTAL_DEVENGOS,
            "TOTAL_DEDUCCIONES": result.TOTAL_DEDUCCIONES,
            "ABSENTISMOS": result.ABSENTISMOS,
            "BC_TEORICA": result.BC_TEORICA,
            "PRORRATA": result.PRORRATA,
            "BC_CON_COMPLEMENTOS": result.BC_CON_COMPLEMENTOS,
            "TOTAL_SEG_SOCIAL": result.TOTAL_SEG_SOCIAL,
            "BONIFICACIONES_SS_TRABAJADOR": result.BONIFICACIONES_SS_TRABAJADOR,
            "TOTAL_RETENCIONES": result.TOTAL_RETENCIONES,
            "TOTAL_RETENCIONES_SS": result.TOTAL_RETENCIONES_SS,
            "LIQUIDO_A_PERCIBIR": result.LIQUIDO_A_PERCIBIR,
            "A_ABONAR": result.A_ABONAR,
            "TOTAL_CUOTA_EMPRESARIAL": result.TOTAL_CUOTA_EMPRESARIAL,
        }
        result = AIMessage(content=json.dumps(formatted_response))
        new_state = state.copy()
        new_state.update({
            "messages": [],
            "sender": name,
            "filename": original_filename,
            "workflowtype": workflowtype,
            "nomina": result.content
        })
    elif isinstance(result, TablaFacturas):
        # Convert the list structure to the dictionary structure expected by the Excel tool
        formatted_response = {}
        for i, row in enumerate(result.rows, 1):
            formatted_response[f"fila {i}"] = {
                "CONCEPTO": row.CONCEPTO,
                "UNIDADES": row.UNIDADES,
                "IMPORTE": row.IMPORTE
            }
        result = AIMessage(content=json.dumps(formatted_response))
        new_state = state.copy()
        new_state.update({
            "messages": [],
            "sender": name,
            "filename": original_filename,
            "workflowtype": workflowtype,
            "tablafacturas": result.content
        })
    elif isinstance(result, TablaTarjetas):
        # Convert the list structure to the dictionary structure expected by the Excel tool
        formatted_response = {}
        for i, row in enumerate(result.rows, 1):
            formatted_response[f"fila {i}"] = {
                "CONCEPTO": row.CONCEPTO,
                "FECHA_VALOR": row.FECHA_VALOR,
                "IMPORTE": row.IMPORTE
            }
        result = AIMessage(content=json.dumps(formatted_response))
        new_state = state.copy()
        new_state.update({
            "messages": [],
            "sender": name,
            "filename": original_filename,
            "workflowtype": workflowtype,
            "tablatarjetas": result.content
        })
    elif isinstance(result, TablaNominas):
        formatted_response = {}
        for i, row in enumerate(result.rows, 1):
            formatted_response[f"fila {i}"] = {
                "DESCRIPCION": row.DESCRIPCION,
                "IMPORTE_UNIDAD": row.IMPORTE_UNIDAD,
                "UNIDAD": row.UNIDAD,
                "DEVENGOS": row.DEVENGOS,
                "DEDUCCIONES": row.DEDUCCIONES
            }
        result = AIMessage(content=json.dumps(formatted_response))
        new_state = state.copy()
        new_state.update({
            "messages": [],
            "sender": name,
            "filename": original_filename,
            "workflowtype": workflowtype,
            "tablanominas": result.content
        })
    else:
        # Handle dictionary results that might contain nested state
        if isinstance(result, dict):
            if 'state' in result:
                result['state']['filename'] = original_filename
                result['state']['workflowtype'] = workflowtype
            if 'filename' in result:
                result['filename'] = original_filename
            if 'workflowtype' in result:
                result['workflowtype'] = workflowtype
            result = AIMessage(**result.model_dump(exclude={"type", "name", "function"}), name=name)
        else:
            result = AIMessage(**result.model_dump(exclude={"type", "name"}), name=name)
        
        new_state = state.copy()
        new_state.update({
            "messages": [result],
            "sender": name,
            "filename": original_filename,
            "workflowtype": workflowtype
        })
    
    try:
        if result.content.startswith("STOP:"):
            return {
                "status": "error", 
                "message": result.content,
                "filename": original_filename,
                "workflowtype": workflowtype
            }
    except Exception as e:
        logger.error(f"Agent processing error: {str(e)}")
        return {
            "status": "error", 
            "message": str(e),
            "filename": original_filename,
            "workflowtype": workflowtype
        }
    
    return new_state