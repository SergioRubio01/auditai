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
import os
from langchain_core.messages import ToolMessage, AIMessage
import json

import weave
from ..models import (
    DocType, Transferencia, TablaFacturas, FacturaRow, 
    TarjetaRow, TablaTarjetas, TablaNominas, NominaRow,
    AgentState
)
import logging
from dotenv import load_dotenv
from ..models.AgentState import validate_nif, extract_nif_from_text, validate_cif

logger = logging.getLogger(__name__)
load_dotenv()

client = weave.init("sergioaeroai-grant-thornton/autoaudit")

def correct_cif_ocr_errors(cif: str) -> str:
    """
    Correct common OCR errors in CIF numbers
    - Replace leading 0 with Q (common OCR misread)
    - Replace leading 0 with other common CIF letters if that makes it valid
    """
    if not cif:
        return cif
        
    cif = cif.strip().upper()
    
    # Pad with leading zero if needed
    if len(cif) == 8:
        cif = '0' + cif
        
    # Check if it already has a valid format
    is_valid, _ = validate_cif(cif)
    if is_valid:
        return cif
        
    # If CIF starts with 0, try replacing with Q (common OCR error)
    if cif.startswith('0'):
        corrected_cif = 'Q' + cif[1:]
        is_valid, _ = validate_cif(corrected_cif)
        if is_valid:
            logger.info(f"Corrected CIF from {cif} to {corrected_cif}")
            return corrected_cif
            
    # Try other common CIF starting letters if it starts with 0
    if cif.startswith('0'):
        for letter in "DOQB":
            corrected_cif = letter + cif[1:]
            is_valid, _ = validate_cif(corrected_cif)
            if is_valid:
                logger.info(f"Corrected CIF from {cif} to {corrected_cif}")
                return corrected_cif
                
    return cif

@weave.op()
def agent_node(state, agent, name):
    # Store the original filename before agent invocation
    original_filename = state.get('filename')
    workflowtype = state.get('workflowtype')
    
    # Invoke agent
    result = agent.invoke(state)
    
    def validate_and_extract_nif(content):
        """Helper function to validate and extract NIF from content"""
        modified = False
        modification_comment = ""
        
        if isinstance(content, dict) and 'NIF' in content:
            nif = content['NIF']
            original_nif = nif
            
            is_valid, error = validate_nif(nif)
            if not is_valid:
                # Try to extract NIF from the entire content if the direct NIF is invalid
                extracted_nif, extract_error = extract_nif_from_text(json.dumps(content))
                if extracted_nif:
                    content['NIF'] = extracted_nif
                    modified = True
                    modification_comment = f"Modified NIF: Original={original_nif}, Extracted={extracted_nif}"
                    logger.info(f"Extracted valid NIF {extracted_nif} from content")
                else:
                    logger.warning(f"Invalid NIF in content: {error}")
            
            # Preserve existing comments when adding new ones
            if modified:
                if 'COMMENTS' in content and content['COMMENTS']:
                    content['COMMENTS'] = f"{content['COMMENTS']}; {modification_comment}"
                else:
                    content['COMMENTS'] = modification_comment
                
        return content
    
    def validate_and_extract_cif(content):
        """Helper function to validate and extract CIF from content"""
        modified = False
        modification_comment = ""
        
        if isinstance(content, dict) and 'CIF' in content:
            cif = content['CIF']
            original_cif = cif
            
            # First try to correct common OCR errors
            corrected_cif = correct_cif_ocr_errors(cif)
            if corrected_cif != cif:
                content['CIF'] = corrected_cif
                cif = corrected_cif
                modified = True
                modification_comment = f"Modified CIF: Original={original_cif}, Corrected={corrected_cif}"
            
            is_valid, error = validate_cif(cif)
            if not is_valid:
                # Try to extract CIF from the entire content if the direct CIF is invalid
                extracted_cif, extract_error = extract_nif_from_text(json.dumps(content))
                if extracted_cif:
                    content['CIF'] = extracted_cif
                    modified = True
                    if modification_comment:
                        modification_comment += f"; Further extracted to {extracted_cif}"
                    else:
                        modification_comment = f"Modified CIF: Original={original_cif}, Extracted={extracted_cif}"
                    logger.info(f"Extracted valid CIF {extracted_cif} from content")
                else:
                    logger.warning(f"Invalid CIF in content: {error}")

            # Always set the comment if there was a modification
            if modified:
                content['COMMENTS'] = modification_comment
                
        return content

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
        # Apply CIF validation/correction to both client and provider CIFs
        if 'CIF_CLIENTE' in formatted_response:
            client_cif = formatted_response['CIF_CLIENTE']
            formatted_response['CIF_CLIENTE'] = correct_cif_ocr_errors(client_cif)
            
        if 'CIF_PROVEEDOR' in formatted_response:
            provider_cif = formatted_response['CIF_PROVEEDOR']
            formatted_response['CIF_PROVEEDOR'] = correct_cif_ocr_errors(provider_cif)
            
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
            "COMMENTS": ""  # Initialize with empty string to ensure validation messages are captured
        }
        
        # Apply both CIF and NIF validation/correction
        formatted_response = validate_and_extract_cif(formatted_response)
        formatted_response = validate_and_extract_nif(formatted_response)
        
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
        if isinstance(result.content, str) and result.content.startswith("STOP:"):
            return {
                "status": "error", 
                "message": result.content,
                "filename": original_filename,
                "workflowtype": workflowtype
            }
        
        # Add NIF/CIF validation for dictionary results
        if isinstance(result.content, str):
            try:
                content_dict = json.loads(result.content)
                if isinstance(content_dict, dict):
                    # Apply both validations but don't raise errors if they fail
                    content_dict = validate_and_extract_nif(content_dict)
                    content_dict = validate_and_extract_cif(content_dict)
                    result.content = json.dumps(content_dict)
            except json.JSONDecodeError:
                pass
            
    except Exception as e:
        logger.error(f"Agent processing error: {str(e)}")
        return {
            "status": "error", 
            "message": str(e),
            "filename": original_filename,
            "workflowtype": workflowtype
        }
    
    return new_state