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

import json
import os
from pathlib import Path
import pandas as pd
from langchain_core.tools import tool
from ..models import AgentState
from .textract_processor import TextractProcessor
import logging
import sqlite3
from langchain_core.messages import AIMessage

EXCEL_OUTPUT_DIR = os.getenv('EXCEL_OUTPUT_DIR', './output')
excel_filename = Path(EXCEL_OUTPUT_DIR) / "Results.xlsx"

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")

logger = logging.getLogger(__name__)

# T = TypeVar('T', bound=BaseModel)

# class APIClient(Generic[T]):
#     @retry(
#         stop=stop_after_attempt(3),
#         wait=wait_exponential(multiplier=1, min=4, max=10)
#     )
#     async def make_request(self, endpoint: str) -> T:
#         try:
#             # Your API call here
#             pass
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=str(e))

@tool
def dic2excel_tarjetas(state: AgentState):
    """Convert a json containing TIPO DOCUMENTO either 'Tarjeta de credito' or 'Extracto movimiento' into an Excel file using pandas dataframe."""
    try:
        # Get the content from the last message
        for message in reversed(state["messages"]):
            if isinstance(message.content, str):
                try:
                    content = json.loads(message.content)
                    logger.info("Found valid JSON content")
                    last_message = message
                    break
                except json.JSONDecodeError:
                    continue
        else:
            logger.error("No valid JSON content found in messages")
            return "Error: No valid JSON content found in messages"

        filename = Path(state['filename']).stem
        
        # Create DataFrame from new content
        new_df = pd.DataFrame(content)
        
        # Use current working directory
        output_path = os.path.join(os.getcwd(), Path(EXCEL_OUTPUT_DIR))
        
        # Check if file exists
        if os.path.exists(output_path):
            # Read existing Excel file
            existing_df = pd.read_excel(output_path)
            # Concatenate existing and new data
            df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            df = new_df
        
        # Ensure consistent column order
        desired_columns = [
            'DOCUMENTO',
            'TIPO DOCUMENTO',
            'FECHA VALOR',
            'ORDENANTE',
            'DESTINATARIO/BENEFICIARIO',
            'CONCEPTO',
            'IMPORTE'
        ]
        
        # Reorder columns if they exist
        df = df.reindex(columns=desired_columns)
        
        # Save with index=False to avoid extra column
        df.to_excel(output_path, index=False)
        return "Done"
    
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def dic2excel_transferencias(state: AgentState):
    """Convert a json containing TIPO DOCUMENTO either 'Orden de transferencia', 'Transferencia emitida', 'Adeudo por transferencia', 'Orden de pago', 'Detalle movimiento'or 'Certificado bancario' into an Excel file using pandas dataframe."""
    try:
        # Get the content from the last message
        for message in reversed(state["messages"]):
            if isinstance(message.content, str):
                try:
                    content = json.loads(message.content)
                    logger.info("Found valid JSON content")
                    last_message = message
                    break
                except json.JSONDecodeError:
                    continue
        else:
            logger.error("No valid JSON content found in messages")
            return "Error: No valid JSON content found in messages"

        filename = Path(state['filename']).stem
        
        # Create DataFrame from content
        new_df = pd.DataFrame(content)
        
        # Use current working directory
        output_path = os.path.join(os.getcwd(), excel_filename)
        
        # Check if file exists
        if os.path.exists(output_path):
            # Read existing Excel file
            existing_df = pd.read_excel(output_path)
            # Concatenate existing and new data
            df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            df = new_df
        
        # Ensure consistent column order
        desired_columns = [
            'DOCUMENTO',
            'TIPO DOCUMENTO',
            'FECHA VALOR',
            'ORDENANTE',
            'DESTINATARIO/BENEFICIARIO',
            'CONCEPTO',
            'IMPORTE'
        ]
        
        # Reorder columns if they exist
        df = df.reindex(columns=desired_columns)
        
        # Save with index=False to avoid extra column
        df.to_excel(output_path, index=False)
        return "Done"
    
    except Exception as e:
        return f"Error: {str(e)}"

def transferencia_post(state: AgentState) -> AgentState:
    """Post a transferencia to the database."""
    try:
        filename = Path(state['filename']).stem
        transferencia_json = state['transferencia']  # This is a JSON string
        print(f"\n\nTransferencia ==== {transferencia_json}\n\n")
        
        if not transferencia_json:
            return {
                "status": "error",
                "message": "Error: No valid content found",
                "sender": "transferencia_post",
                "filename": filename,
                "workflowtype": state.get('workflowtype')
            }
        
        # Parse the JSON string into a dictionary
        try:
            transferencia = json.loads(transferencia_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse transferencia JSON: {str(e)}")
            return {
                "status": "error",
                "message": f"Invalid JSON format - {str(e)}",
                "sender": "transferencia_post",
                "filename": filename,
                "workflowtype": state.get('workflowtype')
            }

        # Insert into database
        with sqlite3.connect(os.getenv("DB_PATH", "./database/database.db")) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO pagos
                (id_documento, tipo_documento, fecha_valor, ordenante, beneficiario, concepto, importe)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                filename,  # Use filename instead of DOCUMENTO
                transferencia["TIPO"],
                transferencia["FECHA_VALOR"],
                transferencia["ORDENANTE"],
                transferencia["BENEFICIARIO"],
                transferencia["CONCEPTO"],
                transferencia["IMPORTE"]
            ))
            conn.commit()
            
        # Return updated state
        new_state = state.copy()
        new_state.update({
            "messages": [AIMessage(content="FINAL ANSWER")],
            "sender": "transferencia_post",
            "status": "success",
            "filename": filename,
            "workflowtype": state.get('workflowtype')
        })
        return new_state
            

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {
            "status": "error",
            "message": f"Error: {str(e)}",
            "sender": "transferencia_post",
            "filename": filename,
            "workflowtype": state.get('workflowtype')
        }

def tarjeta_post(state: AgentState) -> AgentState:
    """Post a tarjeta to the database."""
    try:
        filename = Path(state['filename']).stem
        tarjeta_json = state['tarjeta']  # This is a JSON string
        print(f"\n\nTarjeta ==== {tarjeta_json}\n\n")
        
        if not tarjeta_json:
            return {
                "status": "error",
                "message": "Error: No valid content found",
                "sender": "tarjeta_post",
                "filename": filename,
                "workflowtype": state.get('workflowtype')
            }
        
        # Parse the JSON string into a dictionary
        try:
            tarjeta = json.loads(tarjeta_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tarjeta JSON: {str(e)}")
            return {
                "status": "error",
                "message": f"Invalid JSON format - {str(e)}",
                "sender": "tarjeta_post",
                "filename": filename,
                "workflowtype": state.get('workflowtype')
            }

        # Insert into database
        with sqlite3.connect(os.getenv("DB_PATH", "./database/database.db")) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO pagos
                (id_documento, tipo_documento, fecha_valor, ordenante, beneficiario, concepto, importe)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                filename,
                tarjeta["TIPO"],
                tarjeta["FECHA_VALOR"],
                tarjeta["ORDENANTE"],
                tarjeta["BENEFICIARIO"],
                tarjeta["CONCEPTO"],
                tarjeta["IMPORTE"]
            ))
            conn.commit()
            
        # Return updated state
        new_state = state.copy()
        new_state.update({
            "messages": [AIMessage(content="FINAL ANSWER")],
            "sender": "tarjeta_post",
            "status": "success",
            "filename": filename,
            "workflowtype": state.get('workflowtype')
        })
        return new_state
            

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {
            "status": "error",
            "message": f"Error: {str(e)}",
            "sender": "tarjeta_post",
            "filename": filename,
            "workflowtype": state.get('workflowtype')
        }

def factura_post(state: AgentState) -> AgentState:
    """Post a factura to the database."""
    try:
        filename = Path(state['filename']).stem
        factura_json = state['factura']  # This is a JSON string
        print(f"\n\nFactura ==== {factura_json}\n\n")
        
        if not factura_json:
            return {
                "status": "error",
                "message": "No valid content found",
                "sender": "factura_post",
                "filename": filename,
                "workflowtype": state.get('workflowtype')
            }
        
        # Parse the JSON string into a dictionary
        try:
            factura = json.loads(factura_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse factura JSON: {str(e)}")
            return {
                "status": "error",
                "message": f"Invalid JSON format - {str(e)}",
                "sender": "factura_post",
                "filename": filename,
                "workflowtype": state.get('workflowtype')
            }
            
        # Insert into database
        with sqlite3.connect(os.getenv("DB_PATH", "./database/database.db")) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO facturas
                (id_documento, cif_cliente, cliente, numero_factura, fecha_factura, 
                proveedor, base_imponible, cif_proveedor, irpf, iva, total_factura)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                filename,
                factura["CIF_CLIENTE"],
                factura["CLIENTE"],
                factura["NUMERO_FACTURA"],
                factura["FECHA_FACTURA"],
                factura["PROVEEDOR"],
                factura["BASE_IMPONIBLE"],
                factura["CIF_PROVEEDOR"],
                factura["IRPF"],
                factura["IVA"],
                factura["TOTAL_FACTURA"]
            ))
            conn.commit()
            
        # Return updated state
        new_state = state.copy()
        new_state.update({
            "messages": [AIMessage(content="FINAL ANSWER")],
            "sender": "factura_post",
            "status": "success",
            "filename": filename,
            "workflowtype": state.get('workflowtype')
        })

        return new_state
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "sender": "factura_post",
            "filename": filename,
            "workflowtype": state.get('workflowtype')
        }

def nomina_post(state: AgentState) -> AgentState:
    """Post a nomina to the database."""
    try:
        filename = state.get('filename', '')
        nomina_json = state.get('nomina', '')  # This is a JSON string
        print(f"\n\nNomina ==== {nomina_json}\n\n")
        
        if not nomina_json:
            return {
                "status": "error",
                "message": "No valid content found",
                "sender": "nomina_post",
                "filename": filename,
                "workflowtype": state.get('workflowtype')
            }
        
        # Parse the JSON string into a dictionary
        try:
            # If it's already a dict, use it directly
            if isinstance(nomina_json, dict):
                nomina = nomina_json
            else:
                nomina = json.loads(nomina_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse nomina JSON: {str(e)}")
            return {
                "status": "error",
                "message": f"Invalid JSON format - {str(e)}",
                "sender": "nomina_post",
                "filename": filename,
                "workflowtype": state.get('workflowtype')
            }
        
        # Validate required fields exist, fill with defaults if missing
        required_fields = ["MES", "FECHA_INICIO", "FECHA_FIN", "CIF", "TRABAJADOR", "NAF", "NIF", 
                         "CATEGORIA", "ANTIGUEDAD", "CONTRATO", "TOTAL_DEVENGOS", "TOTAL_DEDUCCIONES"]
        
        for field in required_fields:
            if field not in nomina or nomina[field] is None:
                nomina[field] = ""
                logger.warning(f"Missing required field in nomina: {field}, using empty string")
        
        # Insert into database
        with sqlite3.connect(os.getenv("DB_PATH", "./database/database.db")) as conn:
            cursor = conn.cursor()
            
            # Delete existing record for this document to avoid duplicates
            cursor.execute("DELETE FROM nominas WHERE id_documento = ?", (filename,))
            
            cursor.execute("""
                INSERT INTO nominas
                (id_documento, mes, fecha_inicio, fecha_fin, cif, trabajador, naf, nif, categoria, 
                antiguedad, contrato, total_devengos, total_deducciones, absentismos, bc_teorica, 
                prorrata, bc_con_complementos, total_seg_social, bonificaciones_ss_trabajador, 
                total_retenciones, total_retenciones_ss, liquido_a_percibir, a_abonar, 
                total_cuota_empresarial, comments)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                filename,
                nomina.get("MES", ""),
                nomina.get("FECHA_INICIO", ""),
                nomina.get("FECHA_FIN", ""),
                nomina.get("CIF", ""),
                nomina.get("TRABAJADOR", ""),
                nomina.get("NAF", ""),
                nomina.get("NIF", ""),
                nomina.get("CATEGORIA", ""),
                nomina.get("ANTIGUEDAD", ""),
                nomina.get("CONTRATO", ""),
                nomina.get("TOTAL_DEVENGOS", ""),
                nomina.get("TOTAL_DEDUCCIONES", ""),
                nomina.get("ABSENTISMOS", ""),
                nomina.get("BC_TEORICA", ""),
                nomina.get("PRORRATA", ""),
                nomina.get("BC_CON_COMPLEMENTOS", ""),
                nomina.get("TOTAL_SEG_SOCIAL", ""),
                nomina.get("BONIFICACIONES_SS_TRABAJADOR", ""),
                nomina.get("TOTAL_RETENCIONES", ""),
                nomina.get("TOTAL_RETENCIONES_SS", ""),
                nomina.get("LIQUIDO_A_PERCIBIR", ""),
                nomina.get("A_ABONAR", ""),
                nomina.get("TOTAL_CUOTA_EMPRESARIAL", ""),
                nomina.get('COMMENTS', '')
            ))
            conn.commit()
            
            # Verify data was inserted
            cursor.execute("SELECT COUNT(*) FROM nominas WHERE id_documento = ?", (filename,))
            count = cursor.fetchone()[0]
            logger.info(f"Inserted {count} nomina record for document {filename}")
            
        new_state = state.copy()
        new_state.update({
            "messages": [AIMessage(content="FINAL ANSWER")],
            "sender": "nomina_post",
            "status": "success",
            "filename": filename,
            "workflowtype": state.get('workflowtype')
        })
        return new_state

    except Exception as e:
        logger.error(f"Unexpected error in nomina_post: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "sender": "nomina_post",
            "filename": filename if 'filename' in locals() else "",
            "workflowtype": state.get('workflowtype')
        }

def generate_textract(state: AgentState) -> AgentState:
    """
    Process the current document with AWS Textract to extract table information.
    Returns the extracted tables in a structured format.
    """
    try:
        # Get the filename from state
        filename = Path(state['filename'])
        if not filename:
            return {
                "status": "error",
                "message": "Error: No filename found in state",
                "sender": "generate_textract",
                "filename": filename,
                "workflowtype": state.get('workflowtype')
            }

        # Construct file path
        IMAGES_DIR = Path(os.getenv("IMAGE_INPUT_DIR", "./Images"))
        images_dir = IMAGES_DIR / state.get('workflowtype')
        file_path = images_dir / filename

        if not file_path.exists():
            return {
                "status": "error",
                "message": f"Error: File {filename} not found at {file_path}",
                "sender": "generate_textract",
                "filename": filename,
                "workflowtype": state.get('workflowtype')
            }

        # Read file bytes
        with open(file_path, 'rb') as f:
            file_bytes = f.read()

        # Initialize TextractProcessor
        try:
            textract_processor = TextractProcessor()
        except ValueError as e:
            logger.error(f"Failed to initialize Textract: {str(e)}")
            return {
                "status": "error",
                "message": f"Error: {str(e)}",
                "sender": "generate_textract",
                "filename": filename,
                "workflowtype": state.get('workflowtype')
            }

        # Process with Textract
        import asyncio
        extracted_data = asyncio.run(
            textract_processor.process_document_tables(file_bytes)
        )

        if not extracted_data.get('tables'):
            return {
                "status": "error",
                "message": "No tables found",
                "sender": "generate_textract",
                "filename": filename,
                "workflowtype": state.get('workflowtype')
            }

        # Format the response for the agent
        response = {
            "tables": extracted_data['tables']
        }

        # Create new state with the extracted data
        new_state = state.copy()
        new_state.update({
            "messages": [AIMessage(content=json.dumps(response))],
            "sender": "generate_textract",
            "status": "success",
            "filename": filename,
            "workflowtype": state.get('workflowtype')
        })
        

        return new_state

    except Exception as e:
        logger.error(f"Error in generate_textract: {str(e)}")
        return {
            "status": "error",
            "message": f"Error processing document with Textract: {str(e)}",
            "sender": "generate_textract",
            "filename": filename,
            "workflowtype": state.get('workflowtype')
        }
    
def pago_table_post(state: AgentState) -> AgentState:
    """Post a table of pagos to the database."""
    try:
        filename = Path(state['filename']).stem
        tabla_json = state['tablatarjetas']  # This is a JSON string
        print(f"\n\nTabla Tarjetas ==== {tabla_json}\n\n")
        
        if not tabla_json:
            return {
                "status": "error",
                "message": "No valid content found",
                "sender": "table_upload",
                "filename": filename,
                "workflowtype": state.get('workflowtype')
            }
        
        # Parse the JSON string into a dictionary
        try:
            tabla = json.loads(tabla_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tabla JSON: {str(e)}")
            return {
                "status": "error",
                "message": f"Invalid JSON format - {str(e)}",
                "sender": "table_upload",
                "filename": filename,
                "workflowtype": state.get('workflowtype')
            }
            
        # Insert each row into database
        with sqlite3.connect(os.getenv("DB_PATH", "./database/database.db")) as conn:
            cursor = conn.cursor()
            
            # Iterate through all items in the dictionary
            for row_key, row_data in tabla.items():
                if not row_key.startswith('fila'):  # Skip any non-row entries
                    continue
                    
                cursor.execute("""
                    INSERT INTO pagos_tabla
                    (id_documento, concepto, fecha_valor, importe)
                    VALUES (?, ?, ?, ?)
                """, (
                    filename,
                    row_data["CONCEPTO"],
                    row_data["FECHA_VALOR"],
                    row_data["IMPORTE"]
                ))
            conn.commit()
            
        # Return updated state
        new_state = state.copy()
        new_state.update({
            "messages": [AIMessage(content="FINAL ANSWER")],
            "sender": "table_upload",
            "status": "success",
            "filename": filename,
            "workflowtype": state.get('workflowtype')
        })
        return new_state
            

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "sender": "table_upload",
            "filename": filename,
            "workflowtype": state.get('workflowtype')
        }

def factura_table_post(state: AgentState) -> AgentState:
    """Post a table of facturas to the database."""
    try:
        filename = Path(state['filename']).stem
        tabla_json = state['tablafacturas']  # This is a JSON string
        print(f"\n\nTabla Facturas ==== {tabla_json}\n\n")
        
        if not tabla_json:
            return {
                "status": "error",
                "message": "No valid content found",
                "sender": "table_upload",
                "filename": filename,
                "workflowtype": state.get('workflowtype')
            }
        
        # Parse the JSON string into a dictionary
        try:
            tabla = json.loads(tabla_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tabla JSON: {str(e)}")
            return {
                "status": "error",
                "message": f"Invalid JSON format - {str(e)}",
                "sender": "table_upload",
                "filename": filename,
                "workflowtype": state.get('workflowtype')
            }
            
        # Insert each row into database
        with sqlite3.connect(os.getenv("DB_PATH", "./database/database.db")) as conn:
            cursor = conn.cursor()
            
            # Iterate through all items in the dictionary
            for row_key, row_data in tabla.items():
                if not row_key.startswith('fila'):  # Skip any non-row entries
                    continue
                    
                cursor.execute("""
                    INSERT INTO facturas_tabla
                    (id_documento, concepto, unidades, importe)
                    VALUES (?, ?, ?, ?)
                """, (
                    filename,
                    row_data["CONCEPTO"],
                    row_data["UNIDADES"] if row_data["UNIDADES"] is not None else "",  # Handle null values
                    row_data["IMPORTE"]
                ))
            conn.commit()
            
        # Return updated state
        new_state = state.copy()
        new_state.update({
            "messages": [AIMessage(content="FINAL ANSWER")],
            "sender": "table_upload",
            "status": "success",
            "filename": filename,
            "workflowtype": state.get('workflowtype')
        })
        return new_state
            

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "sender": "table_upload",
            "filename": filename,
            "workflowtype": state.get('workflowtype')
        }

def nomina_table_post(state: AgentState) -> AgentState:
    """Post a table of nominas to the database."""
    try:
        # Get the full filename from state
        filename = Path(state['filename']).stem
        tabla_json = state['tablanominas']
        print(f"\n\nTabla Nomina ==== {tabla_json}\n\n")
        
        if not tabla_json:
            return {
                "status": "error",
                "message": "No valid content found",
                "sender": "table_upload",
                "filename": filename,
                "workflowtype": state.get('workflowtype')
            }
        
        # Parse the JSON string into a dictionary
        try:
            tabla = json.loads(tabla_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tabla JSON: {str(e)}")
            return {
                "status": "error",
                "message": f"Invalid JSON format - {str(e)}",
                "sender": "table_upload",
                "filename": filename,
                "workflowtype": state.get('workflowtype')
            }
            
        # Insert each row into database
        with sqlite3.connect(os.getenv("DB_PATH", "./database/database.db")) as conn:
            cursor = conn.cursor()
            
            for row_key, row_data in tabla.items():
                if not row_key.startswith('fila'):
                    continue
                    
                cursor.execute("""
                    INSERT INTO nominas_tabla
                    (id_documento, descripcion, devengos, deducciones)
                    VALUES (?, ?, ?, ?)
                """, (
                    filename,
                    row_data["DESCRIPCION"],
                    row_data["DEVENGOS"] if row_data["DEVENGOS"] is not None else "",
                    row_data["DEDUCCIONES"] if row_data["DEDUCCIONES"] is not None else ""
                ))
            conn.commit()
            
        new_state = state.copy()
        new_state.update({
            "messages": [AIMessage(content="FINAL ANSWER")],
            "sender": "table_upload",
            "status": "success",
            "filename": filename,
            "workflowtype": state.get('workflowtype')
        })
        return new_state

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "sender": "table_upload",
            "filename": filename,
            "workflowtype": state.get('workflowtype')
        }