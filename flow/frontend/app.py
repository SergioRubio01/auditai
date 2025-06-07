import streamlit as st
import requests
import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv
import json
import time
from datetime import datetime
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import asyncio
from functools import partial
from io import BytesIO
import sqlite3

# Load environment variables
load_dotenv()

# Set Streamlit configuration
st.set_page_config(
    page_title="AutoAudit Pro",
    page_icon="🤖",
    layout="wide"
)

# Increase file size limit to 1GB (1000MB)
st.config.set_option('server.maxUploadSize', 1000)
st.config.set_option('server.maxMessageSize', 1000)

# Constants
API_URL = os.getenv("SERVER_URL", "http://localhost:8000")
ALLOWED_EXTENSIONS = ['.png', '.jpg', '.jpeg']
LOCAL_DB_PATH = os.path.join('/app/database', 'pdf_history.db')

def init_local_db():
    """Initialize local SQLite database for PDF conversion history"""
    try:
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Create PDF conversion history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pdf_conversion_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    filename TEXT NOT NULL,
                    pages INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    file_size TEXT NOT NULL,
                    document_type TEXT NOT NULL
                )
            """)
            
            conn.commit()
            
    except Exception as e:
        st.error(f"Error initializing local database: {str(e)}")

def save_history_to_local_db(history_data):
    """Save PDF conversion history to local database"""
    try:
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Extract required fields
            filename = history_data.get("filename", "")
            pages = history_data.get("pages", 0)
            status = history_data.get("status", "Unknown")
            file_size = history_data.get("file_size", "0 KB")
            document_type = history_data.get("document_type", "General")
            
            # Insert into database
            cursor.execute("""
                INSERT INTO pdf_conversion_history 
                (filename, pages, status, file_size, document_type)
                VALUES (?, ?, ?, ?, ?)
            """, (filename, pages, status, file_size, document_type))
            
            conn.commit()
            
        return True
        
    except Exception as e:
        st.error(f"Error saving history to local database: {str(e)}")
        return False

def get_history_from_local_db():
    """Get PDF conversion history from local database"""
    try:
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            # Configure connection to return rows as dictionaries
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get history records
            cursor.execute("""
                SELECT * FROM pdf_conversion_history 
                ORDER BY timestamp DESC
            """)
            
            # Convert to list of dictionaries
            history = [dict(row) for row in cursor.fetchall()]
            
        return history
        
    except Exception as e:
        st.error(f"Error getting history from local database: {str(e)}")
        return []

def clear_history_from_local_db():
    """Clear all PDF conversion history from local database"""
    try:
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Delete all history records
            cursor.execute("DELETE FROM pdf_conversion_history")
            
            conn.commit()
            
        return True
        
    except Exception as e:
        st.error(f"Error clearing history from local database: {str(e)}")
        return False

def delete_history_record_from_local_db(record_id):
    """Delete a single PDF conversion history record from local database"""
    try:
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Delete the record
            cursor.execute("DELETE FROM pdf_conversion_history WHERE id = ?", (record_id,))
            
            conn.commit()
            
        return True
        
    except Exception as e:
        st.error(f"Error deleting history record from local database: {str(e)}")
        return False

def init_session_state():
    """Initialize session state variables"""
    if 'selected_files' not in st.session_state:
        st.session_state.selected_files = []
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = {}
    if 'show_processed' not in st.session_state:
        st.session_state.show_processed = False
    if 'processing_history' not in st.session_state:
        st.session_state.processing_history = []
    if 'error_log' not in st.session_state:
        st.session_state.error_log = []
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    if 'pdf_conversion_history' not in st.session_state:
        # Initialize the local database
        init_local_db()
        # Try to get PDF conversion history from API
        try:
            response = requests.get(f"{API_URL}/pdf/history")
            if response.status_code == 200:
                history_data = response.json()
                # Convert timestamp strings to datetime objects
                for entry in history_data:
                    if isinstance(entry.get("timestamp"), str):
                        try:
                            entry["timestamp"] = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
                        except (ValueError, TypeError):
                            # Keep as string if conversion fails
                            pass
                st.session_state.pdf_conversion_history = history_data
            else:
                # Fallback to local database
                history_data = get_history_from_local_db()
                for entry in history_data:
                    if isinstance(entry.get("timestamp"), str):
                        try:
                            entry["timestamp"] = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
                        except (ValueError, TypeError):
                            pass
                st.session_state.pdf_conversion_history = history_data
        except Exception as e:
            # Fallback to local database
            history_data = get_history_from_local_db()
            for entry in history_data:
                if isinstance(entry.get("timestamp"), str):
                    try:
                        entry["timestamp"] = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
                    except (ValueError, TypeError):
                        pass
            st.session_state.pdf_conversion_history = history_data
    if 'show_delete_confirmation' not in st.session_state:
        st.session_state.show_delete_confirmation = False

def is_valid_image(file):
    """Check if file has valid image extension"""
    return Path(file.name).suffix.lower() in ALLOWED_EXTENSIONS

def upload_files(files):
    """Upload multiple files to the server"""
    results = []
    for file in files:
        if is_valid_image(file):
            try:
                response = requests.post(
                    f"{API_URL}/images/upload",
                    files={"file": file}
                )
                if response.status_code == 200:
                    results.append({"filename": file.name, "status": "Success"})
                else:
                    results.append({"filename": file.name, "status": f"Error: {response.text}"})
            except Exception as e:
                results.append({"filename": file.name, "status": f"Error: {str(e)}"})
        else:
            results.append({"filename": file.name, "status": "Error: Invalid file type"})
    return results

def process_file_with_status(filename: str, doc_type: str):
    """Process a single file and return its status"""
    try:
        response = requests.post(
            f"{API_URL}/images/{filename}/process/{doc_type}",
            json={"doc_type": doc_type}
        )
        if response.status_code == 200:
            result = response.json()
            return {
                "filename": filename,
                "status": "Success",
                "message": result.get("message", "Processed successfully"),
                "type": doc_type
            }
        else:
            return {
                "filename": filename,
                "status": "Error",
                "message": f"Error: {response.text}",
                "type": doc_type
            }
    except Exception as e:
        return {
            "filename": filename,
            "status": "Error",
            "message": f"Error: {str(e)}",
            "type": doc_type
        }

def display_upload_tab():
    """Display the upload interface with separate sections"""
    st.header("📤 Upload Documents")
    
    # Add delete options in expander
    with st.expander("🗑️ Delete Options"):
        delete_transactions = st.checkbox(
            "Include Transactions", 
            help="Also delete all stored transactions from the database"
        )
        
        delete_pdf_history = st.checkbox(
            "Include PDF Conversion History",
            help="Also delete all PDF conversion history records"
        )
        
        # First button to trigger confirmation
        if st.button("🗑️ Delete All", type="secondary", key="delete_trigger"):
            # Store deletion intent in session state
            st.session_state.show_delete_confirmation = True
        
        # Show confirmation only if triggered
        if st.session_state.get('show_delete_confirmation', False):
            st.warning("⚠️ This action cannot be undone!")
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if st.button("⚠️ Confirm", type="primary", key="confirm_delete"):
                    with st.spinner("Deleting data..."):
                        try:
                            # Make the API call
                            response = requests.delete(
                                f"{API_URL}/images/delete",  # Updated endpoint
                                params={
                                    "delete_transactions": delete_transactions,
                                    "delete_pdf_history": delete_pdf_history
                                }
                            )
                            if response.status_code == 200:
                                result = response.json()
                                st.success("Successfully deleted all data")
                                
                                # Clear session states
                                st.session_state.processed_files = []
                                st.session_state.show_delete_confirmation = False
                                
                                # Clear PDF conversion history if requested
                                if delete_pdf_history:
                                    st.session_state.pdf_conversion_history = []
                                    
                                # Force page refresh
                                st.rerun()
                            else:
                                st.error(f"Error deleting data: {response.text}")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            with col2:
                if st.button("❌ Cancel", type="secondary", key="cancel_delete"):
                    st.session_state.show_delete_confirmation = False
                    st.rerun()
    
    st.divider()
    
    # Create three columns for Facturas, Pagos, and Nominas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📑 Facturas")
        uploaded_facturas = st.file_uploader(
            "Upload invoice documents",
            accept_multiple_files=True,
            type=['zip', 'png', 'jpg', 'jpeg'],
            key="facturas_uploader"
        )
        
        if uploaded_facturas:
            if st.button("Upload Facturas", key="upload_facturas_btn"):
                with st.spinner("Uploading and extracting facturas..."):
                    results = []
                    for file in uploaded_facturas:
                        try:
                            response = requests.post(
                                f"{API_URL}/images/upload/facturas",
                                files={"file": file}
                            )
                            if response.status_code == 200:
                                processed_files = response.json()
                                for processed in processed_files:
                                    results.append({
                                        "filename": processed["filename"],
                                        "status": "Success",
                                        "size": f"{processed['size']/1024:.1f} KB",
                                        "type": "Factura"
                                    })
                            else:
                                results.append({
                                    "filename": file.name,
                                    "status": f"Error: {response.text}",
                                    "size": f"{len(file.getvalue())/1024:.1f} KB",
                                    "type": "Factura"
                                })
                        except Exception as e:
                            results.append({
                                "filename": file.name,
                                "status": f"Error: {str(e)}",
                                "size": f"{len(file.getvalue())/1024:.1f} KB",
                                "type": "Factura"
                            })
                    display_upload_results(results)
    
    with col2:
        st.subheader("💳 Pagos")
        uploaded_pagos = st.file_uploader(
            "Upload payment documents",
            accept_multiple_files=True,
            type=['zip', 'png', 'jpg', 'jpeg'],
            key="pagos_uploader"
        )
        
        if uploaded_pagos:
            if st.button("Upload Pagos", key="upload_pagos_btn"):
                with st.spinner("Uploading and extracting pagos..."):
                    results = []
                    for file in uploaded_pagos:
                        try:
                            response = requests.post(
                                f"{API_URL}/images/upload/pagos",
                                files={"file": file}
                            )
                            if response.status_code == 200:
                                processed_files = response.json()
                                for processed in processed_files:
                                    results.append({
                                        "filename": processed["filename"],
                                        "status": "Success",
                                        "size": f"{processed['size']/1024:.1f} KB",
                                        "type": "Pago"
                                    })
                            else:
                                results.append({
                                    "filename": file.name,
                                    "status": f"Error: {response.text}",
                                    "size": f"{len(file.getvalue())/1024:.1f} KB",
                                    "type": "Pago"
                                })
                        except Exception as e:
                            results.append({
                                "filename": file.name,
                                "status": f"Error: {str(e)}",
                                "size": f"{len(file.getvalue())/1024:.1f} KB",
                                "type": "Pago"
                            })
                    display_upload_results(results)
    
    with col3:
        st.subheader("👥 Nóminas")
        uploaded_nominas = st.file_uploader(
            "Upload payroll documents",
            accept_multiple_files=True,
            type=['zip', 'png', 'jpg', 'jpeg'],
            key="nominas_uploader"
        )
        
        if uploaded_nominas:
            if st.button("Upload Nóminas", key="upload_nominas_btn"):
                with st.spinner("Uploading and extracting nóminas..."):
                    results = []
                    for file in uploaded_nominas:
                        try:
                            response = requests.post(
                                f"{API_URL}/images/upload/nominas",
                                files={"file": file}
                            )
                            if response.status_code == 200:
                                processed_files = response.json()
                                for processed in processed_files:
                                    results.append({
                                        "filename": processed["filename"],
                                        "status": "Success",
                                        "size": f"{processed['size']/1024:.1f} KB",
                                        "type": "Nómina"
                                    })
                            else:
                                results.append({
                                    "filename": file.name,
                                    "status": f"Error: {response.text}",
                                    "size": f"{len(file.getvalue())/1024:.1f} KB",
                                    "type": "Nómina"
                                })
                        except Exception as e:
                            results.append({
                                "filename": file.name,
                                "status": f"Error: {str(e)}",
                                "size": f"{len(file.getvalue())/1024:.1f} KB",
                                "type": "Nómina"
                            })
                    display_upload_results(results)

def display_upload_results(results):
    """Display upload results in a formatted table"""
    if results:
        result_df = pd.DataFrame(results)
        st.dataframe(
            result_df,
            column_config={
                "filename": "Filename",
                "status": "Status",
                "size": "Size",
                "type": "Document Type"
            },
            hide_index=True
        )
        
        # Force refresh to update the process tab
        time.sleep(2)
        st.rerun()

def get_unprocessed_images():
    """[DEPRECATED] Get list of unprocessed images"""
    try:
        response = requests.get(f"{API_URL}/images/", params={"processed": False})
        return response.json()
    except Exception as e:
        st.error(f"Error fetching unprocessed images: {str(e)}")
        return []

def get_processed_images():
    """Get list of processed images for both pagos and facturas"""
    try:
        # Get pagos images
        pagos_response = requests.get(f"{API_URL}/images/pagos", params={"processed": True})
        pagos_images = pagos_response.json() if pagos_response.status_code == 200 else []
        
        # Get facturas images 
        facturas_response = requests.get(f"{API_URL}/images/facturas", params={"processed": True})
        facturas_images = facturas_response.json() if facturas_response.status_code == 200 else []
        
        # Process and combine both lists
        processed_images = []
        
        for img in pagos_images:
            if isinstance(img, dict):
                processed_images.append({
                    "filename": img.get("filename", ""),
                    "status": img.get("status", "Unknown"),
                    "message": img.get("message", ""),
                    "type": "Pago"
                })
            
        for img in facturas_images:
            if isinstance(img, dict):
                processed_images.append({
                    "filename": img.get("filename", ""),
                    "status": img.get("status", "Unknown"),
                    "message": img.get("message", ""),
                    "type": "Factura"
                })
        
        return processed_images
        
    except Exception as e:
        st.error(f"Error fetching processed images: {str(e)}")
        return []

def display_process_tab():
    """Display the processing interface with process and results tabs"""
    # Create tabs for processing and results
    tab1, tab2 = st.tabs(["⚙️ Process", "📊 Results"])
    
    with tab1:
        st.subheader("Process Documents")
        
        # Add batch processing configuration
        with st.expander("🔧 Processing Configuration"):
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=100,
                value=10,
                help="Number of documents to process in parallel"
            )
            
            rewrite = st.checkbox(
                "Reescribir valores",
                help="Reescribe los valores obtenidos previamente por el modelo si se encuentra marcado"
            )
        
        try:
            # Create three columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 📑 Process Facturas")
                process_document_type("facturas", batch_size)
                
            with col2:
                st.markdown("### 💳 Process Pagos")
                process_document_type("pagos", batch_size)
                
            with col3:
                st.markdown("### 👥 Process Nóminas")
                process_document_type("nominas", batch_size)
                
        except Exception as e:
            st.error(f"Error in process tab: {str(e)}")
    
    with tab2:
        st.subheader("Processing Results")
        
        try:
            # Get data from both endpoints
            facturas_response = requests.get(f"{API_URL}/facturas_tabla/")
            pagos_response = requests.get(f"{API_URL}/pagos_tabla/")
            nominas_response = requests.get(f"{API_URL}/nominas_tabla/")
            
            facturas = facturas_response.json() if facturas_response.status_code == 200 else []
            pagos = pagos_response.json() if pagos_response.status_code == 200 else []
            nominas = nominas_response.json() if nominas_response.status_code == 200 else []
            
            # Create tabs for different document types
            subtab1, subtab2, subtab3 = st.tabs(["📑 Facturas", "💳 Pagos", "👥 Nóminas"])
            
            with subtab1:
                if facturas:
                    df_facturas = pd.DataFrame(facturas)
                    st.dataframe(
                        df_facturas,
                        hide_index=True,
                        column_config={
                            "id_documento": "Documento",
                            "cif_cliente": "CIF Cliente",
                            "cliente": "Cliente",
                            "numero_factura": "Número Factura",
                            "fecha_factura": "Fecha Factura",
                            "proveedor": "Proveedor",
                            "base_imponible": st.column_config.NumberColumn("Base Imponible", format="€%.2f"),
                            "cif_proveedor": "CIF Proveedor",
                            "irpf": st.column_config.NumberColumn("IRPF", format="€%.2f"),
                            "iva": st.column_config.NumberColumn("IVA", format="€%.2f"),
                            "total_factura": st.column_config.NumberColumn("Total Factura", format="€%.2f")
                        }
                    )
                else:
                    st.info("No facturas processed")
                    
            with subtab2:
                if pagos:
                    df_pagos = pd.DataFrame(pagos)
                    st.dataframe(
                        df_pagos,
                        hide_index=True,
                        column_config={
                            "DOCUMENTO": "Documento",
                            "TIPO": "Tipo",
                            "FECHA_VALOR": "Fecha Valor",
                            "ORDENANTE": "Ordenante",
                            "BENEFICIARIO": "Beneficiario",
                            "CONCEPTO": "Concepto",
                            "IMPORTE": st.column_config.NumberColumn("Importe", format="€%.2f")
                        }
                    )
                else:
                    st.info("No pagos processed")
                    
            with subtab3:
                if nominas:
                    df_nominas = pd.DataFrame(nominas)
                    st.dataframe(
                        df_nominas,
                        hide_index=True,
                        column_config={
                            "id_documento": "Documento",
                            "mes": "Mes",
                            "fecha_inicio": "Fecha Inicio",
                            "fecha_fin": "Fecha Fin",
                            "cif": "CIF",
                            "trabajador": "Trabajador",
                            "naf": "NAF",
                            "nif": "NIF",
                            "categoria": "Categoría",
                            "antiguedad": "Antigüedad",
                            "contrato": "Contrato",
                            "total_devengos": st.column_config.NumberColumn("Total Devengos", format="€%.2f"),
                            "total_deducciones": st.column_config.NumberColumn("Total Deducciones", format="€%.2f"),
                            "absentismos": "Absentismos",
                            "bc_teorica": st.column_config.NumberColumn("BC Teórica", format="€%.2f"),
                            "prorrata": "Prorrata",
                            "bc_con_complementos": st.column_config.NumberColumn("BC con Complementos", format="€%.2f"),
                            "total_seg_social": st.column_config.NumberColumn("Total Seg. Social", format="€%.2f"),
                            "bonificaciones_ss_trabajador": st.column_config.NumberColumn("Bonificaciones SS", format="€%.2f"),
                            "total_retenciones": st.column_config.NumberColumn("Total Retenciones", format="€%.2f"),
                            "total_retenciones_ss": st.column_config.NumberColumn("Total Retenciones SS", format="€%.2f"),
                            "liquido_a_percibir": st.column_config.NumberColumn("Líquido a Percibir", format="€%.2f"),
                            "a_abonar": st.column_config.NumberColumn("A Abonar", format="€%.2f"),
                            "total_cuota_empresarial": st.column_config.NumberColumn("Total Cuota Empresarial", format="€%.2f")
                        }
                    )
                else:
                    st.info("No nóminas processed")
            
            # Add export functionality
            if facturas or pagos or nominas:
                if st.button("Export All Results", key="export_all_btn", type="primary"):
                    with st.spinner("Generating Excel file..."):
                        try:
                            response = requests.get(f"{API_URL}/excel/download")
                            if response.status_code == 200:
                                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                filename = f"autoaudit_export_{timestamp}.xlsx"
                                
                                st.success("Excel file generated successfully!")
                                st.download_button(
                                    label="📥 Download Excel",
                                    data=response.content,
                                    file_name=filename,
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="download_excel_btn"
                                )
                            else:
                                st.error(f"Failed to generate Excel file: {response.text}")
                        except Exception as e:
                            st.error(f"Error downloading Excel: {str(e)}")
                
        except Exception as e:
            st.error(f"Error displaying results: {str(e)}")

def process_document_type(doc_type: str, batch_size: int):
    """Process either facturas or pagos in batches"""
    try:
        # Get unprocessed files for specific document type
        response = requests.get(f"{API_URL}/images/{doc_type}", params={"processed": False})
        if response.status_code == 200:
            unprocessed_files = response.json()
            
            if not unprocessed_files:
                st.info(f"No {doc_type} pending processing")
                return
            
            # Show total number of files
            st.write(f"Found {len(unprocessed_files)} {doc_type} to process")
            
            # Add a checkbox to show delete buttons
            show_delete_buttons = st.checkbox(f"Show Delete Buttons for {doc_type}")
            
            if show_delete_buttons:
                # Create columns for delete buttons
                delete_cols = st.columns(3)
                
                for i, file in enumerate(unprocessed_files):
                    col_idx = i % 3
                    with delete_cols[col_idx]:
                        filename = file.get("filename", "")
                        # Truncate if filename is too long
                        display_name = filename
                        if len(display_name) > 20:
                            display_name = filename[:17] + "..."
                            
                        if st.button(f"🗑️ Delete {display_name}", key=f"delete_{doc_type}_{i}"):
                            try:
                                # Call API to delete the image
                                delete_response = requests.delete(f"{API_URL}/images/{doc_type}/{filename}")
                                
                                if delete_response.status_code == 200:
                                    st.success(f"Image {filename} deleted successfully!")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error(f"Failed to delete image: {delete_response.text}")
                            except Exception as e:
                                st.error(f"Error deleting image: {str(e)}")
                
                st.divider()
            
            # File selection and processing UI
            selected_files = st.multiselect(
                f"Select {doc_type} to process:",
                options=[f["filename"] for f in unprocessed_files],
                default=[f["filename"] for f in unprocessed_files],
                key=f"process_{doc_type}"
            )
            
            if selected_files:
                if st.button(f"Process Selected {doc_type.title()}", type="primary", key=f"process_{doc_type}_btn"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    total_files = len(selected_files)
                    
                    # Process files in batches
                    with ThreadPoolExecutor(max_workers=batch_size) as executor:
                        futures = []
                        for filename in selected_files:
                            future = executor.submit(process_file_with_status, filename, doc_type)
                            futures.append(future)
                        
                        # Update progress as files complete
                        for i, future in enumerate(concurrent.futures.as_completed(futures)):
                            result = future.result()
                            results.append(result)
                            
                            # Update progress
                            progress = (i + 1) / total_files
                            progress_bar.progress(progress)
                            status_text.text(f"Processed {i + 1} of {total_files} files")
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    display_processing_results(results)
                    
                    # Force refresh after processing
                    time.sleep(2)
                    st.rerun()
                        
        else:
            st.error(f"Failed to fetch unprocessed {doc_type}")
            
    except Exception as e:
        st.error(f"Error processing {doc_type}: {str(e)}")

def display_processing_results(results):
    """Display processing results in a formatted table"""
    if not results:
        return
        
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Apply color coding based on status
    def color_status(val):
        colors = {
            'success': 'background-color: #90EE90',  # Light green
            'warning': 'background-color: #FFE5B4',  # Light orange
            'error': 'background-color: #FFB6C1'     # Light red
        }
        return colors.get(val, '')
    
    # Display styled dataframe
    st.dataframe(
        df.style.apply(lambda x: [color_status(v) for v in x], subset=['status']),
        column_config={
            "filename": "Filename",
            "status": "Status",
            "message": "Message"
        },
        hide_index=True
    )

def display_results_tab():
    """Display results from database"""
    try:
        # Get data from both endpoints
        facturas_response = requests.get(f"{API_URL}/facturas_tabla/")
        pagos_response = requests.get(f"{API_URL}/pagos_tabla/")
        nominas_response = requests.get(f"{API_URL}/nominas_tabla/")
        
        facturas = facturas_response.json() if facturas_response.status_code == 200 else []
        pagos = pagos_response.json() if pagos_response.status_code == 200 else []
        nominas = nominas_response.json() if nominas_response.status_code == 200 else []
        
        # Create tabs for different document types
        tab1, tab2, tab3 = st.tabs(["📑 Facturas", "💳 Pagos", "👥 Nóminas"])
        
        with tab1:
            if facturas:
                df_facturas = pd.DataFrame(facturas)
                st.dataframe(
                    df_facturas,
                    hide_index=True,
                    column_config={
                        "id_documento": "Documento",
                        "cif_cliente": "CIF Cliente",
                        "cliente": "Cliente",
                        "numero_factura": "Número Factura",
                        "fecha_factura": "Fecha Factura",
                        "proveedor": "Proveedor",
                        "base_imponible": st.column_config.NumberColumn("Base Imponible", format="€%.2f"),
                        "cif_proveedor": "CIF Proveedor",
                        "irpf": st.column_config.NumberColumn("IRPF", format="€%.2f"),
                        "iva": st.column_config.NumberColumn("IVA", format="€%.2f"),
                        "total_factura": st.column_config.NumberColumn("Total Factura", format="€%.2f")
                    }
                )
            else:
                st.info("No facturas processed")
                
        with tab2:
            if pagos:
                df_pagos = pd.DataFrame(pagos)
                st.dataframe(
                    df_pagos,
                    hide_index=True,
                    column_config={
                        "DOCUMENTO": "Documento",
                        "TIPO": "Tipo",
                        "FECHA_VALOR": "Fecha Valor",
                        "ORDENANTE": "Ordenante",
                        "BENEFICIARIO": "Beneficiario",
                        "CONCEPTO": "Concepto",
                        "IMPORTE": st.column_config.NumberColumn("Importe", format="€%.2f")
                    }
                )
            else:
                st.info("No pagos processed")
                
        with tab3:
            if nominas:
                df_nominas = pd.DataFrame(nominas)
                st.dataframe(
                    df_nominas,
                    hide_index=True,
                    column_config={
                        "id_documento": "Documento",
                        "mes": "Mes",
                        "fecha_inicio": "Fecha Inicio",
                        "fecha_fin": "Fecha Fin",
                        "cif": "CIF",
                        "trabajador": "Trabajador",
                        "naf": "NAF",
                        "nif": "NIF",
                        "categoria": "Categoría",
                        "antiguedad": "Antigüedad",
                        "contrato": "Contrato",
                        "total_devengos": st.column_config.NumberColumn("Total Devengos", format="€%.2f"),
                        "total_deducciones": st.column_config.NumberColumn("Total Deducciones", format="€%.2f"),
                        "absentismos": "Absentismos",
                        "bc_teorica": st.column_config.NumberColumn("BC Teórica", format="€%.2f"),
                        "prorrata": "Prorrata",
                        "bc_con_complementos": st.column_config.NumberColumn("BC con Complementos", format="€%.2f"),
                        "total_seg_social": st.column_config.NumberColumn("Total Seg. Social", format="€%.2f"),
                        "bonificaciones_ss_trabajador": st.column_config.NumberColumn("Bonificaciones SS", format="€%.2f"),
                        "total_retenciones": st.column_config.NumberColumn("Total Retenciones", format="€%.2f"),
                        "total_retenciones_ss": st.column_config.NumberColumn("Total Retenciones SS", format="€%.2f"),
                        "liquido_a_percibir": st.column_config.NumberColumn("Líquido a Percibir", format="€%.2f"),
                        "a_abonar": st.column_config.NumberColumn("A Abonar", format="€%.2f"),
                        "total_cuota_empresarial": st.column_config.NumberColumn("Total Cuota Empresarial", format="€%.2f")
                    }
                )
            else:
                st.info("No nóminas processed")
        
        # Add export functionality
        if facturas or pagos or nominas:
            if st.button("Export All Results", key="export_all_btn", type="primary"):
                with st.spinner("Generating Excel file..."):
                    try:
                        response = requests.get(f"{API_URL}/excel/download")
                        if response.status_code == 200:
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            filename = f"autoaudit_export_{timestamp}.xlsx"
                            
                            st.success("Excel file generated successfully!")
                            st.download_button(
                                label="📥 Download Excel",
                                data=response.content,
                                file_name=filename,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="download_excel_btn"
                            )
                        else:
                            st.error(f"Failed to generate Excel file: {response.text}")
                    except Exception as e:
                        st.error(f"Error downloading Excel: {str(e)}")
            
    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")

def display_workflow_tab():
    """Display the workflow visualization"""
    st.header("🔄 Workflow Visualization")
    
    try:
        response = requests.get(f"{API_URL}/workflow/graph")
        if response.status_code == 200:
            data = response.json()
            
            if data["success"]:
                # Display the graph
                st.markdown("### Agent Workflow Diagram")
                
                # Get the image file
                image_path = data["file_path"]
                if image_path and Path(image_path).exists():
                    # Display the image
                    st.image(image_path, caption="Agent Workflow Diagram")
                    
                    # Add download button
                    with open(image_path, "rb") as f:
                        image_bytes = f.read()
                        st.download_button(
                            label="📥 Download Workflow Diagram",
                            data=image_bytes,
                            file_name="agent_workflow.png",
                            mime="image/png",
                            help="Download the workflow diagram as PNG"
                        )
                        
                    # Add explanation
                    with st.expander("ℹ️ About the Workflow"):
                        st.markdown("""
                        This diagram shows how different agents work together to process your documents:
                        
                        1. **Document Type Agent**: Identifies the type of document
                        2. **Retrieval Agents**: Extract relevant information based on document type
                        3. **Supervisor Agent**: Coordinates the workflow and validates results
                        4. **Tools**: Handle specific tasks like database storage and Excel export
                        
                        The arrows show how information flows between agents.
                        """)
                else:
                    st.error("Graph image file not found")
            else:
                st.error(f"Failed to generate graph: {data['message']}")
        else:
            st.error(f"Error fetching workflow graph: {response.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Try refreshing the page if the graph doesn't appear")

def download_converted_images(doc_type: str, results: list):
    """Add download buttons for converted images"""
    if not results:
        return
        
    st.write("### Download Options")
    
    # Create two columns for download options
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Download Individual Images")
        # Create a selectbox for individual image download
        selected_image = st.selectbox(
            "Select image to download:",
            options=[result.get("filename") for result in results],
            format_func=lambda x: f"{x} ({next((result.get('size_kb') for result in results if result.get('filename') == x), '')})"
        )
        
        if selected_image:
            try:
                response = requests.get(
                    f"{API_URL}/images/{doc_type}/download/{selected_image}",
                    stream=True
                )
                if response.status_code == 200:
                    st.download_button(
                        label="📥 Download Selected Image",
                        data=response.content,
                        file_name=selected_image,
                        mime="image/png"
                    )
                else:
                    st.error("Failed to prepare image for download")
            except Exception as e:
                st.error(f"Error preparing download: {str(e)}")
    
    with col2:
        st.write("#### Download All Images")
        try:
            response = requests.get(
                f"{API_URL}/images/{doc_type}/download-all",
                stream=True
            )
            if response.status_code == 200:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                zip_filename = f"converted_images_{doc_type}_{timestamp}.zip"
                st.download_button(
                    label="📥 Download All as ZIP",
                    data=response.content,
                    file_name=zip_filename,
                    mime="application/zip"
                )
            else:
                st.error("Failed to prepare ZIP archive")
        except Exception as e:
            st.error(f"Error preparing ZIP download: {str(e)}")

def display_pdf2png_tab():
    """Display the PDF to PNG conversion interface"""
    # Create tabs for different sections of PDF2PNG
    tab1, tab2, tab3 = st.tabs(["📋 Overview", "📤 Upload & Convert", "📊 History"])
    
    with tab1:
        st.header("📄 PDF to PNG Converter Overview")
        
        st.markdown("""
        ### Automated PDF to PNG Conversion Tool
        
        This tool allows you to convert PDF documents to PNG images for further processing in the system.
        
        #### 🚀 Key Features
        
        - **Batch Processing**: Upload a single PDF file or a ZIP archive containing multiple PDFs
        - **Memory Efficient**: Large PDFs are automatically processed in chunks to prevent memory issues
        - **Database Integration**: All generated images are stored in the database for easy access
        - **Filtering**: Non-PDF files are automatically filtered out from ZIP archives
        - **Document Type Support**: Integrates with facturas, pagos and nóminas modules
        - **Download Options**: Download converted images individually or as a ZIP archive
        
        #### 💼 Business Benefits
        
        - **Time Savings**: Eliminate manual conversion steps
        - **Error Reduction**: Standardized processing prevents conversion errors
        - **Immediate Availability**: Converted images are immediately available for processing
        - **Tracking**: Conversion history is maintained for audit purposes
        
        #### 📝 How It Works
        
        1. **Upload**: Submit a PDF file or ZIP archive containing PDFs
        2. **Conversion**: System automatically converts all PDF pages to PNG images
        3. **Storage**: Images are stored in the database and file system
        4. **Processing**: Converted images are ready for document processing
        5. **Download**: Download converted images individually or as a ZIP archive
        
        #### 📊 Supported Formats
        
        - Individual PDF files (`.pdf`)
        - ZIP archives containing PDF files (`.zip`)
        
        #### ⚙️ Technical Details
        
        - Each PDF page becomes a separate PNG image
        - Large PDFs (over 100 pages) are processed in chunks
        - Only PDF files are processed, other file types are ignored
        """)
    
    with tab2:
        st.header("📤 Upload & Convert PDFs")
        
        # Add document type selection
        doc_type = st.radio(
            "Select Document Type:",
            ["General", "Facturas", "Pagos", "Nóminas"],
            horizontal=True,
            help="Select the type of document you're uploading to organize files appropriately"
        )
        
        # Map selection to API endpoint
        doc_type_map = {
            "General": None,
            "Facturas": "facturas",
            "Pagos": "pagos", 
            "Nóminas": "nominas"
        }
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload a PDF or ZIP file containing PDFs",
            type=["pdf", "zip"],
            help="Select a single PDF file or a ZIP archive containing multiple PDFs"
        )
        
        if uploaded_file:
            file_details = {
                "FileName": uploaded_file.name,
                "FileType": uploaded_file.type,
                "FileSize": f"{uploaded_file.size / 1024:.2f} KB"
            }
            
            # Display file details
            st.write("### File Details:")
            for key, value in file_details.items():
                st.write(f"**{key}:** {value}")
            
            # Process button
            if st.button("🔄 Convert to PNG", type="primary"):
                with st.spinner("Converting PDF to PNG images..."):
                    try:
                        # Determine endpoint based on document type
                        selected_doc_type = doc_type_map[doc_type]
                        
                        if not selected_doc_type:
                            st.error("Please select a document type (Facturas, Pagos, or Nóminas)")
                            return
                        
                        # Send file to API
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        endpoint = f"{API_URL}/upload/pdf/{selected_doc_type}"
                        response = requests.post(endpoint, files=files)
                        
                        if response.status_code == 200:
                            results = response.json()
                            
                            # Display success message with count
                            st.success(f"✅ Successfully converted to {len(results)} PNG images!")
                            
                            # For our local database backup - conversion history will be recorded by the backend
                            history_entry = {
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "filename": uploaded_file.name,
                                "pages": len(results),
                                "status": "Success",
                                "file_size": file_details["FileSize"],
                                "document_type": doc_type
                            }
                            
                            # Save to local DB as backup
                            save_history_to_local_db(history_entry)
                            
                            # Update session state
                            if st.session_state.pdf_conversion_history is None:
                                st.session_state.pdf_conversion_history = []
                            st.session_state.pdf_conversion_history.insert(0, history_entry)
                            
                            # Display results in a table
                            if results:
                                df = pd.DataFrame(results)
                                if "size" in df.columns:
                                    df["size_kb"] = df["size"].apply(lambda x: f"{x/1024:.2f} KB")
                                
                                st.write("### Generated Images:")
                                st.dataframe(
                                    df,
                                    column_config={
                                        "filename": st.column_config.TextColumn("Filename"),
                                        "size_kb": st.column_config.TextColumn("Size")
                                    },
                                    hide_index=True
                                )
                                
                                # Add download functionality for converted images
                                download_converted_images(selected_doc_type, results)
                                
                                # Add button to view images
                                if st.button("👁️ View Generated Images"):
                                    # Redirect to upload tab where images can be viewed/processed
                                    st.session_state.show_processed = True
                                    st.rerun()
                                    
                                # Add refresh button to update history
                                if st.button("📋 View Conversion History"):
                                    # Reset PDF conversion history
                                    try:
                                        response = requests.get(f"{API_URL}/pdf/history")
                                        if response.status_code == 200:
                                            history_data = response.json()
                                            # Convert timestamp strings to datetime objects
                                            for entry in history_data:
                                                if isinstance(entry.get("timestamp"), str):
                                                    try:
                                                        entry["timestamp"] = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
                                                    except (ValueError, TypeError):
                                                        # Keep as string if conversion fails
                                                        pass
                                            st.session_state.pdf_conversion_history = history_data
                                    except Exception:
                                        # Silently continue if API fails
                                        pass
                                        
                                    # Switch to history tab
                                    st.rerun()
                                
                        else:
                            error_message = f"Error converting PDF: {response.text}"
                            st.error(error_message)
                            
                            # Add failed conversion to local history (API should handle this too)
                            history_entry = {
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "filename": uploaded_file.name,
                                "pages": 0,
                                "status": f"Failed: {response.text}",
                                "file_size": file_details["FileSize"],
                                "document_type": doc_type
                            }
                            
                            # Save to local DB as backup
                            save_history_to_local_db(history_entry)
                            
                            # Update session state
                            if st.session_state.pdf_conversion_history is None:
                                st.session_state.pdf_conversion_history = []
                            st.session_state.pdf_conversion_history.insert(0, history_entry)
                            
                    except Exception as e:
                        error_message = f"Error: {str(e)}"
                        st.error(error_message)
                        
                        # Add failed conversion to local history
                        history_entry = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "filename": uploaded_file.name,
                            "pages": 0,
                            "status": f"Failed: {str(e)}",
                            "file_size": file_details["FileSize"],
                            "document_type": doc_type
                        }
                        
                        # Save to local DB as backup
                        save_history_to_local_db(history_entry)
                        
                        # Update session state
                        if st.session_state.pdf_conversion_history is None:
                            st.session_state.pdf_conversion_history = []
                        st.session_state.pdf_conversion_history.insert(0, history_entry)
    
    with tab3:
        st.header("📊 Conversion History")
        
        # Add download section at the top of the History tab
        st.subheader("📥 Download Converted Images")
        
        # Create tabs for different document types
        doc_tabs = st.tabs(["📑 Facturas", "💳 Pagos", "👥 Nóminas"])
        
        for idx, (doc_type, api_type) in enumerate([
            ("Facturas", "facturas"),
            ("Pagos", "pagos"),
            ("Nóminas", "nominas")
        ]):
            with doc_tabs[idx]:
                try:
                    # Get images for this document type
                    response = requests.get(f"{API_URL}/images/{api_type}")
                    if response.status_code == 200:
                        images = response.json()
                        if images:
                            # Create two columns for download options
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"#### Individual {doc_type} Images")
                                # Create a selectbox for individual image download
                                selected_image = st.selectbox(
                                    f"Select {doc_type} image:",
                                    options=[img.get("filename") for img in images],
                                    key=f"select_{api_type}"
                                )
                                
                                if selected_image:
                                    try:
                                        response = requests.get(
                                            f"{API_URL}/images/{api_type}/download/{selected_image}",
                                            stream=True
                                        )
                                        if response.status_code == 200:
                                            st.download_button(
                                                label=f"📥 Download {doc_type} Image",
                                                data=response.content,
                                                file_name=selected_image,
                                                mime="image/png",
                                                key=f"download_{api_type}_single"
                                            )
                                    except Exception as e:
                                        st.error(f"Error preparing download: {str(e)}")
                            
                            with col2:
                                st.write(f"#### All {doc_type} Images")
                                try:
                                    response = requests.get(
                                        f"{API_URL}/images/{api_type}/download-all",
                                        stream=True
                                    )
                                    if response.status_code == 200:
                                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                        zip_filename = f"converted_images_{api_type}_{timestamp}.zip"
                                        st.download_button(
                                            label=f"📥 Download All {doc_type} as ZIP",
                                            data=response.content,
                                            file_name=zip_filename,
                                            mime="application/zip",
                                            key=f"download_{api_type}_zip"
                                        )
                                except Exception as e:
                                    st.error(f"Error preparing ZIP download: {str(e)}")
                        else:
                            st.info(f"No converted {doc_type} images available")
                except Exception as e:
                    st.error(f"Error loading {doc_type} images: {str(e)}")
        
        st.divider()
        
        # Refresh button to load latest history from database
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🔄 Refresh History", key="refresh_db_history"):
                # Try to get history from API first, fall back to local DB if needed
                try:
                    response = requests.get(f"{API_URL}/pdf/history")
                    if response.status_code == 200:
                        history_data = response.json()
                        # Convert timestamp strings to datetime objects
                        for entry in history_data:
                            if isinstance(entry.get("timestamp"), str):
                                try:
                                    entry["timestamp"] = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
                                except (ValueError, TypeError):
                                    # Keep as string if conversion fails
                                    pass
                        st.session_state.pdf_conversion_history = history_data
                        st.success("History refreshed from server database!")
                    else:
                        # Fallback to local database
                        history_data = get_history_from_local_db()
                        for entry in history_data:
                            if isinstance(entry.get("timestamp"), str):
                                try:
                                    entry["timestamp"] = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
                                except (ValueError, TypeError):
                                    pass
                        st.session_state.pdf_conversion_history = history_data
                        st.info("Using local database (server history not available)")
                except Exception as e:
                    # Fallback to local database
                    history_data = get_history_from_local_db()
                    for entry in history_data:
                        if isinstance(entry.get("timestamp"), str):
                            try:
                                entry["timestamp"] = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
                            except (ValueError, TypeError):
                                pass
                    st.session_state.pdf_conversion_history = history_data
                    st.warning(f"Error connecting to server, using local history: {str(e)}")
        
        with col2:
            # Add a button to delete all PDF files from the history
            if st.button("🗑️ Delete All PDF Files", key="delete_all_pdfs"):
                st.warning("⚠️ This will delete all PDF files and their conversion history")
                confirm = st.checkbox("I understand this action cannot be undone", key="confirm_pdf_deletion")
                
                if confirm and st.button("⚠️ Confirm PDF Deletion", type="primary", key="confirm_pdf_delete"):
                    with st.spinner("Deleting PDF files and history..."):
                        try:
                            # Call API to clear PDF history
                            response = requests.delete(f"{API_URL}/pdf/history/clear")
                            
                            # Also delete PDF files from the filesystem
                            response2 = requests.delete(
                                f"{API_URL}/images/delete",
                                params={
                                    "delete_transactions": False,
                                    "delete_pdf_history": True
                                }
                            )
                            
                            if response.status_code == 200 and response2.status_code == 200:
                                # Clear local session state history
                                st.session_state.pdf_conversion_history = []
                                st.success("Successfully deleted all PDF files and history!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"Failed to delete PDF history: {response.text} or {response2.text}")
                        except Exception as e:
                            st.error(f"Error deleting PDF history: {str(e)}")
        
        # Display conversion history if available
        if st.session_state.pdf_conversion_history:
            # First create a dataframe for display
            history_df = pd.DataFrame(st.session_state.pdf_conversion_history)
            
            # Container for the table
            table_container = st.container()
            
            # Create buttons for each row with expander
            with st.expander("🗑️ Delete Individual Records", expanded=False):
                st.write("Select records to delete:")
                
                # Create a container for delete buttons
                delete_cols = st.columns(3)
                
                for i, record in enumerate(st.session_state.pdf_conversion_history):
                    # Determine which column to use (cycle through the 3 columns)
                    col_idx = i % 3
                    
                    # Display delete button in the appropriate column
                    with delete_cols[col_idx]:
                        record_id = record.get("id")
                        if record_id:
                            filename = record.get("filename", "Unknown")
                            # Truncate filename if too long
                            if len(filename) > 20:
                                display_name = filename[:17] + "..."
                            else:
                                display_name = filename
                                
                            if st.button(f"🗑️ Delete {display_name}", key=f"delete_record_{record_id}"):
                                try:
                                    # Try to delete from API first
                                    api_success = False
                                    try:
                                        response = requests.delete(f"{API_URL}/pdf/history/{record_id}")
                                        if response.status_code == 200:
                                            api_success = True
                                    except Exception:
                                        api_success = False
                                        
                                    # Always delete from local DB as backup
                                    local_success = delete_history_record_from_local_db(record_id)
                                    
                                    if api_success or local_success:
                                        # Remove from session state
                                        st.session_state.pdf_conversion_history = [
                                            r for r in st.session_state.pdf_conversion_history 
                                            if r.get("id") != record_id
                                        ]
                                        st.success(f"Record for '{filename}' deleted!")
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error(f"Failed to delete record")
                                except Exception as e:
                                    st.error(f"Error deleting record: {str(e)}")
            
            # Add download functionality for converted images
            with st.expander("📥 Download Converted Images", expanded=True):
                st.write("Download images from conversion history:")
                
                # Group records by document type
                doc_types = history_df["document_type"].unique()
                
                for doc_type in doc_types:
                    if doc_type in ["Facturas", "Pagos", "Nóminas"]:
                        # Map document types to API endpoints
                        doc_type_map = {
                            "Facturas": "facturas",
                            "Pagos": "pagos",
                            "Nóminas": "nominas"
                        }
                        
                        api_doc_type = doc_type_map.get(doc_type)
                        if api_doc_type:
                            st.write(f"#### {doc_type}")
                            
                            # Get converted images for this document type
                            try:
                                response = requests.get(f"{API_URL}/images/{api_doc_type}")
                                if response.status_code == 200:
                                    images = response.json()
                                    if images:
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            # Individual image download
                                            selected_image = st.selectbox(
                                                f"Select {doc_type} image to download:",
                                                options=[img.get("filename") for img in images],
                                                key=f"select_{api_doc_type}"
                                            )
                                            
                                            if selected_image:
                                                try:
                                                    response = requests.get(
                                                        f"{API_URL}/images/{api_doc_type}/download/{selected_image}",
                                                        stream=True
                                                    )
                                                    if response.status_code == 200:
                                                        st.download_button(
                                                            label=f"📥 Download {doc_type} Image",
                                                            data=response.content,
                                                            file_name=selected_image,
                                                            mime="image/png",
                                                            key=f"download_{api_doc_type}_single"
                                                        )
                                                except Exception as e:
                                                    st.error(f"Error preparing download: {str(e)}")
                                        
                                        with col2:
                                            # Download all as ZIP
                                            try:
                                                response = requests.get(
                                                    f"{API_URL}/images/{api_doc_type}/download-all",
                                                    stream=True
                                                )
                                                if response.status_code == 200:
                                                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                                    zip_filename = f"converted_images_{api_doc_type}_{timestamp}.zip"
                                                    st.download_button(
                                                        label=f"📥 Download All {doc_type} as ZIP",
                                                        data=response.content,
                                                        file_name=zip_filename,
                                                        mime="application/zip",
                                                        key=f"download_{api_doc_type}_zip"
                                                    )
                                            except Exception as e:
                                                st.error(f"Error preparing ZIP download: {str(e)}")
                                    else:
                                        st.info(f"No converted images found for {doc_type}")
                            except Exception as e:
                                st.error(f"Error fetching images for {doc_type}: {str(e)}")
            
            # Display the history table
            with table_container:
                st.dataframe(
                    history_df,
                    column_config={
                        "id": st.column_config.NumberColumn("ID"),
                        "timestamp": st.column_config.DatetimeColumn("Timestamp"),
                        "filename": "Filename",
                        "pages": "Pages",
                        "status": "Status",
                        "file_size": "File Size",
                        "document_type": "Document Type"
                    },
                    hide_index=True
                )
            
            col1, col2 = st.columns([1, 3])
            with col2:
                if st.button("🗑️ Clear All History", key="clear_history"):
                    try:
                        # Try API first
                        api_success = False
                        try:
                            response = requests.delete(f"{API_URL}/pdf/history/clear")
                            if response.status_code == 200:
                                api_success = True
                        except Exception:
                            api_success = False
                            
                        # Always clear local DB as backup
                        local_success = clear_history_from_local_db()
                        
                        if api_success or local_success:
                            # Clear local session state history
                            st.session_state.pdf_conversion_history = []
                            st.success("History cleared!" + (" (API Success)" if api_success else " (Local DB only)"))
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Failed to clear history")
                    except Exception as e:
                        st.error(f"Error clearing history: {str(e)}")
        else:
            st.info("No conversion history available. Convert some PDF files to see the history here.")

def main():
    # Initialize session state
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 AutoAudit Pro")
        st.markdown("---")
        
        # Updated module selection to include Upload
        selected_page = st.radio(
            "Select Module:",
            ["PDF2PNG", "Upload", "Processing"],
            index=0,
            format_func=lambda x: f"📄 {x}" if x == "PDF2PNG" else f"📤 {x}" if x == "Upload" else f"⚙️ {x}"
        )
    
    # Main content based on selection
    if selected_page == "PDF2PNG":
        st.title("📄 PDF to PNG Converter")
        display_pdf2png_tab()
    elif selected_page == "Upload":
        st.title("📤 Document Upload")
        display_upload_tab()
    else:  # Processing
        st.title("⚙️ Document Processing")
        display_process_tab()

if __name__ == "__main__":
    main()
