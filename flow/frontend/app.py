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

# Load environment variables
load_dotenv()

# Constants
TEXTRACT_CONFIDENCE_THRESHOLD = float(os.getenv("TEXTRACT_CONFIDENCE_THRESHOLD", 0.7))
API_URL = os.getenv("SERVER_URL", "http://localhost:8000")
ALLOWED_EXTENSIONS = ['.png', '.jpg', '.jpeg']

def init_session_state():
    """Initialize session state variables"""
    if 'selected_files' not in st.session_state:
        st.session_state.selected_files = []
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = {}
    if 'show_processed' not in st.session_state:
        st.session_state.show_processed = False
    if 'projects' not in st.session_state:
        st.session_state.projects = {
            "ICEX": {
                "name": "ICEX",
                "description": "Document processing system for ICEX (Instituto Español de Comercio Exterior)",
                "modules": ["PAGOS", "FACTURAS", "NÓMINAS"]
            },
            "UAB": {
                "name": "UAB",
                "description": "Document processing system for UAB (Universitat Autònoma de Barcelona)",
                "modules": ["PAGOS", "FACTURAS", "NÓMINAS"]
            }
        }
    if 'selected_project' not in st.session_state:
        st.session_state.selected_project = "ICEX"
    if 'show_project_creator' not in st.session_state:
        st.session_state.show_project_creator = False
    if 'processing_history' not in st.session_state:
        st.session_state.processing_history = []
    if 'error_log' not in st.session_state:
        st.session_state.error_log = []
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()

def create_new_project():
    """Display the project creation interface"""
    with st.sidebar.expander("➕ Create New Project", expanded=st.session_state.show_project_creator):
        st.markdown("### Create New Project")
        
        # Project name input
        new_project_name = st.text_input("Project Name", key="new_project_name")
        
        # Module selection
        selected_modules = st.multiselect(
            "Select Modules",
            ["PAGOS y FACTURAS", "NÓMINAS"],
            default=["PAGOS"],
            key="new_project_modules"
        )
        
        # Project description
        project_description = st.text_area(
            "Project Description",
            help="Brief description of the project",
            key="new_project_description"
        )
        
        # Create project button
        if st.button("Create Project", type="primary"):
            if new_project_name and selected_modules:
                if new_project_name in st.session_state.projects:
                    st.error("A project with this name already exists!")
                else:
                    # Add new project to session state
                    st.session_state.projects[new_project_name] = {
                        "name": new_project_name,
                        "description": project_description,
                        "modules": selected_modules
                    }
                    # Select the new project
                    st.session_state.selected_project = new_project_name
                    st.session_state.show_project_creator = False
                    st.rerun()
            else:
                st.error("Please provide a project name and select at least one module!")

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

def display_upload_tab():
    """Display the upload interface with separate sections"""
    st.header("📤 Upload Documents")
    
    # Add delete options in expander
    with st.expander("🗑️ Delete Options"):
        delete_transactions = st.checkbox(
            "Include Transactions", 
            help="Also delete all stored transactions from the database"
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
                                    "delete_transactions": delete_transactions
                                }
                            )
                            if response.status_code == 200:
                                result = response.json()
                                st.success("Successfully deleted all data")
                                
                                # Clear session states
                                st.session_state.processed_files = []
                                st.session_state.show_delete_confirmation = False
                                
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

def display_process_tab():
    """Display the process tab content"""
    st.subheader("Process Documents")
    
    # Add batch processing configuration
    with st.expander("🔧 Processing Configuration"):
        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=10,
            value=3,
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

def display_pagos_overview(project):
    """Display project-specific pagos overview"""
    if project == "ICEX":
        st.markdown("""
        ### 📊 ICEX Payment Processing System
        
        This module processes various payment documents for ICEX:
        
        #### 📄 Supported Document Types
        - Bank Transfers
        - Credit Card Statements
        - Payment Orders
        - Bank Certificates
        
        #### 🔄 Processing Features
        - Automated data extraction
        - Payment validation
        - Transaction categorization
        - Audit trail generation
        """)
    else:  # UAB
        st.markdown("""
        ### 📊 UAB Payment Processing System
        
        This module processes various payment documents for UAB:
        
        #### 📄 Supported Document Types
        - University Payments
        - Research Grants
        - Department Expenses
        - Student Payments
        
        #### 🔄 Processing Features
        - Automated data extraction
        - Department allocation
        - Budget tracking
        - Audit trail generation
        """)

def display_facturas_overview(project):
    """Display project-specific facturas overview"""
    if project == "ICEX":
        st.markdown("""
        ### 📑 ICEX Invoice Processing System
        
        This module processes invoices and commercial documents for ICEX:
        
        #### 📄 Supported Document Types
        - Commercial Invoices
        - Expense Reports
        - Purchase Orders
        - Credit Notes
        
        #### 🔄 Processing Features
        - Automated data extraction
        - VAT and tax calculation validation
        - Supplier verification
        - Invoice matching and reconciliation
        - Audit trail generation
        """)
    else:  # UAB
        st.markdown("""
        ### 📑 UAB Invoice Processing System
        
        *Coming soon*
        
        This module will process invoices and commercial documents for UAB.
        """)

def display_nominas_overview(project):
    """Display project-specific nominas overview"""
    if project == "ICEX":
        st.markdown("""
        ### 👥 ICEX Payroll Processing System
        
        *Coming soon*
        
        This module will process invoice documents for ICEX.
        """)
    else:  # UAB
        st.markdown("""
        ### 👥 UAB Payroll Processing System
        
        *Coming soon*
        
        This module will process invoice documents for UAB.
        """)

def display_facturas_results():
    """Display the results interface for facturas"""
    st.header("📊 Results")
    
    try:
        # Get processed facturas from the API
        response = requests.get(f"{API_URL}/facturas/")
        if response.status_code != 200:
            st.error("Failed to fetch invoice data")
            return
            
        facturas = response.json()
        
        if not facturas:
            st.info("No processed invoices found")
            return
        
        # Create DataFrame from all rows in all facturas
        all_rows = []
        for factura in facturas:
            for row in factura['rows']:
                all_rows.append({
                    "CIF Cliente": row['CIF_CLIENTE'],
                    "Cliente": row['CLIENTE'],
                    "Fichero": row['FICHERO'],
                    "Número Factura": row['NUMERO_FACTURA'],
                    "Fecha Factura": row['FECHA_FACTURA'],
                    "Proveedor": row['PROVEEDOR'],
                    "Base Imponible": row['BASE_IMPONIBLE'],
                    "CIF Proveedor": row['CIF_PROVEEDOR'],
                    "IRPF": row['IRPF'],
                    "IVA": row['IVA'],
                    "Total Factura": row['TOTAL_FACTURA']
                })
        
        df = pd.DataFrame(all_rows)
        
        # Display the DataFrame
        st.dataframe(
            df,
            hide_index=True,
            column_config={
                "CIF Cliente": "CIF Cliente",
                "Cliente": "Cliente",
                "Fichero": "Fichero",
                "Número Factura": "Número Factura",
                "Fecha Factura": "Fecha Factura",
                "Proveedor": "Proveedor",
                "Base Imponible": st.column_config.NumberColumn("Base Imponible", format="€%.2f"),
                "CIF Proveedor": "CIF Proveedor",
                "IRPF": st.column_config.NumberColumn("IRPF", format="%.2f%%"),
                "IVA": st.column_config.NumberColumn("IVA", format="%.2f%%"),
                "Total Factura": st.column_config.NumberColumn("Total Factura", format="€%.2f")
            }
        )
        
        # Add export functionality using the combined endpoint
        if st.button("Export Results to Excel", key="export_facturas_btn"):
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
                            key="download_facturas_excel_btn"
                        )
                    else:
                        st.error(f"Failed to generate Excel file: {response.text}")
                except Exception as e:
                    st.error(f"Error downloading Excel: {str(e)}")
            
    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")

def process_factura(filename):
    """Process a single invoice image"""
    try:
        response = requests.post(f"{API_URL}/images/{filename}/process")
        return response.json()
    except Exception as e:
        return {"success": False, "message": str(e)}

def main():
    # Set page config
    st.set_page_config(
        page_title="AutoAudit Pro",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 AutoAudit Pro")
        st.markdown("---")
        
        # Project management
        col1, col2 = st.columns([3, 1])
        with col1:
            # Project selection
            st.session_state.selected_project = st.selectbox(
                "Select Project:",
                options=list(st.session_state.projects.keys()),
                key="project_selector"
            )
        with col2:
            # New project button
            if st.button("➕ New", key="new_project_btn"):
                st.session_state.show_project_creator = True
                st.rerun()
        
        # Show project creator if requested
        if st.session_state.show_project_creator:
            create_new_project()
        
        # Get current project modules
        current_project = st.session_state.projects[st.session_state.selected_project]
        available_modules = current_project["modules"]
        
        # Module selection
        selected_page = st.radio(
            "Select Module:",
            available_modules,
            index=0,
            format_func=lambda x: f"📊 {x}" if x == "PAGOS" else (f"📑 {x}" if x == "FACTURAS" else f"👥 {x}")
        )
        
        st.markdown("---")
        
        # Project-specific information
        st.markdown(f"### 📋 {st.session_state.selected_project}")
        st.markdown(current_project["description"])
        
        st.markdown("---")
        st.markdown("### 📞 Support")
        st.markdown("""
        For technical support:
        - ✉️ user@contact.com
        - 📱 +34 688 99 00 93
        """)
    
    # Main content
    st.title(f"📊 {st.session_state.selected_project} - {selected_page}")
    
    # Display tabs based on selected module
    if selected_page == "PAGOS" and "PAGOS" in available_modules:
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Overview", 
            "📤 Upload", 
            "⚙️ Process", 
            "📊 Results"
        ])
        
        with tab1:
            display_pagos_overview(st.session_state.selected_project)
        with tab2:
            display_upload_tab()
        with tab3:
            display_process_tab()
        with tab4:
            display_results_tab()
            
    elif selected_page == "FACTURAS" and "FACTURAS" in available_modules:
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Overview",
            "📤 Upload",
            "⚙️ Process",
            "📊 Results"
        ])
        
        with tab1:
            display_facturas_overview(st.session_state.selected_project)
        with tab2:
            display_upload_tab()
        with tab3:
            display_process_tab()
        with tab4:
            display_results_tab()
            
    elif selected_page == "NÓMINAS" and "NÓMINAS" in available_modules:
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Overview",
            "📤 Upload",
            "⚙️ Process",
            "📊 Results"
        ])
        
        with tab1:
            display_nominas_overview(st.session_state.selected_project)
        with tab2:
            display_upload_tab()
        with tab3:
            display_process_tab()
        with tab4:
            display_results_tab()

if __name__ == "__main__":
    main()
