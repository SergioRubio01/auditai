import streamlit as st
import pandas as pd
import sys
import os
import subprocess
import logging
import datetime
import zipfile
import shutil
from pathlib import Path
import time
import json

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.common import set_page_config, get_model_suffix, get_model_selection
from backend.config import MODEL_NAME, VM_IP, USERNAME, PASSWORD, LOCAL_PATH, FLAG_TESTINLOCAL, IMPORT_HFMODEL, COMMUNITY_NAME

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('streamlit_app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Use common page config
set_page_config()

st.title("🤖 Panel de Control de Entrenamiento IA 🧠")

# Document type selection
st.subheader("📄 Tipo de Documento")
doc_type = st.selectbox(
    "Seleccione el tipo de documento a procesar",
    options=[
        "📊 Facturas - GT",
        "💰 Nóminas - GT", 
        "💳 Pagos - GT",
        "🔧 Modelo Personalizado"
    ],
    format_func=lambda x: x
)

# Show "Coming Soon" message for custom model
if doc_type == "🔧 Modelo Personalizado":
    st.info("Muy Pronto disponible")

# Update model name based on selection
model_suffix = get_model_suffix(doc_type)

# After document type selection and before server configuration
if IMPORT_HFMODEL:
    hf_model = get_model_selection()
    if hf_model:
        st.session_state['hf_model'] = hf_model

# File Upload Section
st.subheader("📁 Subir Archivos")
with st.expander("Subir y Procesar Archivos. Se deben subir tanto un archivo ZIP como un archivo Excel", expanded=True):
    uploaded_zip = st.file_uploader("Subir archivo ZIP con imágenes", type="zip")
    uploaded_excel = st.file_uploader("Subir archivo Excel con datos", type=["xlsx", "xls"])
    
    # Get the community name from the ZIP file as soon as it's uploaded
    if uploaded_zip:
        zip_filename = Path(uploaded_zip.name).stem
        os.environ['COMMUNITY_NAME'] = zip_filename
        st.session_state['community_name'] = zip_filename
        st.info(f"Comunidad detectada: {zip_filename}")
    
    if uploaded_zip and uploaded_excel:
        # Create a trigger file to start training
        with open("start_training.trigger", "w") as f:
            pass
        # Create temporary directory for processing
        temp_dir = "temp_upload"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save and extract ZIP file
        zip_path = os.path.join(temp_dir, "pdfs.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getvalue())
        
        # Extract ZIP and verify its contents
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List all files in the ZIP
            file_list = zip_ref.namelist()
            logger.info(f"Files in ZIP: {file_list}")
            
            # Extract all files
            zip_ref.extractall(temp_dir)
        
        # Save Excel file
        excel_path = os.path.join(temp_dir, "data.xlsx")
        with open(excel_path, "wb") as f:
            f.write(uploaded_excel.getvalue())
        
        # Read Excel file and verify file references
        df = pd.read_excel(excel_path)
        
        # Get the path to the extracted files
        extracted_files = []
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png')):
                    extracted_files.append(os.path.join(root, file))
        
        if not extracted_files:
            st.error("❌ No se encontraron archivos de imagen o PDF en el ZIP")
            st.stop()
        
        logger.info(f"Found {len(extracted_files)} files in the extracted directory")
        st.info(f"📁 Se encontraron {len(extracted_files)} archivos en el ZIP")
        
        # Read Excel file to get column names
        columns = df.columns.tolist()
        
        # Display available columns
        st.write("📊 Columnas disponibles en el archivo Excel:")
        for col in columns:
            st.write(f"- {col}")
        
        # Let user select multiple columns for the solution
        selected_columns = st.multiselect(
            "Seleccione las columnas que contienen la información relevante",
            options=columns,
            help="Seleccione todas las columnas que contengan información necesaria para el entrenamiento"
        )
        
        # Let user select the filename column separately
        filename_column = st.selectbox(
            "Seleccione la columna que contiene los nombres de archivo",
            options=columns
        )
        
        if st.button("✨ Procesar Archivos"):
            if not selected_columns:
                st.error("❌ Debe seleccionar al menos una columna para procesar")
                st.stop()
            
            try:
                # Create new DataFrame with selected columns plus filename
                columns_to_use = list(set([filename_column] + selected_columns))
                new_df = df[columns_to_use]
                
                # Create dataset.json from the DataFrame
                dataset_records = []
                for _, row in new_df.iterrows():
                    record = {
                        "file_name": row[filename_column],
                        "text": " ".join(str(row[col]) for col in selected_columns if col != filename_column)
                    }
                    dataset_records.append(record)
                
                # Save dataset.json in the community directory
                backend_dir = os.path.join("..", "backend")
                
                # Ensure we have a valid community name, default to 'Community' if None
                community_name = st.session_state.get('community_name')
                if not community_name:
                    community_name = 'Community'
                    logger.warning(f"No community name found, using default: {community_name}")
                
                community_dir = os.path.join(backend_dir, community_name)
                os.makedirs(community_dir, exist_ok=True)
                
                # Save dataset.json in both locations (community dir and root)
                dataset_path = os.path.join(community_dir, "dataset.json")
                root_dataset_path = os.path.join("..", "backend", "dataset.json")
                
                logger.info(f"Saving dataset to: {dataset_path} and {root_dataset_path}")
                
                # Create the dataset JSON
                dataset_json = {"data": dataset_records}
                
                # Save to community directory
                with open(dataset_path, 'w', encoding='utf-8') as f:
                    json.dump(dataset_json, f, ensure_ascii=False, indent=2)
                
                # Also save to root backend directory for the training script
                with open(root_dataset_path, 'w', encoding='utf-8') as f:
                    json.dump(dataset_json, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Dataset saved to {dataset_path} and {root_dataset_path}")
                
                # Copy files to backend directory
                backend_dir = os.path.join("..", "backend")
                community_dir = os.path.join(backend_dir, COMMUNITY_NAME)
                    
                # Create community directory if it doesn't exist
                os.makedirs(community_dir, exist_ok=True)
                
                # Get the path to the extracted images
                images_path = None
                for root, dirs, files in os.walk(temp_dir):
                    for dir_name in dirs:
                        if dir_name.lower().endswith('all'):  # Matches PagoAll, FacturaAll, etc.
                            images_path = os.path.join(root, dir_name)
                            break
                    if images_path:
                        break
                
                # Copy images folder if found
                if images_path:
                    target_images_path = os.path.join(community_dir, os.path.basename(images_path))
                    if os.path.exists(target_images_path):
                        shutil.rmtree(target_images_path)
                    shutil.copytree(images_path, target_images_path)
                else:
                    st.warning("No se encontró la carpeta de imágenes en el archivo ZIP")
                
                # Copy processed Excel file
                shutil.copy2(excel_path, os.path.join(community_dir, "data.xlsx"))
                
                st.success("✅ Archivos procesados y preparados correctamente!")
                
                # Update base_commands in the training section to use correct transformers version
                base_commands = [
                    """bash -l -c 'source ~/miniconda3/etc/profile.d/conda.sh && \
                        conda activate base && \
                        pip3 install --upgrade pip && \
                        pip3 install jinja2>=3.1.0 && \
                        pip3 install Pillow==10.1.0 && \
                        pip3 install python-dotenv && \
                        pip3 install transformers>=4.46.1 && \
                        pip3 install unsloth && \
                        pip3 install pandas && \
                        pip3 install openpyxl && \
                        pip3 install datasets && \
                        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
                        python3 finetune.py'"""
                ]
                
            except pd.errors.EmptyDataError:
                st.error("❌ El archivo Excel está vacío")
                logger.error("Empty Excel file uploaded")
            except KeyError as e:
                st.error(f"❌ Columna no encontrada en el archivo Excel: {str(e)}")
                logger.error(f"Column not found in Excel file: {str(e)}")
            except PermissionError:
                st.error("❌ Error de permisos al acceder a los archivos")
                logger.error("Permission error while accessing files")
            except Exception as e:
                st.error(f"❌ Error inesperado: {str(e)}")
                logger.error(f"Unexpected error: {str(e)}")
            finally:
                # Clean up temporary directory
                if os.path.exists(temp_dir):
                    try:
                        # Add a small delay to ensure all file handles are released
                        time.sleep(0.5)
                        
                        def remove_readonly(func, path, _):
                            """Clear the readonly bit and reattempt removal"""
                            os.chmod(path, 0o777)
                            func(path)
                        
                        # Use shutil.rmtree with onerror handler
                        shutil.rmtree(temp_dir, onerror=remove_readonly)
                        
                    except FileNotFoundError:
                        logger.warning(f"Some files in {temp_dir} were already removed")
                    except Exception as e:
                        logger.error(f"Error removing temporary directory: {str(e)}")

# Server configuration section
st.subheader("⚙️ Configuración del Servidor")
with st.expander("Mostrar/Editar Configuración"):
    new_vm_ip = st.text_input("IP del Servidor", value=VM_IP)
    new_username = st.text_input("Usuario", value=USERNAME)
    new_password = st.text_input("Contraseña", value=PASSWORD, type="password")
    new_local_path = st.text_input("Ruta Local", value=LOCAL_PATH)
    new_model = st.text_input("Nombre del Modelo", value=f"lora_{model_suffix}")
    new_flag_test = st.checkbox("Modo Prueba Local", value=FLAG_TESTINLOCAL)
    
    if st.button("💾 Guardar Configuración"):
        try:
            with open('../backend/config.py', 'w', encoding='utf-8') as f:
                f.write(f'''# Configuration settings
MODEL_NAME = "{new_model}"
VM_IP = "{new_vm_ip}"
USERNAME = "{new_username}"
PASSWORD = "{new_password}"
LOCAL_PATH = "{new_local_path}"
FLAG_TESTINLOCAL = {new_flag_test}
IMPORT_HFMODEL = {IMPORT_HFMODEL}
HF_MODEL_NAME = "{st.session_state.get('hf_model', '')}"
COMMUNITY_NAME = "{st.session_state.get('community_name', 'Comunidad')}"
''')
            st.success("✅ Configuración guardada exitosamente!")
        except Exception as e:
            st.error(f"❌ Error al guardar la configuración: {str(e)}")

# Training control section
st.subheader(f"🚀 Control de Entrenamiento - {doc_type}")
col1, col2, col3 = st.columns(3)

# Initialize session state for UI elements if they don't exist
if 'training_status' not in st.session_state:
    st.session_state.training_status = {
        'progress': 0,
        'status': '',
        'logs': [],
        'is_training': False
    }

# Create persistent UI elements
progress_bar = st.progress(st.session_state.training_status['progress'])
status_text = st.empty()
log_container = st.container()

with col1:
    if st.button("🎯 Iniciar Entrenamiento", key="start_training"):
        try:
            logger.info(f"Iniciando entrenamiento para {doc_type}")
            st.session_state.training_status['is_training'] = True
            
            # Create trigger file to start training
            trigger_file = os.path.join("..", "backend", "start_training.trigger")
            with open(trigger_file, "w") as f:
                f.write("start")
            
            with st.spinner(f"Entrenamiento de {doc_type} en progreso..."):
                start_time = datetime.datetime.now()
                
                process = subprocess.Popen(
                    ["python", "../backend/main.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Create log expander
                log_expander = log_container.expander("Ver Registros de Entrenamiento", expanded=True)
                
                while True:
                    output = process.stdout.readline()
                    error = process.stderr.readline()
                    
                    if output:
                        st.session_state.training_status['logs'].append(f"INFO: {output.strip()}")
                    if error:
                        st.session_state.training_status['logs'].append(f"ERROR: {error.strip()}")
                    
                    # Update logs
                    log_expander.code('\n'.join(st.session_state.training_status['logs']))
                    
                    # Update progress
                    if "Successfully uploaded" in output:
                        st.session_state.training_status['progress'] = 0.3
                        st.session_state.training_status['status'] = "Archivos subidos..."
                    elif "Starting training" in output:
                        st.session_state.training_status['progress'] = 0.6
                        st.session_state.training_status['status'] = "Entrenamiento en progreso..."
                    
                    progress_bar.progress(st.session_state.training_status['progress'])
                    status_text.text(st.session_state.training_status['status'])
                    
                    return_code = process.poll()
                    if return_code is not None:
                        break
                
                execution_time = datetime.datetime.now() - start_time
                logger.info(f"Tiempo de ejecución: {execution_time}")
                
                if return_code == 0:
                    st.session_state.training_status['progress'] = 1.0
                    st.session_state.training_status['status'] = "¡Completado!"
                    progress_bar.progress(1.0)
                    status_text.text("¡Completado!")
                    st.success(f"✅ Entrenamiento de {doc_type} completado!")
                else:
                    st.error(f"❌ El entrenamiento falló. Revise los logs.")
                    
        except Exception as e:
            logger.error(f"Error en el entrenamiento: {str(e)}")
            st.error(f"❌ Error: {str(e)}")
        finally:
            st.session_state.training_status['is_training'] = False

with col2:
    if st.button("🛑 Detener Entrenamiento"):
        try:
            # Find and kill the training process
            subprocess.run(["pkill", "-f", "main.py"])
            st.warning("⚠️ Entrenamiento detenido")
            logger.warning("Entrenamiento detenido por el usuario")
        except Exception as e:
            st.error(f"Error al detener el entrenamiento: {str(e)}")

with col3:
    if st.button("🔄 Limpiar Logs"):
        try:
            open('streamlit_app.log', 'w').close()
            st.success("Logs limpiados")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Error al limpiar logs: {str(e)}")

# Training status section
st.subheader("📊 Estado del Entrenamiento")
status_col1, status_col2 = st.columns(2)

with status_col1:
    st.metric("Progreso", "0%")
with status_col2:
    st.metric("Uso de GPU", "0 GB")

# Activity log section
with st.expander("📋 Ver Registro de Actividad", expanded=False):
    try:
        with open('streamlit_app.log', 'r', encoding='utf-8') as log_file:
            st.code(log_file.read())
    except FileNotFoundError:
        st.info("No hay registros disponibles.")