import streamlit as st
import pandas as pd
import sys
import subprocess
import os
from utils.common import set_page_config, get_model_suffix
sys.path.append('../../')
from backend.config import MODEL_NAME, VM_IP, USERNAME, PASSWORD, LOCAL_PATH, FLAG_TESTINLOCAL

# Use common page config
set_page_config()

st.title("🔍 Prueba de Modelos IA")

# Document type selection
st.subheader("📄 Tipo de Documento")
doc_type = st.selectbox(
    "Seleccione el tipo de documento a procesar",
    options=[
        "📊 Facturas",
        "💰 Nóminas",
        "💳 Pagos"
    ],
    format_func=lambda x: x
)

model_suffix = get_model_suffix(doc_type)
model_name = f"lora_{model_suffix}"

# File upload section
st.subheader("📎 Cargar Archivo")
uploaded_file = st.file_uploader("Seleccione un archivo Excel", type=['xlsx', 'xls'])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.write("Vista previa del archivo:")
        st.dataframe(df.head())
        
        # Test button
        if st.button("🔬 Iniciar Prueba"):
            with st.spinner(f"Procesando {doc_type}..."):
                try:
                    # Save uploaded file temporarily
                    with open("temp_test.xlsx", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Execute test script
                    result = subprocess.run(
                        ["python", "../backend/test.py", 
                         "--model", model_name,
                         "--input", "temp_test.xlsx"],
                        capture_output=True,
                        text=True
                    )
                    
                    # Show results
                    if result.returncode == 0:
                        st.success("✅ Prueba completada exitosamente!")
                        
                        # If results file exists, offer download
                        if os.path.exists("results.xlsx"):
                            with open("results.xlsx", "rb") as file:
                                st.download_button(
                                    label="📥 Descargar Resultados",
                                    data=file,
                                    file_name=f"resultados_{doc_type.split()[1].lower()}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                    else:
                        st.error("❌ Error en la prueba. Revise los logs para más detalles.")
                    
                    # Show logs
                    with st.expander("Ver Logs", expanded=True):
                        if result.stdout:
                            st.text("Salida Estándar:")
                            st.code(result.stdout)
                        if result.stderr:
                            st.text("Error Estándar:")
                            st.code(result.stderr)
                
                finally:
                    # Clean up temporary files
                    if os.path.exists("temp_test.xlsx"):
                        os.remove("temp_test.xlsx")
                    if os.path.exists("results.xlsx"):
                        os.remove("results.xlsx")
                        
    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {str(e)}")

# Add instructions
st.markdown("---")
st.markdown(f"""
    ### 📝 Instrucciones para Pruebas:
    1. Seleccione el tipo de documento a procesar
    2. Cargue su archivo Excel
    3. Haga clic en 'Iniciar Prueba'
    4. Descargue los resultados cuando estén listos
    
    ℹ️ Modelo a utilizar: `{model_name}`
    📍 Ubicación: {'Local' if FLAG_TESTINLOCAL else 'Servidor VM'}
""")