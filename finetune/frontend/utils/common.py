import streamlit as st
import os
import sys

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Now we can import from backend
from backend.config import (MODEL_NAME, VM_IP, USERNAME, PASSWORD, 
                          LOCAL_PATH, FLAG_TESTINLOCAL, IMPORT_HFMODEL)

def set_page_config():
    st.set_page_config(
        page_title="Panel de Control IA 🤖",
        page_icon="🧠",
        layout="wide"
    )

def get_model_suffix(doc_type):
    return {
        "📊 Facturas": "facturas",
        "💰 Nóminas": "nominas",
        "💳 Pagos": "pagos"
    }[doc_type]

def get_model_selection():
    """Common function for model selection section"""
    if IMPORT_HFMODEL:
        st.subheader("🤗 Modelo de HuggingFace")
        hf_model = st.text_input(
            "Nombre del modelo (ejemplo: unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit)",
            value="unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
            help="Ingrese el nombre completo del modelo de HuggingFace"
        )
        
        # Add a note about supported models
        st.markdown("""
        ℹ️ **Modelos Soportados**:
        - unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit
        - unsloth/Llama-3.2-11B-Vision-bnb-4bit
        - unsloth/Pixtral-12B-2409-bnb-4bit
        - unsloth/Qwen2-VL-7B-Instruct-bnb-4bit
        - unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit
        """)
        return hf_model
    return None