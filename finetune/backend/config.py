"""
Configuration Settings for Fine-tuning System

This module contains configuration variables used across the fine-tuning pipeline.
It defines key parameters for model training, VM access, and local testing.

Key configuration variables:
- MODEL_NAME: Name identifier for the fine-tuned model
- VM_IP: IP address of the remote training VM
- USERNAME: SSH username for VM access
- PASSWORD: SSH password for VM access
- LOCAL_PATH: Local directory path for model files
- FLAG_TESTINLOCAL: Boolean to enable local testing mode
- IMPORT_HFMODEL: Boolean to control HuggingFace model importing
- HF_MODEL_NAME: Name of HuggingFace model to import (if enabled)
- FLAG_UPLOAD_DATASET: Boolean to control dataset uploading
- FLAG_CREATE_ENV: Boolean to control environment creation
- DATASET_PATH: Path to the dataset directory

Usage:
    Import specific variables:
    from config import MODEL_NAME, VM_IP

    Import all variables:
    from config import *

Security Note:
    In production, sensitive values (VM_IP, USERNAME, PASSWORD) should be
    stored in environment variables or secure configuration management systems.
"""
import os

# Configuration settings
MODEL_NAME = "pagos_v1"
VM_IP = "209.137.198.193"
USERNAME = "Ubuntu"
VM_PASSWORD = "940f221f-e106-447c-bdb6-856afe92efc0"
LOCAL_PATH = "C:/Users/Sergio/GT/PagosICEX/finetune/backend"
FLAG_TESTINLOCAL = True
IMPORT_HFMODEL = False
HF_MODEL_NAME = ""
FLAG_UPLOAD_DATASET = 0
FLAG_CREATE_ENV = 0
DATASET_PATH = "Baleares/dataset/dataset_transferencias"