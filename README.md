# ICEX Payment Document Processing System

![Project Logo](https://www.pdfgear.com/chat-pdf/img/best-ai-pdf-analyzers-1.png)

![Python Version](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Finetune](https://img.shields.io/badge/framework-unsloth-orange)
![Agents](https://img.shields.io/badge/framework-langgraph-orange)

## Overview

This project implements an advanced document processing system for ICEX payments, developed by Grant Thornton. It combines vision-language models with fine-tuning capabilities to accurately process and analyze payment documents.

### Key Features

- 🔍 Automated document analysis and data extraction
- 🤖 Fine-tuned vision-language models for payment processing
- 📊 Excel and JSON data source support
- 🔄 PDF to PNG conversion pipeline
- 🚀 Distributed training support on remote VMs
- 📝 Support for multiple document types:
  - Transferencia emitida (Bank Transfer)
  - Detalle movimiento (Transaction Details)
  - Solicitud de transferencia (Transfer Request)
  - More document types coming soon!

## System Architecture

The project consists of three main components:

1. **Fine-tuning Pipeline** (`/finetune`)
   - Model training and adaptation
   - Document preprocessing with multi-format support
   - Dataset management (Excel or JSON sources)

2. **Inference System** (`/flow`)
   - Multi-agent system for document processing
   - Automated Excel output generation
   - Batch processing capabilities

3. **Data Processing Pipeline**
   - PDF to PNG conversion with multi-page support
   - Flexible data source integration
   - Automated data validation

## Installation

### Prerequisites

- Python 3.11
- CUDA-compatible GPU (16GB+ VRAM recommended)
- Conda package manager

### Basic Setup

1. Create and activate Conda environment:
```bash
conda create --name icex_payments python=3.11 pytorch-cuda=12.1 pytorch cudatoolkit ipywidgets -c pytorch -c nvidia -y
conda activate icex_payments

# Install PyTorch and torchvision first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

2. Install Triton and Unsloth:
```bash
# For Windows:
pip install https://github.com/woct0rdho/triton-windows/releases/download/v3.1.0-windows.post5/triton-3.1.0-cp311-cp311-win_amd64.whl
pip install transformers==4.46.2
pip install git+https://github.com/unslothai/unsloth.git

# For Linux:
pip install triton==2.1.0 transformers==4.46.2
pip install git+https://github.com/unslothai/unsloth.git
```

3. Install additional dependencies:
```bash
pip install -r requirements.txt
```

### Configuration Setup

The project uses `config.py` for configuration management:

```python
# finetune/backend/config.py

# VM Configuration
VM_IP = "your_vm_ip"
USERNAME = "your_vm_username"
VM_PASSWORD = "your_vm_password"

# Model Configuration
MODEL_NAME = "pagos"
LOCAL_PATH = "path/to/local/storage"
FLAG_TESTINLOCAL = True
FLAG_UPLOAD_DATASET = True  # Set to True to upload dataset to VM
```

### Environment Setup

1. Create a `.env` file in the `finetune/backend` directory:
```bash
# finetune/backend/.env
HF_TOKEN=your_huggingface_token
```

2. Add .env to your .gitignore:
```bash
# .gitignore
.env
*.env
```

3. Make sure your .env file has the correct permissions:
```bash
chmod 600 finetune/backend/.env  # Linux/Mac
```

## Usage

### Document Processing Pipeline

1. **Prepare Documents**
```bash
# Process PDFs with specific document types
python finetune/backend/pdf2png.py -c <community_name> -cn <excel_column_name> -t "Transferencia emitida" "Detalle movimiento" "Solicitud de transferencia"
```

2. **Create Dataset**
```bash
# From Excel source
python finetune/backend/database.py -c <community_name> -t excel -e <excel_filename> -s1 <sheet1_name> -s2 <sheet2_name>

# From JSON source
python finetune/backend/database.py -c <community_name> -t json -j <json_folder>
```

3. **Run Inference**
```bash
# Clone repository
git clone https://github.com/SergioRubio01/AutoAudit
cd AutoAudit

# Set up environment
poetry install
poetry shell

# Configure environment
cp .env.example .env  # Edit with your settings

# Initialize database
poetry run alembic upgrade head

# Start development server
poetry run uvicorn main:app
```

Visit `http://localhost:8000` for the web interface.


## 📚 Documentation

- [Contributing Guide](CONTRIBUTING.md)
- [API Documentation](http://localhost:8000/docs)
- [Development Plan](development_plan.md)
- [Public Python SDK](https://github.com/SergioRubio01/autoaudit-sdk-python)


## 📄 License

Proprietary - See [LICENSE](LICENSE) for details.

---

<div align="center">
Made with ❤️ by the AutoAudit Team
</div>
