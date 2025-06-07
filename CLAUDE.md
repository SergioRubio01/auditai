# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoAudit is a multi-agent document processing system for ICEX payments developed by Grant Thornton. It uses vision-language models with fine-tuning capabilities to process and analyze payment documents.

## Essential Commands

### Running the Application

```bash
# Using Docker (recommended)
docker-compose up

# Local development
uvicorn flow.backend.api:app --host 0.0.0.0 --port 8000  # Backend
streamlit run flow/frontend/app.py                       # Frontend
```

### Document Processing

```bash
# Process PDFs to images
python finetune/backend/pdf2png.py -c <community_name> -cn <excel_column_name> -t "Transferencia emitida" "Detalle movimiento" "Solicitud de transferencia"

# Create dataset from Excel
python finetune/backend/database.py -c <community_name> -t excel -e <excel_filename> -s1 <sheet1_name> -s2 <sheet2_name>

# Create dataset from JSON
python finetune/backend/database.py -c <community_name> -t json -j <json_folder>
```

### Environment Setup

```bash
# Create Conda environment
conda create --name icex_payments python=3.11 pytorch-cuda=12.1 pytorch cudatoolkit ipywidgets -c pytorch -c nvidia -y
conda activate icex_payments

# Install dependencies
pip install -r requirements.txt
```

## High-Level Architecture

### Multi-Agent System Architecture

The system uses LangGraph to orchestrate specialized agents:

1. **Document Classification Agent**: Determines document type (transfers, credit cards, invoices, payrolls)
2. **Retrieval Agents**: Extract specific information based on document type
3. **Table Format Agents**: Process tabular data using AWS Textract
4. **Post-processing Agents**: Validate and store data in SQLite database

### Workflow Types

- **Pagos (Payments)**: Bank transfers and credit card statements
- **Facturas (Invoices)**: Invoice data with tax information
- **Nominas (Payrolls)**: Payroll documents
- **Matching**: Document matching operations

### Key Components

- **Backend API** (`flow/backend/api.py`): FastAPI server handling uploads and batch processing
- **Workflow Manager** (`flow/backend/workflowmanager.py`): Orchestrates document workflows
- **Processor** (`flow/backend/processor.py`): Manages individual image processing
- **Frontend** (`flow/frontend/app.py`): Streamlit web interface

### Processing Flow

1. Documents uploaded via API or web interface
2. PDFs converted to images (if needed)
3. Images encoded to base64
4. Document type classification
5. Routing to specialized workflow
6. Sequential agent processing
7. Data storage in SQLite
8. Results export to Excel

### Environment Variables

Required `.env` file:
```
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key
```

### Key Dependencies

- **LangGraph**: Workflow orchestration
- **LangChain**: LLM integration (Groq, OpenAI, Ollama)
- **AWS Textract**: OCR and table extraction
- **FastAPI**: Backend API framework
- **Streamlit**: Frontend framework
- **SQLite**: Data persistence

### Important Notes

- Multiple LLM models are used for different tasks (llm4, llm5, llm7, etc.)
- Rate limiting with exponential backoff is implemented
- Batch processing uses multiprocessing for concurrent operations
- Error handling includes retry logic for API calls