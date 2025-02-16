# ICEX Document Processing System

A scalable document processing system for financial documents using FastAPI, PostgreSQL, and AI-powered analysis.

## Features

- 🔍 Multi-document type support (Payments, Invoices, Payroll)
- 🤖 AI-powered data extraction with vision-language models
- 📊 Automated Excel report generation
- 🔄 PDF to PNG conversion pipeline
- 🚀 Distributed processing with background tasks
- 🔒 Rate limiting and security features

## Architecture

The system consists of three main components:

1. **FastAPI Backend** (`/flow/backend`)
   - RESTful API endpoints
   - PostgreSQL database integration
   - Background task processing
   - Rate limiting middleware

2. **Streamlit Frontend** (`/flow/frontend`)
   - Document upload interface
   - Processing status monitoring
   - Results visualization
   - Excel export functionality

3. **AI Processing Pipeline**
   - Vision-language model integration
   - Multi-agent workflow system
   - Distributed task processing

## Quick Start

1. Clone the repository and create environment:
```bash
conda create --name icex_payments python=3.11
conda activate icex_payments
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# .env
OPENAI_API_KEY=your_key
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=eu-west-1
TEXTRACT_CONFIDENCE_THRESHOLD=0.7
DB_URL=postgresql://user:password@localhost:5432/icex
IMAGE_INPUT_DIR=./uploads/Images
```

4. Fine-tune your models:
   - Create your own fine-tuned models using OpenAI API or other supported providers
   - Update the model names in `flow/backend/utils/llm.py`:
```python
# Example for OpenAI fine-tuned models
llm1 = ChatOpenAI(model="ft:gpt-4-vision-preview:your-org:your-model-name:version")
llm4 = ChatOpenAI(model="ft:gpt-4:your-org:your-model-2:version")
# ... update other model names accordingly
```

5. Run with Docker:
```bash
docker-compose up --build
```

6. Start the application:
```bash
python -m uvicorn flow.backend.api:app --reload --host 0.0.0.0
```

## API Documentation

Access the interactive API documentation at:
- OpenAPI: `http://localhost:8000/docs`

## Project Structure

```
├── flow/
│   ├── backend/
│   │   ├── api.py          # FastAPI application
│   │   ├── models/         # Pydantic models
│   │   ├── routes/         # API endpoints
│   │   └── utils/          # Helper functions
│   └── frontend/
│       └── app.py          # Streamlit interface
├── docker-compose.yml
└── requirements.txt
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

Apache License 2.0 - See LICENSE file for details.

## Contact

For support: user@contact.com