#!/bin/bash
# Start the FastAPI backend
uvicorn flow.backend.api:app --host 0.0.0.0 --port 8000 &
# Start the Streamlit frontend
streamlit run flow/frontend/app.py --server.port 8501 --server.address 0.0.0.0 