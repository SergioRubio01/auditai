#!/bin/bash
# Fixed startup script for proper Streamlit routing

# Function to handle shutdown
cleanup() {
    echo "Shutting down services..."
    kill $API_PID $STREAMLIT_PID 2>/dev/null
    wait $API_PID $STREAMLIT_PID 2>/dev/null
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Start the FastAPI backend
echo "Starting FastAPI backend..."
uvicorn flow.backend.api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Start the Streamlit frontend with proper configuration
echo "Starting Streamlit frontend..."
streamlit run flow/frontend/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.baseUrlPath /app \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --server.enableWebsocketCompression true &
STREAMLIT_PID=$!

# Wait for both processes
echo "Both services started. Waiting..."
wait $API_PID $STREAMLIT_PID