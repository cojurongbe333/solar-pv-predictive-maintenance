# Dockerfile for FastAPI ML Model Deployment

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy API script and model folder
COPY api/ ./api/
COPY models/ ./models/

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn joblib numpy pydantic

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
