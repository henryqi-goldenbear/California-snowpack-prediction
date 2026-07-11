# California Snowpack Prediction Dockerfile
# Multi-stage build for smaller final image

# Build stage
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.9-slim

WORKDIR /app

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir /home/appuser/app && \
    chown appuser:appuser /home/appuser/app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Make sure scripts in .local are usable
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy application code
COPY . /home/appuser/app
WORKDIR /home/appuser/app

# Set environment variables
ENV PYTHONPATH=/home/appuser/app
ENV MISTRAL_API_KEY=""

# Expose port for potential API service
EXPOSE 8000

# Set default command
ENTRYPOINT ["python"]
CMD ["main.py"]

# Alternative: Run as a service
# CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
