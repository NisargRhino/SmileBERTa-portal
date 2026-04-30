FROM python:3.11-slim

WORKDIR /app

# System libraries required by RDKit for 2D rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only first (much smaller than the CUDA build)
RUN pip install --no-cache-dir torch==2.4.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements-render.txt .
RUN pip install --no-cache-dir -r requirements-render.txt

# Copy application code and data files
COPY . .

EXPOSE 10000

CMD ["gunicorn", "app:app", "--workers", "1", "--timeout", "120", "--bind", "0.0.0.0:10000"]
