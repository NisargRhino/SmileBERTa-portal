FROM python:3.11-slim

WORKDIR /app

# System libraries required by RDKit / Open Babel rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 \
    libxext6 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# AutoDock Vina 1.2.7 (official static binary)
ADD https://github.com/forlilab/AutoDock-Vina/releases/download/v1.2.7/vina_1.2.7_linux_x86_64 /usr/local/bin/vina
RUN chmod +x /usr/local/bin/vina

# PyTorch CPU-only first (much smaller than the CUDA build)
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements-render.txt .
RUN pip install --no-cache-dir -r requirements-render.txt

# Bake the Hugging Face models into the image so cold starts don't re-download them
ENV HF_HOME=/app/.hf
RUN python -c "from transformers import RobertaForMaskedLM, RobertaTokenizer; \
[(RobertaTokenizer.from_pretrained(m), RobertaForMaskedLM.from_pretrained(m)) for m in ('NisargRhino/drug-classification', 'NisargRhino/protein-models')]"

COPY . .

ENV PYTHONUNBUFFERED=1 MODEL_CACHE_SIZE=1 OMP_NUM_THREADS=1
EXPOSE 10000

# One worker (jobs + model cache live in-process), threads so polling works during docking
CMD ["gunicorn", "app:app", "--workers", "1", "--worker-class", "gthread", "--threads", "4", "--timeout", "300", "--bind", "0.0.0.0:10000"]
