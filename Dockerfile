FROM python:3.13-slim

WORKDIR /app

# Instalar herramientas de compilación necesarias para sentencepiece y tokenizers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Instalar PyTorch CPU-only PRIMERO (antes de otras dependencias)
# Esto evita que dependencias instalen PyTorch completo
RUN pip install --no-cache-dir torch>=2.5.0 --index-url https://download.pytorch.org/whl/cpu

# Copiar requirements.txt
COPY requirements.txt .

# Instalar otras dependencias (usarán PyTorch CPU ya instalado)
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código esencial
COPY src/ /app/src/
COPY Modules/ /app/Modules/
COPY app.py /app/

# Exponer puerto
EXPOSE 5000

# Variables de entorno (se configuran en Railway)
ENV PORT=5000

# Comando de inicio
CMD ["python", "app.py"]
