# Utilizamos la imagen oficial de RunPod con PyTorch y CUDA optimizado para H100
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Configurar variables de entorno para evitar interacciones durante la instalación
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Directorio de trabajo del contenedor
WORKDIR /app

# Instalar dependencias del sistema operativo (por si las necesitamos para Pillow o manejo de imágenes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar el archivo de requerimientos primero para cachear la capa de dependencias
COPY requirements.txt .

# Instalar dependencias de Python (diffusers, runpod, torch, boto3, etc.)
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar el código fuente del Worker
COPY handler.py .

# Comando de inicio: RunPod Serverless ejecuta el handler directamente
CMD ["python", "-u", "handler.py"]
