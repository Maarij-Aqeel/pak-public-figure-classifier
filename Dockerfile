FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/root/.cache/torch

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    libgl1 ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

RUN python -c "import torchvision.models as m; \
    m.resnet50(weights='IMAGENET1K_V2'); \
    m.efficientnet_b3(weights='IMAGENET1K_V1')" || true
RUN python -c "from facenet_pytorch import MTCNN, InceptionResnetV1; \
    MTCNN(); InceptionResnetV1(pretrained='vggface2')" || true

COPY . .

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -fs http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
