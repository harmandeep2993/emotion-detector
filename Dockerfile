# base image — slim Python to keep image size small
FROM python:3.12-slim

# set working directory inside container
WORKDIR /app

# install system dependencies OpenCV needs
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libgles2 \
    libegl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install uv
RUN pip install uv

# copy dependency files first
# Docker caches this layer — only reinstalls if dependencies change
COPY pyproject.toml .
COPY uv.lock .

# install dependencies
RUN uv pip install --system --no-cache \
    fastapi \
    uvicorn \
    python-multipart \
    pillow \
    opencv-python-headless \
    torch \
    torchvision \
    hsemotion \
    "timm==0.9.2" \
    numpy \
    mediapipe

# copy project files
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

# expose port
EXPOSE 8000

# start FastAPI with uvicorn
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]