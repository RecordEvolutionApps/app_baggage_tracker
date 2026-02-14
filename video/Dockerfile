# Multi-platform Dockerfile for Jetson AGX (CUDA/TensorRT) and CPU-only dev
ARG BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

FROM ${BASE_IMAGE}

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:0.4.20 /uv /usr/local/bin/uv

# Install system dependencies
RUN apt-get -y update && \
	apt-get install -y \
		curl \
		ffmpeg \
		unzip \
		procps \
		v4l-utils \
		git \
		usbutils \
		udev \
		libgl1 \
		libglib2.0-0 \
		libgomp1 \
		libsm6 \
		libxext6 \
		libxrender-dev \
		build-essential && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

WORKDIR /app/video

# Install Python dependencies (split into layers for better caching)
COPY requirements.txt requirements.txt

# Install MMDetection stack first (slow, but cached)
RUN uv pip install --system openmim==0.3.9 && \
	mim install mmcv==2.1.0

RUN uv pip install --system mmengine==0.10.4 mmdet==3.3.0

# Install remaining dependencies (changes more frequently)
RUN uv pip install --system ironflock==1.0.6 matplotlib==3.7.5 pyyaml==6.0.1 defusedxml==0.7.1 \
				pillow==10.3.0 lapx==0.5.5 yt_dlp==2024.4.9 opencv-python==4.8.1.78 numpy==1.24.4 \
				fastapi==0.115.0 "uvicorn[standard]==0.30.6"

# Install supervision separately (version compatibility)
RUN uv pip install --system supervision==0.25.1 --no-deps

# Create model cache directory
RUN mkdir -p /app/download /data/mmdet

# Copy patch file and apply before copying all source
COPY patch/polygon_zone.py /app/video/patch/polygon_zone.py
RUN SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])") && \
    cp /app/video/patch/polygon_zone.py ${SITE_PACKAGES}/supervision/detection/tools/polygon_zone.py

# Copy application code (changes frequently — keep last)
COPY . /app/video

# Make entrypoint executable
RUN chmod +x /app/video/entrypoint.sh

# Environment for CUDA/OpenGL on Jetson
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

EXPOSE 8000

ENTRYPOINT ["/app/video/entrypoint.sh"]