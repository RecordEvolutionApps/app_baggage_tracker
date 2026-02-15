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

# Register L4T's pre-installed packages with pip/uv so they are never overwritten.
# The L4T base image ships CUDA-accelerated OpenCV, NumPy, and PyTorch — pip must
# not replace them with CPU-only rebuilds.
RUN python3 -c "\
import site, os, cv2, numpy;\
sp = site.getsitepackages()[0];\
pkgs = [\
  ('opencv-python', 'opencv_python', cv2.__version__),\
  ('opencv-python-headless', 'opencv_python_headless', cv2.__version__),\
  ('opencv-contrib-python', 'opencv_contrib_python', cv2.__version__),\
  ('numpy', 'numpy', numpy.__version__),\
];\
[(\
  os.makedirs(d:=os.path.join(sp, f'{dist}-{ver}.dist-info'), exist_ok=True),\
  open(os.path.join(d,'METADATA'),'w').write(f'Metadata-Version: 2.1\nName: {name}\nVersion: {ver}\n'),\
  open(os.path.join(d,'top_level.txt'),'w').write(name.split('-')[0]+'\n'),\
  open(os.path.join(d,'RECORD'),'w').close(),\
  open(os.path.join(d,'INSTALLER'),'w').write('system\n'),\
) for name,dist,ver in pkgs]"

# Install MMDetection stack first (slow, but cached)
# TORCH_CUDA_ARCH_LIST tells nvcc which GPU architectures to compile for.
# Jetson Xavier = 7.2, Jetson Orin = 8.7.  Without this, mmcv CUDA ops
# (like NMS) may silently skip the Jetson GPU and fail at runtime with
# "nms_impl: implementation for device cuda:0 not found".
ENV TORCH_CUDA_ARCH_LIST="7.2;8.7"
ENV FORCE_CUDA="1"
RUN uv pip install --system openmim==0.3.9 && \
	mim install mmcv==2.1.0

RUN uv pip install --system mmengine==0.10.4 mmdet==3.3.0

# Install remaining dependencies (changes more frequently)
RUN uv pip install --system ironflock==1.0.6 matplotlib==3.7.5 pyyaml==6.0.1 defusedxml==0.7.1 \
				pillow==10.3.0 lapx==0.5.5 yt_dlp==2024.4.9 \
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
# libgomp: required for OpenMP (numpy, torch)
# libGLdispatch: must be preloaded to avoid "cannot allocate memory in static TLS block"
#                which prevents GStreamer nvvidconv/nvvideo4linux2 plugins from loading
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1:/lib/aarch64-linux-gnu/libGLdispatch.so.0

EXPOSE 8000

ENTRYPOINT ["/app/video/entrypoint.sh"]