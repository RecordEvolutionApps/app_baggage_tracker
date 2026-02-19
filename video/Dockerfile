# ============================================================================
# Unified multi-architecture Dockerfile (arm64 / Jetson + amd64 / CPU dev)
# ============================================================================
# Docker automatically sets TARGETARCH to "arm64" or "amd64".
# We define both base images as named stages and select the right one.
# Only the selected base ends up in the final image; the other is pulled
# once and cached but adds zero bytes to the output.
# ============================================================================

# ── Arch-specific base images ────────────────────────────────────────────────
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3 AS base-arm64
FROM python:3.12-slim-bookworm                      AS base-amd64

# ── Select the correct base based on TARGETARCH ─────────────────────────────
ARG TARGETARCH
FROM base-${TARGETARCH}

ARG TARGETARCH

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:0.4.20 /uv /usr/local/bin/uv

# ── System dependencies ─────────────────────────────────────────────────────
# Common packages for both architectures
RUN apt-get -y update && \
	apt-get install -y \
		curl \
		ffmpeg \
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

# AMD64 needs extra packages: GStreamer dev libs, cmake, ffmpeg dev headers
# (on ARM64 these come pre-installed or aren't needed)
RUN if [ "$TARGETARCH" = "amd64" ]; then \
		apt-get -y update && \
		apt-get install -y \
			cmake \
			pkg-config \
			libavcodec-dev \
			libavformat-dev \
			libavutil-dev \
			libswscale-dev \
			libswresample-dev \
			libgstreamer1.0-dev \
			libgstreamer-plugins-base1.0-dev \
			gstreamer1.0-plugins-base \
			gstreamer1.0-plugins-good \
			gstreamer1.0-plugins-bad \
			gstreamer1.0-plugins-ugly \
			gstreamer1.0-libav \
			gstreamer1.0-tools \
			libgstrtspserver-1.0-dev && \
		apt-get clean && \
		rm -rf /var/lib/apt/lists/*; \
	fi

WORKDIR /app/video

COPY requirements.txt requirements.txt

# ── ARM64 (Jetson): register L4T pre-installed packages so pip won't replace them
RUN if [ "$TARGETARCH" = "arm64" ]; then \
		python3 -c "\
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
) for name,dist,ver in pkgs]"; \
	fi

# ── AMD64: install PyTorch CPU from pip (ARM64 has it pre-installed in L4T)
# Use pip (not uv) here because uv cannot resolve the +cpu wheel variants
# from PyTorch's custom index.
RUN if [ "$TARGETARCH" = "amd64" ]; then \
		pip install torch==2.6.0 torchvision==0.21.0 \
			--index-url https://download.pytorch.org/whl/cpu; \
	fi

# ── ML framework + Python dependencies (architecture-specific) ──────────────
# Single uv resolution pass per arch to avoid uv dropping transitive deps
# across separate install steps.
# ARM64 (Jetson): keep MMDetection + add compatible HF Transformers
# AMD64: drop MMDetection entirely, go full HF Transformers (modern stack)
RUN if [ "$TARGETARCH" = "arm64" ]; then \
		export TORCH_CUDA_ARCH_LIST="7.2;8.7" FORCE_CUDA=1 && \
		uv pip install --system openmim==0.3.9 && \
		mim install mmcv==2.1.0 && \
		uv pip install --system mmengine==0.10.4 mmdet==3.3.0 mmpretrain==1.2.0 && \
		uv pip install --system \
			"transformers==4.45.2" "huggingface_hub>=0.25,<1" "safetensors>=0.4" \
			ironflock==1.0.6 matplotlib==3.7.5 pyyaml==6.0.1 \
			defusedxml==0.7.1 pillow==10.3.0 lapx==0.5.5 yt_dlp==2024.4.9 \
			fastapi==0.115.0 "uvicorn[standard]==0.30.6" \
			onnx onnxruntime; \
	else \
		uv pip install --system \
			"transformers==5.2.0" "huggingface_hub>=1.3,<2" \
			"safetensors>=0.5" "timm>=1.0.11" \
			ironflock==1.0.6 matplotlib==3.8.4 pyyaml==6.0.1 \
			defusedxml==0.7.1 pillow==10.3.0 lapx==0.5.5 yt_dlp==2024.4.9 \
			numpy==1.26.4 fastapi==0.115.0 "uvicorn[standard]==0.30.6" \
			onnx onnxruntime; \
	fi

# ── AMD64: Build OpenCV from source with GStreamer + Python bindings ─────────
# The pip opencv wheels have no GStreamer support. We need GStreamer to push
# H.264 RTP frames to mediasoup. Explicitly provide Python paths so cmake
# generates the cv2.so Python extension (the slim base image puts Python in
# /usr/local which cmake's FindPython3 doesn't search by default).
RUN if [ "$TARGETARCH" = "amd64" ]; then \
		pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null; true && \
		PYTHON_INC=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))") && \
		PYTHON_LIB=$(find /usr/local/lib -name 'libpython3*.so.1.0' | head -1) && \
		NUMPY_INC=$(python3 -c "import numpy; print(numpy.get_include())") && \
		PYTHON_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])") && \
		export PKG_CONFIG_PATH="/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/share/pkgconfig" && \
		cd /tmp && \
		git clone --depth 1 --branch 4.8.1 https://github.com/opencv/opencv.git && \
		mkdir opencv/build && cd opencv/build && \
		cmake -D CMAKE_BUILD_TYPE=RELEASE \
			-D CMAKE_INSTALL_PREFIX=/usr/local \
			-D WITH_GSTREAMER=ON \
			-D WITH_FFMPEG=ON \
			-D BUILD_opencv_python3=ON \
			-D BUILD_opencv_python2=OFF \
			-D PYTHON3_EXECUTABLE=$(which python3) \
			-D PYTHON3_INCLUDE_DIR=${PYTHON_INC} \
			-D PYTHON3_LIBRARY=${PYTHON_LIB} \
			-D PYTHON3_NUMPY_INCLUDE_DIRS=${NUMPY_INC} \
			-D PYTHON3_PACKAGES_PATH=${PYTHON_SITE} \
			-D BUILD_TESTS=OFF \
			-D BUILD_PERF_TESTS=OFF \
			-D BUILD_EXAMPLES=OFF \
			.. 2>&1 | tee /tmp/cmake_out.txt && \
		grep -i "gstreamer" /tmp/cmake_out.txt && \
		grep -q "GStreamer.*YES" /tmp/cmake_out.txt || \
			{ echo "ERROR: GStreamer not detected by cmake — aborting"; exit 1; } && \
		make -j$(nproc) && \
		make install && \
		ldconfig && \
		python3 -c "import cv2; info=cv2.getBuildInformation(); gst=[l for l in info.split('\n') if 'GStreamer' in l]; print('GStreamer lines:', gst); assert any('YES' in l for l in gst), 'GStreamer not enabled in cv2'" && \
		rm -rf /tmp/opencv /tmp/cmake_out.txt; \
	fi

# ── AMD64: patch cv2 Python wrappers ────────────────────────────────────────
# Two issues arise when gapi is absent or partially compiled:
# 1. cv2/gapi/__init__.py line 323 unconditionally assigns gapi_wip_gst_GStreamerPipeline
# 2. cv2/typing/__init__.py references cv2.GMat/GOpaqueT/GArrayT at module level
# Both cause AttributeError on `import cv2`.  Guard them with hasattr/try-except.
COPY patch/patch_cv2_wrappers.py /tmp/patch_cv2_wrappers.py
RUN if [ "$TARGETARCH" = "amd64" ]; then python3 /tmp/patch_cv2_wrappers.py; fi

# Install supervision separately (version compatibility)
# ARM64 stays on 0.25.1 (last version supporting Python 3.8)
# AMD64 gets latest with from_transformers segmentation support
RUN if [ "$TARGETARCH" = "arm64" ]; then \
		uv pip install --system supervision==0.25.1 --no-deps; \
	else \
		uv pip install --system supervision==0.27.0; \
	fi

# ── yt-dlp standalone binary (arch-specific URL) ────────────────────────────
RUN if [ "$TARGETARCH" = "arm64" ]; then \
		curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_linux_aarch64 \
			-o /usr/local/bin/yt-dlp; \
	else \
		curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp \
			-o /usr/local/bin/yt-dlp; \
	fi && \
	chmod +x /usr/local/bin/yt-dlp

# Create model cache directory
RUN mkdir -p /app/download /data/mmdet /data/huggingface

# Copy patch file and apply before copying all source
COPY patch/polygon_zone.py /app/video/patch/polygon_zone.py
RUN SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])") && \
    cp /app/video/patch/polygon_zone.py ${SITE_PACKAGES}/supervision/detection/tools/polygon_zone.py

# Copy application code (changes frequently — keep last)
COPY . /app/video

# Make entrypoint executable
RUN chmod +x /app/video/entrypoint.sh

# ── ARM64 (Jetson): CUDA/OpenGL environment ─────────────────────────────────
# libgomp: required for OpenMP (numpy, torch)
# libGLdispatch: must be preloaded to avoid "cannot allocate memory in static TLS block"
#                which prevents GStreamer nvvidconv/nvvideo4linux2 plugins from loading
# Write an env file sourced by entrypoint.sh so LD_PRELOAD only applies on Jetson.
RUN if [ "$TARGETARCH" = "arm64" ]; then \
		echo 'LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1:/lib/aarch64-linux-gnu/libGLdispatch.so.0' \
			> /etc/video-env.conf; \
	else \
		touch /etc/video-env.conf; \
	fi

EXPOSE 8000

ENTRYPOINT ["/app/video/entrypoint.sh"]
