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
FROM python:3.10-slim-bookworm                      AS base-amd64

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
		pip install torch==2.1.2 torchvision==0.16.2 \
			--index-url https://download.pytorch.org/whl/cpu; \
	fi

# ── MMDetection stack (both arches, but ARM64 needs CUDA compile flags) ─────
# TORCH_CUDA_ARCH_LIST and FORCE_CUDA are set inline for the mmcv build step
# so they don't pollute the runtime environment on amd64.
RUN if [ "$TARGETARCH" = "arm64" ]; then \
		export TORCH_CUDA_ARCH_LIST="7.2;8.7" FORCE_CUDA=1; \
	fi && \
	uv pip install --system openmim==0.3.9 && \
	mim install mmcv==2.1.0 && \
	uv pip install --system mmengine==0.10.4 mmdet==3.3.0 mmpretrain==1.2.0

# ── Remaining Python dependencies ───────────────────────────────────────────
RUN if [ "$TARGETARCH" = "arm64" ]; then \
		uv pip install --system ironflock==1.0.6 matplotlib==3.7.5 pyyaml==6.0.1 \
			defusedxml==0.7.1 pillow==10.3.0 lapx==0.5.5 yt_dlp==2024.4.9 \
			fastapi==0.115.0 "uvicorn[standard]==0.30.6" \
			onnx onnxruntime; \
	else \
		uv pip install --system ironflock==1.0.6 matplotlib==3.8.4 pyyaml==6.0.1 \
			defusedxml==0.7.1 pillow==10.3.0 lapx==0.5.5 yt_dlp==2024.4.9 \
			numpy==1.26.4 fastapi==0.115.0 "uvicorn[standard]==0.30.6"; \
	fi

# ── AMD64: Build OpenCV from source with GStreamer support ───────────────────
RUN if [ "$TARGETARCH" = "amd64" ]; then \
		pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null; true && \
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
			-D BUILD_TESTS=OFF \
			-D BUILD_PERF_TESTS=OFF \
			-D BUILD_EXAMPLES=OFF \
			-D BUILD_LIST=core,imgproc,imgcodecs,videoio,video,highgui,python3 \
			.. && \
		make -j$(nproc) && \
		make install && \
		ldconfig && \
		rm -rf /tmp/opencv; \
	fi

# Install supervision separately (version compatibility)
RUN uv pip install --system supervision==0.25.1 --no-deps

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
RUN mkdir -p /app/download /data/mmdet

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
