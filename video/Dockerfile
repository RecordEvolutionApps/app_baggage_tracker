FROM ultralytics/ultralytics:8.2.31-jetson-jetpack5

RUN apt-get -y update && \
	apt-get install -y \
		# libmicrohttpd-dev libjansson-dev \
		# libssl-dev libsofia-sip-ua-dev libglib2.0-dev \
		# libopus-dev libogg-dev libcurl4-openssl-dev liblua5.3-dev \
		# libconfig-dev pkg-config libtool automake \
		# libavutil-dev \
		# libavformat-dev \
		# libavcodec-dev \
		# libwebsockets-dev \
		# meson \
		# libgtk-3-dev \
		curl \
		ffmpeg \
		openssh-server \
		unzip nginx procps v4l-utils git \
		usbutils udev && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

WORKDIR /app/video

COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt

# needs to be done seperately cus you can't install with no deps in requirements.txt
RUN python3 -m pip install supervision --no-deps


RUN mkdir -p /app/download

RUN rm -rf /usr/src/ultralytics/ultralytics/assets/*

COPY patch/polygon_zone.py /usr/local/lib/python3.8/dist-packages/supervision/detection/tools/polygon_zone.py

COPY . /app/video

RUN echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config

ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0

ENTRYPOINT ["/app/video/entrypoint.sh"]