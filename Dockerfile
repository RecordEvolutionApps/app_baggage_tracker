FROM ultralytics/ultralytics:latest-jetson

RUN apt-get -y update && apt-get -y upgrade && \
	apt-get install -y \
		libmicrohttpd-dev libjansson-dev \
		libssl-dev libsofia-sip-ua-dev libglib2.0-dev \
		libopus-dev libogg-dev libcurl4-openssl-dev liblua5.3-dev \
		libconfig-dev pkg-config libtool automake \
		libavutil-dev \
		libavformat-dev \
		libavcodec-dev \
		## libusrsctp1 \
		libwebsockets-dev \
		# libnanomsg5 \
		libnice-dev \
		##  libsrtp2-dev \
		# libnss3-dev \
		# extras
		libgtk-3-dev \
		curl \
		ffmpeg \
		nginx procps v4l-utils git \
		usbutils && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y 

# RUN curl -sL -o /etc/apt/trusted.gpg.d/morph027-janus.asc https://packaging.gitlab.io/janus/gpg.key && \
# 	. /etc/lsb-release; echo "deb https://packaging.gitlab.io/janus/$DISTRIB_CODENAME $DISTRIB_CODENAME main" | tee /etc/apt/sources.list.d/morph027-janus.list
# RUN apt-get update && apt-get install -y janus

RUN cd /tmp && \
	wget https://github.com/cisco/libsrtp/archive/v2.5.0.tar.gz && \
	tar xfv v2.5.0.tar.gz && \
	cd libsrtp-2.5.0 && \
	./configure --prefix=/usr --enable-openssl && \
	make shared_library && \
	make install

# RUN git clone https://gitlab.freedesktop.org/libnice/libnice && \
# 	cd libnice && \
# 	meson --prefix=/usr build && ninja -C build && sudo ninja -C build install
WORKDIR /usr/local/src
# RUN wget https://github.com/meetecho/janus-gateway/archive/refs/tags/v1.2.0.tar.gz && \
# 	tar xfv v1.2.0.tar.gz && \
RUN git clone --depth=1 https://github.com/meetecho/janus-gateway.git && \
    cd /usr/local/src/janus-gateway && \
	sh autogen.sh && \
	./configure --enable-post-processing --disable-rabbitmq --disable-mqtt --disable-plugin-videoroom --disable-aes-gcm --enable-libsrtp2 --prefix=/usr/local && \
	make && \
	make install && \
	make configs

WORKDIR /app

RUN wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt 

# COPY requirements.txt /app/requirements.txt
# RUN python3 -m pip install -r requirements.txt


COPY . /app
COPY janus/* /usr/local/etc/janus/

# CMD ["/usr/local/bin/janus"]

COPY nginx.conf /etc/nginx/nginx.conf
COPY web /web

# CMD ["python3", "-u", "index.py"]
# CMD ["sleep", "infinity"]
CMD ["./entrypoint.sh"]