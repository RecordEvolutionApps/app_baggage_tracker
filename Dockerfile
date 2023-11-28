FROM ultralytics/ultralytics:8.0.211-jetson

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
		unzip nginx procps v4l-utils git \
		usbutils udev && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y 

RUN cd /tmp && \
	wget https://github.com/cisco/libsrtp/archive/v2.5.0.tar.gz && \
	tar xfv v2.5.0.tar.gz && \
	cd libsrtp-2.5.0 && \
	./configure --prefix=/usr --enable-openssl && \
	make shared_library && \
	make install

WORKDIR /usr/local/src

RUN git clone --depth=1 https://github.com/meetecho/janus-gateway.git && \
    cd /usr/local/src/janus-gateway && \
	sh autogen.sh && \
	./configure --enable-post-processing --disable-rabbitmq --disable-mqtt --disable-plugin-videoroom --disable-aes-gcm --enable-libsrtp2 --prefix=/usr/local && \
	make && \
	make install && \
	make configs

WORKDIR /app

RUN wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt 

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install -r requirements.txt

RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/root/.bun/bin:${PATH}"
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash \
	&& . /root/.bashrc && nvm install 18.0.0

# For live developing code in your running container install the vscode cli and start a tunnel with `./code tunnel` in the /app folder
# RUN curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-arm64' --output vscode_cli.tar.gz &&\
# 	tar -xf vscode_cli.tar.gz

COPY backend /app/backend
RUN cd backend && bun i --frozen-lockfile --production && bun run build

COPY web /app/web
RUN cd web && . /root/.bashrc && bun i && bun run build

COPY janus/* /usr/local/etc/janus/
COPY entrypoint.sh env-template.yml port-template.yml /app/

CMD ["./entrypoint.sh"]