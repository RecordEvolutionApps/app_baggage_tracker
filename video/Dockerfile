FROM ultralytics/ultralytics:8.2.2-jetson

RUN apt-get -y update && apt-get -y upgrade && \
	apt-get install -y \
		libmicrohttpd-dev libjansson-dev \
		libssl-dev libsofia-sip-ua-dev libglib2.0-dev \
		libopus-dev libogg-dev libcurl4-openssl-dev liblua5.3-dev \
		libconfig-dev pkg-config libtool automake \
		libavutil-dev \
		libavformat-dev \
		libavcodec-dev \
		libwebsockets-dev \
		meson \
		libgtk-3-dev \
		curl \
		ffmpeg \
		openssh-server \
		unzip nginx procps v4l-utils git \
		usbutils udev && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

RUN mkdir -p /data/ssh && mkdir -p ~/.ssh
RUN ssh-keygen -t rsa -b 4096 -f /data/ssh/mykey -N "" -O exist && \
    cp -a /data/ssh/* ~/.ssh

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install -r requirements.txt

# needs to be done seperately cus you can't install with no deps in requirements.txt
RUN python3 -m pip install supervision --no-deps

WORKDIR /app

RUN mkdir -p /app/download

RUN rm -rf /usr/src/ultralytics/ultralytics/assets/*

COPY patch/polygon_zone.py /usr/local/lib/python3.8/dist-packages/supervision/detection/tools/polygon_zone.py

COPY . /app/video

CMD ["/usr/sbin/sshd", "-D"]