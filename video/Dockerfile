FROM ultralytics/ultralytics:8.2.2-jetson

RUN apt-get -y update && apt-get -y upgrade && \
	apt-get install -y \
		curl \
		ffmpeg \
		unzip nginx procps v4l-utils git \
		usbutils udev && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

RUN mkdir -p /data/ssh
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

CMD ["sleep", "infinity"]