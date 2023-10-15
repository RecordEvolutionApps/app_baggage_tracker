FROM ultralytics/ultralytics:latest-jetson

RUN apt-get update && apt-get install -y \
    libgtk-3-dev

WORKDIR /app

RUN wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt 

COPY . /app

# RUN python3 -m pip install reswarm==0.0.16

CMD ["python3", "-u", "index.py"]
# CMD ["sleep", "infinity"]