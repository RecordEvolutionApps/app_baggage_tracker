#!/bin/bash

CAM_PATH="/dev/video$CAM_NUM"

echo "Camera Path: $CAM_PATH"

v4l2-ctl --list-formats-ext -d $CAM_PATH

# CAM=$(v4l2-ctl --list-devices | grep -m 1 -o '/dev/video[0-9]*')
# echo $CAM
nvgpu_out=$(lsmod | grep nvgpu)

# if [ -n "$nvgpu_out" ]; then
    # echo "NVIDIA GPU available!"
    # nohup gst-launch-1.0 v4l2src device=$CAM_PATH ! videorate ! \
    #     image/jpeg,format=I420,width=${RESOLUTION_X},height=${RESOLUTION_Y},framerate=${FRAMERATE}/1 ! \
    #     nvv4l2decoder mjpeg=1 ! \
    #     nvv4l2h265enc maxperf-enable=1 ! h265parse ! rtph265pay ! \
    #     udpsink host=127.0.0.1 port=5006 &
# else
    # echo "NVIDIA GPU NOT available!"
    # nohup gst-launch-1.0 v4l2src device=/dev/video0 !  \
    #     "image/jpeg,format=I420,width=${RESOLUTION_X},height=${RESOLUTION_Y}" ! \
    #     jpegparse ! jpegdec ! \
    #     clockoverlay time-format="%D %H:%M:%S" !\
    #     vp8enc deadline=2 threads=2 keyframe-max-dist=60 ! video/x-vp8 ! rtpvp8pay ! \
    #     udpsink host=127.0.0.1 port=5004 &
# fi

# nohup python3 -u index.py &

# nohup gst-launch-1.0 v4l2src device=$CAM ! \
#     'image/jpeg,framerate=30/1' ! \
#     jpegparse ! jpegdec ! \
#     clockoverlay time-format="%D %H:%M:%S" ! \
#     vp8enc deadline=2 threads=2 keyframe-max-dist=60 ! video/x-vp8 ! rtpvp8pay ! \
#     udpsink host=127.0.0.1 port=5004 &

# nohup ./code tunnel --accept-server-license-terms &

bun backend/src/index.ts &

/usr/local/bin/janus --daemon

# sleep infinity
exec python3 -u /app/backend/src/index.py


