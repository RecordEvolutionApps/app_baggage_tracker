#!/bin/bash

for dev in `ls /sys/class/video4linux -1`
do
    v4l2-ctl --list-formats --device /dev/$dev | \
    grep -qE '\[[0-9]\]' && \
    echo /dev/$dev:`cat /sys/class/video4linux/$dev/name | cut -d ":" -f 1`:`udevadm info --query=property /dev/$dev | grep DEVPATH= | awk -F '=' '{print $NF}'`
done

exit 0