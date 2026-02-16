#!/bin/bash
#
# Enumerate local cameras (USB, MIPI CSI, GMSL) with supported resolutions.
# Output format (one line per device):
#   /dev/videoN:Name:devpath:WxH,WxH,...:interface
#
# Interface types detected:
#   usb   — USB Video Class (UVC) cameras
#   csi   — MIPI CSI cameras (Jetson ISP / tegra-capture-vi)
#   gmsl  — GMSL cameras via MAX9296/MAX96712 deserializers
#   other — Anything else (virtual, PCI, etc.)
#
# The resolution list is gathered via v4l2-ctl --list-framesizes for each
# pixel format the device supports.  Duplicates are removed.

detect_interface() {
    local devpath="$1"
    local name="$2"
    # GMSL deserializers (Maxim MAX9296, MAX96712, etc.)
    if echo "$devpath" | grep -qiE 'max9296|max96712|gmsl'; then
        echo "gmsl"
    # MIPI CSI on Jetson (tegra-capture-vi, tegra-video, nvcsi)
    elif echo "$devpath" | grep -qiE 'tegra-capture-vi|tegra-video|nvcsi|i2c.*imx|i2c.*ov[0-9]'; then
        echo "csi"
    # Also detect CSI by driver/name patterns
    elif echo "$name" | grep -qiE 'imx[0-9]|ov[0-9]|ar[0-9]|tegra'; then
        echo "csi"
    # USB (most common path: usb/ or UVC in name)
    elif echo "$devpath" | grep -qiE 'usb'; then
        echo "usb"
    else
        echo "other"
    fi
}

for dev in $(ls /sys/class/video4linux -1); do
    # Only include devices that actually advertise at least one video format
    v4l2-ctl --list-formats --device /dev/$dev 2>/dev/null | \
        grep -qE '\[[0-9]\]' || continue

    path="/dev/$dev"
    name=$(cat /sys/class/video4linux/$dev/name 2>/dev/null | cut -d ":" -f 1)
    devpath=$(udevadm info --query=property /dev/$dev 2>/dev/null | grep DEVPATH= | awk -F '=' '{print $NF}')

    # Detect camera interface type
    interface=$(detect_interface "$devpath" "$name")

    # Collect supported resolutions across all pixel formats
    resolutions=""
    for pixfmt in $(v4l2-ctl --list-formats --device "$path" 2>/dev/null \
                    | grep -oP "'\K[A-Z0-9]+(?=')" | sort -u); do
        sizes=$(v4l2-ctl --list-framesizes "$pixfmt" --device "$path" 2>/dev/null \
                | grep -oP '\d+x\d+' | sort -u)
        for s in $sizes; do
            resolutions="${resolutions:+$resolutions,}$s"
        done
    done

    # De-duplicate and sort by width descending
    if [ -n "$resolutions" ]; then
        resolutions=$(echo "$resolutions" | tr ',' '\n' | sort -t 'x' -k1 -n -r -u | paste -sd ',')
    fi

    echo "${path}:${name}:${devpath}:${resolutions}:${interface}"
done

exit 0