import cv2
import json
import pyudev


context = pyudev.Context()

result = []

for device in context.list_devices(subsystem='video4linux'):
    device_path = device.device_node
    cap = cv2.VideoCapture(device_path)
    #
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #
        # Get the USB device name using pyudev
        usb_device_name = device.get('ID_MODEL', 'Unknown')
        usb_serial_number = device.get('ID_SERIAL_SHORT', 'Unknown')
    #
    else:
        # If unable to open, retrieve available information
        width = 0
        height = 0
        usb_device_name = device.get('ID_MODEL', 'Unknown')
        usb_serial_number = device.get('ID_SERIAL_SHORT', 'Unknown')
    
    if width > 0:
        result.append({
            "path": device_path,
            "name": usb_device_name,
            "serial": usb_serial_number,
            "width": width,
            "height": height
        })
    #
    cap.release()

result = [{"path": '/dev/video0'}, {"path": '/dev/video1'}, {"path": '/dev/video2'}, {"path": '/dev/video3'}]

# Print the result as JSON
print(json.dumps(result))
