# Traffic - Object Detection
Simple Object Detection example with Ultralytics YOLOv8.
The results are filtered to cars, buses, trucks and motorcycles.

The detection data history is collected in the cloud storage backend of the app. Each swarm automatically uses its own private cloud data storage. This ensures that the app dashboard in a swarm can only access and show the data of that swarm.

This is how clients data separation is reliably and securely enforced.

## Dashboarding
<div style="display:flex;flex-direction:row;align-items:center;justify-content:center;">
    <img src="https://storage.googleapis.com/reswarm-images/dashboards/dashboard_devices_camera.png" width="600px">
</div>

The video snapshot that is shown in the dashboard is recorded historically with only one frame per second while the live stream has a higher frame rate and lower latency.

## Live edge footage

The app additionally provides a web interface that allows users to view the low latency live video stream via WebRTC and also allows users to configure the camera on the individual edge device.

If multiple cameras are connected to the device, the camera can be selected in the web interface.

<div style="display:flex;flex-direction:row;align-items:center;justify-content:center;">
    <img src="https://storage.googleapis.com/reswarm-images/PastedGraphic-1.png" width="600px">
</div>

## Requirements

This app requires NVIDIA Jetson Xavier and Orin systems.

Any USB or IP camera connected to the Nvidia IPC can be used with the app. 