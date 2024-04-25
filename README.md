# Automotive traffic - selective object detection
Interactive camera application for selective object detection. Real-time object counting is available in user-defined areas of the video image. The application can differentiate between vehicles such as trucks, cars, buses, and motorcycles.

Users can interact with the live video and select areas of interest. Object detection and counting will be isolated to the detection zones specified by the user. Multiple areas can be defined, and the number of objects will be displayed for each zone.

## Dashboarding Insights
<div style="display:flex;flex-direction:row;align-items:center;justify-content:center;">
    <img src="https://storage.googleapis.com/reswarm-images/dashboards/dashboard_devices_camera.png" width="600px">
</div>

Besides the user interface for interacting with the video stream, the app also provides a dashboard. Collected data, such as objects detected and the current video frame, will be displayed. A history of detections is persisted, providing a summary in different graphs.

The detection data history is collected in the cloud storage backend of the app. Each swarm automatically uses its own private cloud data storage. This ensures that the app dashboard in a swarm can only access and show the data of that swarm. This is how clients data separation is reliably and securely enforced.

*The video snapshot that is shown in the dashboard is recorded historically with only one frame per second while the live stream has a higher frame rate and lower latency.*

## Live footage from the edge

The app additionally provides a web interface that allows users to view the low latency live video stream via WebRTC and also allows users to configure the camera on the individual edge device.

If multiple cameras are connected to the device, the camera can be selected via the web interface.

<div style="display:flex;flex-direction:row;align-items:center;justify-content:center;">
    <img src="https://res.cloudinary.com/dotw7ar1m/image/upload/v1714052452/APPmockup.png" width="600px">
</div>

## Available Environment Variables

#### Change the model: ####
 - Select a version for one of the YOLO Object Detection Models

#### Detection Classes: ####

- A list of class ids from the COCO dataset. If left empty, then all classes are used. See https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml

#### Confidence (0-1): ####
 - Sets the minimum confidence threshold for detections. Objects detected with confidence below this threshold will be disregarded. Adjusting this value can help reduce false positives.

#### Intersection over Union (0-1): ####
 - Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS). Lower values result in fewer detections by eliminating overlapping boxes, useful for reducing duplicates.

#### Width (1-10000): ####
 - Horizontal Resolution in pixel

#### Height (1-10000): ####
- Vertical Resolution in pixel



## Requirements

This app requires NVIDIA Jetson Xavier and Orin systems.

Any USB or IP camera connected to the Nvidia IPC can be used with the app. 