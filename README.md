<div style="display:flex;flex-direction:row;align-items:center;justify-content:center;">
    <img src="https://res.cloudinary.com/dotw7ar1m/image/upload/v1717493714/baggaeAppBanner.png" width="650px">
</div>

<br>

# End-To-End Smart Baggage Tracking System with custom detection zones

Interactive camera application for selective luggage detection. Real-time object counting is available in user-defined areas of the video image.

Users can interact with the live video and select areas of interest. The detection and counting will be isolated to the detection zones specified by the user. Multiple areas can be defined, and the number of objects will be displayed for each zone.

## Hardware

The Revis box is an End-to-End solution that can be configured for your needs. This means that even though parts can be exchanged, we provide a turnkey solution. While we rely on NVIDIA devices, the camera itself can be selected depending on the use case. This has several advantages such as extending existing cameras with AI functionality or selecting a special camera, for instance, night vision.


## Use on your camera

To use the app on your own devices:

1. Register an NVIDIA Jetson PC in your device fleet on IronFlock
2. Install this app on the device.
3. Enable remote access in the app settings to access the app's home page for camera configuration and live stream.

To create a multi camera setup, just repeat the steps above with as many devices as you like.

The detection data of all cameras in your swarm will be available in the app dashboard.

> There is a demo video included in the app. You can use this video by choosing IP Camera in the app configuration and then using `demoVideo` as the url.

## Dashboarding Insights

<div style="display:flex;flex-direction:row;align-items:center;justify-content:center;">
    <img src="https://res.cloudinary.com/dotw7ar1m/image/upload/v1717494096/baggageAppScreenshotDashboard.png" width="600px">
</div>
<br>

This app provides a dashboard summarizing detection data from all swarm cameras into a common view. Collected data histories, such as the number of detected objects per class and the current video frame, will be displayed in the dashboard. The history of detections is persisted, providing a summary in different graphs.

The detection data history is collected in the cloud storage backend of the app. Each swarm automatically uses its own private cloud data storage. This ensures that the app dashboard in a swarm can only access and show the data of that swarm. This is how client data separation is reliably and securely enforced.

> The video snapshot that is shown in the dashboard is recorded historically with only one frame per second while the live stream (see below) has a higher frame rate and lower latency.

## Live footage from the edge

The app additionally provides a web interface on each individual camera that allows users to view the low latency live video stream from that camera. This interface also allows users to configure the video and detection model for the camera.

<div style="display:flex;flex-direction:row;align-items:center;justify-content:center;">
    <img src="https://res.cloudinary.com/dotw7ar1m/image/upload/v1717494386/tunnelBaggageApp.png" width="600px">
</div>

## App Parameters

Each camera can also be configured by external parameters in the app settings on the device. These settings can be given on device or group level for the whole swarm at the same time.

Parameter | Default | Description
-|-|-
Model | Yolov8s | Select a version for one of the YOLO Object Detection Models
Detection Classes | 24, 26, 28 | A list of class ids from the COCO dataset. If left empty, then all classes are used. See [here](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)
Smoothing | false | Smoothing uses previous frames to determine the average detection box of an object in the current frame.
USE_SAHI | false | SAHI is a method to slice a high resolution frame into chunks of 640x640. This improves the detection quality for small objects on high resolution images, but may slow down inference time.
Confidence | 0.1 | (0 - 1) Sets the minimum confidence threshold for detections. Objects detected with confidence below this threshold will be disregarded. Adjusting this value can help reduce false positives.
Intersection over Union | 0.7 | (0 - 1) Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS). Lower values result in fewer detections by eliminating overlapping boxes, useful for reducing duplicates.
Width | 1280 | Horizontal Resolution in pixel of the camera stream. The camera should support the selected resolution.
Height | 720 | Vertical Resolution in pixel of the camera stream. The camera should support the selected resolution.

## Requirements

This app can currently only be used NVIDIA Jetson systems.

Any USB or IP camera connected to the Nvidia PC can be used with the app.
