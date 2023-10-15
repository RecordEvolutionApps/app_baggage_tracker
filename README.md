# Example App

This example app demonstrates how to publish data from a device to be received and stored later in a data pod.

You can use this app as a template to build your own IoT app.
The app is written in Python, but you can also use any other programming language.

# How To Develop

## Use Docker
Each app must have a Dockerfile called "Dockerfile" in the top folder. The Dockerfile describes the system setup your application code will run in.
Basically you use the Dockerfile to create a custom linux configuration starting from a standard base image like ubuntu or alpine linux.
Alternatively or additionally you can add Dockerfiles for specific target system architectures "Dockerfile.armv7" or "Dockerfile.arm64" if you want to make your app run on armv7 (Raspberry pi) or arm64 (Jetson) systems.
The "Dockerfile" will be used if there is no specific dockerfile present for that device's system architecture.
Refer to the [official documentation](https://docs.docker.com/engine/reference/builder/) for more information on dockerfiles.

## Editing Code
You can either edit the source code of your app online in our integrated code editor or you can use your local development environment.
To edit the source code online, simply click the "Develop" button. The online code editor that appears is your personal development environment in the cloud that only you have ever access to. It is based on the popular open source [VS-Code](https://code.visualstudio.com/) editor from Microsoft.

Code editing is based on the git workflow which is the industry standard way for creating maintainable and auditable code for collaborative development. This means every change you make to the code is only present in your personal environment until you "commit" and "push" your code to the "master" branch of your application.

You can build and run your code **on a remote device (!)** during development using the side menu buttons. You need the "Develop" privilege on the device to use it in the code editor development context.

Using the terminal in the side menu you can log into a running app container directly on a device which is often useful for debugging on a device. The terminal from the side menu is not the same as the terminal accessible via the code editor's menu:

- The **side-menu-terminal** logs into the app container on a device.
- The **code-editor-terminal** logs into your development container in the cloud that runs the code editor and contains your clone of the git repository of the app.

> The side-menu-terminal used to access a remote device does not use ssh and needs no open ports on the device.


### Interface with GitHub/GitLab/Bitbucket and your local editor
<style>
    img {
        margin: 12px;

    }
</style>
<img src="https://storage.googleapis.com/reswarm-images/GitHub-Mark-64px.png"/>
<img src="https://storage.googleapis.com/reswarm-images/gitlab.png" width="65"/>
<img src="https://storage.googleapis.com/reswarm-images/bitbucket.png" width="70"/>

Each app has a central Git repository that is used as the common code point for all developers of this app. By default this central Git repository is fully managed within the Record Evolution App Studio.
When you create an app you can choose to use another service instead. Currently the Git services from GitHub, GitLab and Bitbucket are supported.
For external Git repositories you need a "Personal Access Token" from the service to authenticate yourself. You can add these tokens in your Record Evolution User Profile.

> In the application settings you can migrate an internal repository to one of those external services at any time later.

This way you can use your own local editor setup and push to your repo at GitHub. You can still use the online editor in the Record Evolution studio as before to run and debug the code on actual devices. You can also directly release your GitHub app code to the Record Evolution App Store for deployment on swarms of devices.

## Application Parameters
As an app developer you sometimes want to expose parameters to the user of the app.
E.g. you build an app to record data from a temperature sensor and you want the user of the app to be able to change the recording frequency. In general Application Parameters can be used for simple apps where you want to give the user an easy control over some aspects of the app.

To expose application paramters to the user, you can add a file "env-template.yml" to your top folder.
The contents of that file will be rendered as a nice form for users and will be accessible at the Settings icon of the running app on the device or device group. The default values you specify in the env-template.yml will be used when running the app during development.

<img src="https://storage.googleapis.com/reswarm-images/prod/rendered_template.png" height="300" style="margin-left: auto; margin-right: auto; margin-bottom: 16px; display: block;">

The user provided parameters will be available to your application as linux environment variables.

There are some standard environment variables that are always available in your application:

* **DEVICE_NAME**            current device's name (could change)
* **DEVICE_SERIAL_NUMBER**   the unique identifier of the device (is immutable)
* **SWARM_KEY**              the unique identifier of the device's swarm (could change)

With these environment variables your app can detect which device it is running on.

> Using Application Parameters the user can specify paramters for all devices in a group at once.
> Parameters can be provided by users even if the app is not running or a device is offline.

**Examples** where Application Parameters should be used are apps that run on many devices and where you want to specify a common target S3 bucket to push the data to. Or if you want to change model parameters for an AI-App on all devices at once.

## Application User Interface
What kind of user interface you want to provide for your app depends very much on the type of app you are developing. Some apps don't need any user input, some just require a simple form (see above) and some require a sophisticated web UI.

If you want to provide a custom user interface to the users of your app, you have to include a web server into your application that serves a web frontend to users.
Since the application will run on the users edge devices, which might be street lamps or machines in a production line, the web interface will not be accessible through the internet.
To enable external access you can add a "port-template.yml" file to your top folder and specify the ports on which your web server is serving the user frontend.

<img src="https://storage.googleapis.com/reswarm-images/prod/rendered_port_template.png" height="200" style="margin-left: auto; margin-right: auto; margin-bottom: 16px; display: block;">

The user can then enable and disable web access at the settings icon of the running app on a device to access the front end of your app on that device.
(Enabling access effectively creates a secure web-tunnel that makes the user interface on the edge device accessible from the internet.)
Only users with the NETWORK privilege on a device can enable external web access.

> Although a web interface can provide very sophisticated functionality to users, settings the user makes in an interface like that are always only valid for the single device the app runs on. You can however use Application Parameters and an Application User Interface at the same time.

**Examples** where this kind of interface is applicable are robot training software or complex sensor hardware configuration software.

## The App Store
If you want to release the app to your users, you can publish the app to the central App Store.

To publish the app you need to go to the "Releases" tab of the app's settings page in the app studio.
There you need to choose a device for the publishing process. After you entered a version string in the version field and press "Publish"
the device builds and then publishes that app to your private App Store. The device you choose for the publishing process determines for which architecture the app will be published.
So if you choose an amd64 device, then the app will be available as an amd64 app in the App Store.
You can publish your app with multiple devices of different architectures to make the app available for multiple architectures at the same time.
Make sure to use the same version string for all architecture versions of your app to bundle them under one app version.
Now the app can be installed by users on their devices, but only if they have the "Use IoT App" privilege for that app.

> Your app is _not_ automatically visible in the public App Store for arbitrary users!

You can setup user privileges for your app in the "Privileges" tab of the app's settings page in the app studio.
There you can also invite other developers to collaborate with you on the app.

To make your app available in the globally public App Store you need to enable the App Store switch on the app's settings page in the app studio.

## Data volumes
If your app saves data on the device this data will normally not be retained accross app or device restarts.
To make your data persistant accross restarts, there are two folders you can use:

* **/data** A folder that is private to your app. Other apps can not access it's content.
* **/shared** A folder that is shared by all apps on the device.

## Receive and Store Data

To receive and store data from an app running on a device you can use the integrated [Data Studio](/pods). In this studio you can create a cloud database (aka "data pod") to collect data from the devices of a swarm. 

First you have to make sure the data pod owner has "Data Access" privileges for the swarm from which it should collect data. These privileges can be toggled in the swarm's settings.

Within your data pod you have to setup the swarm as a "Source" in the sources panel. When you activate the source, the data pod is actively listening for messages published from the devices of the swarm.

To send data from a device, your app can use the code modules below. To send data you need to specify a "topic" in the code. (The topic can also be given by users if you choose to expose it as an App Parameter.)
Within your data pod you now create a raw table and enter that topic in it's settings. As soon as you activate the raw table the table will receive all data sent by the configured swarm's devices on this topic.

 >We provide a python library [reswarm-py](https://pypi.org/project/reswarm/) and a node.js library [reswarm-js](https://www.npmjs.com/package/reswarm). Please refer to the given links for further documentation. If you want to use other programming languages you can use the autobahn client libraries with the WAMP message protocol.

## Hardware access
Your app may want to interact with hardware like sensors that are attached to your device.
These devices are mounted into the `/dev` directory of your application container.

# LICENSE
### MIT
Copyright (c) 2021 Record Evolution GmbH
See license file on the source code