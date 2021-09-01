# TATS
Target acquisition and tracking system. The aim of this repo is to use various deep learning techniques for detecting and tracking targets on servo driven pan/tilt camera systems. Optimized for linux IOT devices like the Raspberry Pi and NVIDIA Jetson product lines.

[![TATS Technical Overview](http://img.youtube.com/vi/5aenaehoWtQ/0.jpg)](http://www.youtube.com/watch?v=5aenaehoWtQ "TATS Technical Overview")

## Requirements
* libtorch 1.7.0
* libboost 1.61.0
* OpenCV 4.4.0

## Highlights
* Target detection 
	* Choice of ARUCO, cascade, RCNN, or Yolo5 models.
* Advance PID autotuning 
	* Soft Actor Critic Reinforcement learning optimizes PID gains for fast response rates.
* Predictive object tracking
	* AI model predicts future locations when objects are occluded temporarily.
* Dynamically responds to different movement characteristics (constant speed vs. accelerating objects), avoiding oscillations and overshooting common in traditional PID systems.
* Performance optimized with low level C++ API's
* Highly flexable
	* Can utilize PIDS, Autotuned PIDS, or an AI to calculate angle vectors.
	* Use of different machine vision models.
	* Can be configured to different resolution/speed cameras and servos.

## Recommended Hardware
* [Raspberry Pi 4 Model B](https://www.raspberrypi.org/products/raspberry-pi-4-model-b/) or [JETSON XAVIER NX](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-xavier-nx/)
* [PI Camera Module V2.1](https://www.raspberrypi.org/products/camera-module-v2/)
* 2x High speed servos. Ideally, those with metal gear train and quick reponse rate.
* Pan-tilt servo mounts [like these](https://www.servocity.com/pan-tilt-kits/).

## Installation
* Navigate to choice of build directory
    * cd /home/{username}

* Make Python3 Env
	* sudo -H apt-get install python3-venv
	* sudo python3 -m venv env
	* source env/bin/activate 

* Pytorch 1.7.0 & OpenCV 4.4
	* ./install-opencv.sh
	* ./install-pytorch.sh	

* lib Boost
	* sudo apt install libboost-dev=1.65.*

* Unzip models
	* tar -xvzf yolov5SCoco640.tar.gz (tar -zcvf "filename" to compress)
	* tar -xvzf haarModel.tar.gz

* Build and Run TATS
	* Update the CMakeLists.txt with desired example program from ./examples
	* cmake . 
		* May need to set CMAKE_PREFIX_PATH for libtorch in CMakeLists.txt to its Python3 library locatiion
    * make TATS
    