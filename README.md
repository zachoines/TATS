# TATS
Target acquisition and tracking system. The aim of this repo is to use various deep learning techniques for detecting and tracking targets on servo driven pan/tilt camera systems. Optimized for linux IOT devices like the Raspberry Pi and NVIDIA Jetson product lines.

[![TATS Technical Overview](http://img.youtube.com/vi/5aenaehoWtQ/0.jpg)](http://www.youtube.com/watch?v=5aenaehoWtQ "TATS Technical Overview")

## Requirements
* libtorch 1.7.0
* libboost 1.61.0
* OpenCV 4.4.0

## Highlights
* Target detection 
	* Choice of cascade models, RCNN models, or Yolo5 models.
* Advance PID autotuning 
	* Soft Actor Critic Reinforcement learning optimizes PID gains for fast response rates.
* Predictive object tracking
	* AI model predicts fiture locations for occluded objects 
* Dynamically responds to first order and second order movement characteristics (constant speed vs. accelerating objects)
* Avoids setpoint oscillation common in traditional PID systems
* Performance optimized with fast C++ api's, asynchronous multithreading, and multiprocessing.
* Highly flexable
	* Can utilize plain PIDS, Autotuned PIDS, or an AI to calculate angles 
	* Use of RCNN, Cascade, or Yolo5 machine vision models. Plan to add more in future.
	* Can be configured to different resoultuion/speed cameras and servos

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
	* tar -xvzf yoloModel.tar.gz
	* tar -xvzf haarModel.tar.gz

* Build and Run TATS
	* cmake . 
		* May need to set CMAKE_PREFIX_PATH for libtorch in CMakeLists.txt to its Python3 library locatiion
    * make TATS
    