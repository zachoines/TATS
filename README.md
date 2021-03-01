# TATS
Target acquisition and tracking system. The aim of this repo is to use various deep learning techniques for detecting and tracking targets on servo driven pan/tilt camera systems. Optimized for linux IOT devices like the Raspberry Pi and NVIDIA Jetson product lines.

[![TATS Technical Overview](http://img.youtube.com/vi/5aenaehoWtQ/0.jpg)](http://www.youtube.com/watch?v=5aenaehoWtQ "TATS Technical Overview")

## Requirements
* libtorch 1.7.0
* libboost 1.61.0
* OpenCV 4.4.0

## Technology Highlights
* Target detection 
	* Choice of cascade models, RCNN models, or Yolo5 models.
* Advance PID autotuning 
	* Soft Actor Critic Reinforcement learning optimizes gains
* Predictive object tracking
	* AI model predicts potential next locations for temporarily occluded objects 

## Benefits
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
	* sudo apt-get install python3-venv
	* sudo python3 -m venv env
	* source env/bin/activate 
* OpenCV 4 c++
	* [Here](https://cv-tricks.com/installation/opencv-4-1-ubuntu18-04/) is an excellent walkthrough of the process.

* Pytorch c++
	* Install from source
		* git clone http://github.com/pytorch/pytorch
		* cd pytorch
		* sudo pip3 install -U setuptools
		* sudo pip3 install -r requirements.txt
		* git submodule update --init --recursive
		* Set args (For Jetson)
			* export USE_NCCL=0
			* export DEBUG=1
			* export USE_DISTRIBUTED=0      
			* export USE_QNNPACK=0
			* export USE_CUDA=1
			* export USE_PYTORCH_QNNPACK=0
			* export TORCH_CUDA_ARCH_LIST="5.3;6.2;7.2"
			* export PYTORCH_BUILD_VERSION=1.7.1
			* export PYTORCH_BUILD_NUMBER=0
			* export NO_QNNPACK=1
			* export BUILD_TEST=0
		* sudo python3 setup.py develop
        * sudo python3 setup.py install
	* Can also use Python3 libs to build TATS
		* On jetson platform, install pytorch from (Nvidia Jetson Wheels)[https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-7-0-now-available/72048]

* lib Boost
	* Sudo apt-get install libboost

* Build TATS
	* cmake . 
		* May need to set CMAKE_PREFIX_PATH for libtorch in CMakeLists.txt
    * make TATS
    
* Run install
    * May need to set search path for libtorch if there are linker errors when running
		* From source - export LD_LIBRARY_PATH={base path here}/pytorch/build/lib:$LD_LIBRARY_PATH 
		* Python libs - export LD_LIBRARY_PATH={base path here}/python3.6/site-packages/torch/:$LD_LIBRARY_PATH 
    * ./TATS

