# TATS
Target acquisition and tracking system. The aim of this repo is to use various deep learning techniques for detecting and tracking targets, optimized for IOT systems like the RPI 4 and NVIDEA Jetson product lines.

## Requirements
* libtorch 1.6.0
* libboost 1.61.0
* OpenCV 4.4.0

## Technologies
* Target detection done via a choice of cascade models, RCNN models, or Yolo5 models.
* Target tracking performed by OpenCV 4 C++ API. 
* Optimized PID's for servo control.
* Advance PID autotuning via Soft Actor Critic Reinforcement learning.
* Performance optimized via asynchronous multithreading, and multiprocessing techniques.

## Recommended Hardware
* [Raspberry Pi 4 Model B](https://www.raspberrypi.org/products/raspberry-pi-4-model-b/) or [JETSON XAVIER NX](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-xavier-nx/)
* [PI Camera Module V2.1](https://www.raspberrypi.org/products/camera-module-v2/)
* 2x High speed servos. Ideally, those with metal gear train and low amperage.
* Pan-tilt servo mounts, [like these](https://www.servocity.com/pan-tilt-kits/).

## Installation
* Navigate to your choice of build directory
    * cd /home/{username}
* Make Python3 Env
	* sudo apt-get install python3-venv
	  sudo python3 -m venv env
	  source env/bin/activate 
* OpenCV 4 c++
	* [Here](https://cv-tricks.com/installation/opencv-4-1-ubuntu18-04/) is an excellent walkthrough of the process.

* Pytorch c++
	* Install from source
		* git clone http://github.com/pytorch/pytorch
		* cd pytorch
		* git submodule update --init
		* sudo pip3 install -U setuptools
		* sudo pip3 install -r requirements.txt
		* git submodule update --init --recursive
		* sudo python3 setup.py develop
        * python3 setup.py install

* lib Boost
	* Sudo apt-get install libboost
	* Include dir via linker command: -I/usr/include/boost

* Build TATS
    * cmake -DCMAKE_PREFIX_PATH={path to build dir}/pytorch/build/lib.linux-aarch64-3.6/
        * Its also possible to use python installation of Pytorch for the build as well
    * make TATS
* Run install
    * export LD_LIBRARY_PATH=/home/{username}/pytorch/build/lib:$LD_LIBRARY_PATH 
    * ./TATS

