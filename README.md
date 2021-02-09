# TATS
Target acquisition and tracking system. The aim of this repo is to use various deep learning techniques for detecting and tracking targets. Optimized for linux IOT systems like the Raspberry Pi and NVIDIA Jetson product lines.

## Requirements
* libtorch 1.7.0
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
		* Set args
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
	* Or just install (Nvidia Jetson Wheels)[https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-7-0-now-available/72048] if local development is not intended.

* lib Boost
	* Sudo apt-get install libboost

* Build TATS
	* cmake . 
		* Make sure to set CMAKE_PREFIX_PATH for libtorch in CMakeLists.txt
        * Use python installation of Pytorch by setting CMake to "{python libs}/python3.6/site-packages/torch/".
    * make TATS
    
* Run install
    * May need to set lib search path for pytorch
		* From source - export LD_LIBRARY_PATH={base path here}/pytorch/build/lib:$LD_LIBRARY_PATH 
		* Python libs - export LD_LIBRARY_PATH={base path here}/python3.6/site-packages/torch/:$LD_LIBRARY_PATH 
    * ./TATS

