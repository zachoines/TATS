#!/bin/bash
echo "Installing OpenCV 4.4.0 on your Jetson Nano"
echo "It will take 2 hours !"

# reveal the CUDA location
cd ~
sudo sh -c "echo '/usr/local/cuda/lib64' >> /etc/ld.so.conf.d/nvidia-tegra.conf"
sudo ldconfig

# install the dependencies
sudo apt-get install -y build-essential cmake git unzip pkg-config
sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install -y libgtk2.0-dev libcanberra-gtk*
sudo apt-get install -y python3-dev python3-numpy python3-pip
sudo apt-get install -y libxvidcore-dev libx264-dev libgtk-3-dev
sudo apt-get install -y libtbb2 libtbb-dev libdc1394-22-dev
sudo apt-get install -y gstreamer1.0-tools libv4l-dev v4l-utils
sudo apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt-get install -y libavresample-dev libvorbis-dev libxine2-dev
sudo apt-get install -y libfaac-dev libmp3lame-dev libtheora-dev
sudo apt-get install -y libopencore-amrnb-dev libopencore-amrwb-dev
sudo apt-get install -y libopenblas-dev libatlas-base-dev libblas-dev
sudo apt-get install -y liblapack-dev libeigen3-dev gfortran
sudo apt-get install -y libhdf5-dev protobuf-compiler
sudo apt-get install -y libprotobuf-dev libgoogle-glog-dev libgflags-dev

# remove old versions or previous builds
cd ~ 
sudo rm -rf opencv*
# download version 4.4.0
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.4.0.zip 
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.4.0.zip 
# unpack
unzip opencv.zip 
unzip opencv_contrib.zip 
# some administration to make live easier later on
mv opencv-4.4.0 opencv
mv opencv_contrib-4.4.0 opencv_contrib
# clean up the zip files
rm opencv.zip
rm opencv_contrib.zip

# set install dir
cd ~/opencv
mkdir build
cd build

# run cmake (The python3 include and lib dirs may need adjusting)
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr \
-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
-D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
-D WITH_OPENCL=OFF \
-D WITH_CUDA=ON \
-D PYTHON3_EXECUTABLE=/usr/bin/python3 \
-D PYTHON3_LIBRARY= /usr/lib/python3.6/config-3.6m-aarch64-linux-gnu/libpython3.6m.so \
-D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/lib/python3/dist-packages/numpy/core/include/ \
-D CUDA_ARCH_BIN=7.2 \
-D CUDA_ARCH_PTX="" \
-D WITH_CUDNN=ON \
-D WITH_CUBLAS=ON \
-D ENABLE_FAST_MATH=ON \
-D CUDA_FAST_MATH=ON \
-D OPENCV_DNN_CUDA=ON \
-D ENABLE_NEON=ON \
-D WITH_QT=OFF \
-D WITH_OPENMP=ON \
-D WITH_OPENGL=ON \
-D BUILD_TIFF=ON \
-D WITH_FFMPEG=ON \
-D WITH_GSTREAMER=ON \
-D WITH_TBB=ON \
-D BUILD_TBB=ON \
-D BUILD_TESTS=OFF \
-D WITH_EIGEN=ON \
-D WITH_V4L=ON \
-D WITH_LIBV4L=ON \
-D OPENCV_ENABLE_NONFREE=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D BUILD_NEW_PYTHON_SUPPORT=ON \
-D BUILD_opencv_python3=TRUE \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D BUILD_EXAMPLES=OFF ..

# run make
FREE_MEM="$(free -m | awk '/^Swap/ {print $2}')"
# Use "-j 4" only swap space is larger than 3.5GB
if [[ "FREE_MEM" -gt "3500" ]]; then
  NO_JOB=4
else
  NO_JOB=1
fi
make -j ${NO_JOB} 

# sudo rm -r /usr/include/opencv4/opencv2
sudo make install
sudo ldconfig

# cleaning (frees 300 MB)
make clean
sudo apt-get update
