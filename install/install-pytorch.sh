#!/bin/bash
echo "Installing PyTorch 1.7.0 on your Jetson Nano"

wget -O torch-1.7.0-cp36-cp36m-linux_aarch64.whl https://nvidia.box.com/shared/static/cs3xn3td6sfgtene6jdvsxlr366m2dhq.whl
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev 
pip3 -H install Cython
pip3 -H install numpy torch-1.8.0-cp36-cp36m-linux_aarch64.whl
rm torch-1.7.0-cp36-cp36m-linux_aarch64.whl