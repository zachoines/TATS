// C++ libs
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

// Pytorch imports
#include <torch/torch.h>
#include <stdio.h>

// OpenCV imports
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv4/opencv2/dnn/dnn.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/video/tracking.hpp"
#include <opencv2/ximgproc/segmentation.hpp>

// Local imports
#include "./src/detection/CascadeDetector.h"
#include "./src/wire/Wire.h"
#include "./src/servo/PCA9685.h"
#include "./src/util/util.h"

#define SERVOMIN  150 // This is the 'minimum' pulse length count (out of 4096)
#define SERVOMAX  600 // This is the 'maximum' pulse length count (out of 4096)
#define USMIN  600 // This is the rounded 'minimum' microsecond length based on the minimum pulse of 150
#define USMAX  2400 // This is the rounded 'maximum' microsecond length based on the maximum pulse of 600
#define SERVO_FREQ 50 // Analog servos run at ~50 Hz updates

void setServoPulse(uint8_t n, double pulse, PCA9685 pwm) {
  double pulselength;
  
  pulselength = 1000000;   // 1,000,000 us per second
  pulselength /= SERVO_FREQ;   // Analog servos run at ~60 Hz updates
  pulselength /= 4096;  // 12 bits of resolution
  pulse *= 1000000;  // convert input seconds to us
  pulse /= pulselength;
  pwm.setPWM(n, 0, pulse);
}

int main(int argc, char** argv)
{

    torch::Device cpu(torch::kCPU);
    torch::Device cuda(torch::kCUDA);

    if (torch::cuda::is_available()) {
		std::cout << "CUDA is available! Training on GPU." << std::endl;
	}
	else {
		std::cout << "CUDA is not available! Training on CPU." << std::endl;
	}
    Wire* wire = new Wire();
    const uint8_t base_address = 0x40;
    PCA9685 pwm(base_address, wire);
    pwm.begin();

    uint8_t servonum = 1;
    pwm.setOscillatorFrequency(27000000);
    pwm.setPWMFreq(SERVO_FREQ);  // Analog servos run at ~50 Hz updates

    for (int angle = -70; angle < 70; angle += 5) {
        int pulselen = Utility::angleToTicks(angle);
        pwm.setPWM(servonum, 0, pulselen);
        wire->delay(500);
    }

    wire->delay(500);
    for (uint16_t pulselen = SERVOMAX; pulselen > SERVOMIN; pulselen+=5) {
        pwm.setPWM(servonum, 0, pulselen);
        wire->delay(500);
    }

    // Drive each servo one at a time using writeMicroseconds(), it's not precise due to calculation rounding!
    // The writeMicroseconds() function is used to mimic the Arduino Servo library writeMicroseconds() behavior. 
    // for (uint16_t microsec = USMIN; microsec < USMAX; microsec += 5) {
    //     pwm.writeMicroseconds(servonum, microsec);
    //     wire->delay(500);
    // }

    // wire->delay(500);
    // for (uint16_t microsec = USMAX; microsec > USMIN; microsec -= 5) {
    //     pwm.writeMicroseconds(servonum, microsec);
    //     wire->delay(500);
    // }
    
    // using namespace cv;
    // using namespace cv::ximgproc::segmentation;
    // Ptr<SelectiveSearchSegmentation> ss = createSelectiveSearchSegmentation();
    // torch::Tensor A = torch::ones(1000, cuda);
    // torch::Tensor B = torch::ones(1000, cuda);
    // torch::Tensor C = A + B;

}

/*
    //Build command
    g++ -o test *.cpp
    -std=gnu++17
    -Wl,--no-as-needed
    -g
    -I/home/zachoines/Documents/pytorch/build/lib.linux-aarch64-3.6/torch/include 
    -I/home/zachoines/Documents/pytorch/build/lib.linux-aarch64-3.6/torch/include/torch/csrc/api/include
    -L/home/zachoines/Documents/pytorch/build/lib.linux-aarch64-3.6/torch/lib
    -lc10_cuda -lc10 -ltorch -ltorch_cuda -ltorch_cpu

    // Export Libs
    export LD_LIBRARY_PATH=/home/zachoines/Documents/pytorch/build/lib.linux-aarch64-3.6/torch/lib:$LD_LIBRARY_PATH 
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH 
    export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH 

    // Include dirs
    -lc10_cuda -lc10 -ltorch -ltorch_cuda -ltorch_cpu
    -I/home/zachoines/Documents/pytorch/build/lib.linux-aarch64-3.6/torch
    -I/home/zachoines/Documents/pytorch/build/lib.linux-aarch64-3.6/torch/include 
    -I/home/zachoines/Documents/pytorch/build/lib.linux-aarch64-3.6/torch/include/torch
    -I/home/zachoines/Documents/pytorch/build/lib.linux-aarch64-3.6/torch/include/torch/csrc 
    -I/home/zachoines/Documents/pytorch/build/lib.linux-aarch64-3.6/torch/include/torch/csrc/api/include
    -I/home/zachoines/Documents/pytorch/build/lib.linux-aarch64-3.6/torch/include/torch/csrc/api/include/torch

    // Linkage dirs
    -L/usr/local/cuda/lib64
    -L/usr/local/cuda-10.2/lib64
    -L/home/zachoines/Documents/pytorch/build/lib
    -L/home/zachoines/Documents/pytorch/build/lib.linux-aarch64-3.6/torch/lib
/*


/* 
    // Linking against  python3 intall

    // export commands
    export LD_LIBRARY_PATH=/usr/local/lib/python3.6/dist-packages/torch/lib:$LD_LIBRARY_PATH 
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH 
    export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH 

    // Include dirs
    /usr/local/lib/python3.6/dist-packages/torch/include
    /usr/local/lib/python3.6/dist-packages/torch/include/torch/csrc
    /usr/local/lib/python3.6/dist-packages/torch/include/torch/csrc/api/include/torch/

    /usr/local/cuda/include
    /usr/local/cuda-10.2/include

    // linkage dirs
    /usr/local/lib/python3.6/dist-packages/torch/lib
    /usr/local/cuda/lib64
    /usr/local/cuda-10.2/lib64

    -Wl,--no-as-needed

    // Full command
    g++ -o test *.cpp
    -std=gnu++14
    -Wl,--no-as-needed
    -I/usr/local/cuda-10.2/include
    -I/usr/local/lib/python3.6/dist-packages/torch/include
    -I/usr/local/lib/python3.6/dist-packages/torch/include/torch
    -I/usr/local/lib/python3.6/dist-packages/torch/include/torch/csrc
    -I/usr/local/lib/python3.6/dist-packages/torch/include/torch/csrc/api/include
    -I/usr/local/lib/python3.6/dist-packages/torch/include/torch/csrc/api/include/torch
    -L/usr/local/cuda-10.2/lib64
    -L/usr/local/lib/python3.6/dist-packages/torch/lib
    -lc10_cuda -lc10 -ltorch -ltorch_cuda -ltorch_cpu
    

*/


// g++ -o test /home/zachoines/Documents/repos/test/pytorch/test.cpp -std=gnu++17 -Wl,--no-as-needed -g -I/home/zachoines/Documents/pytorch/build/lib.linux-aarch64-3.6/torch/include -I/home/zachoines/Documents/pytorch/build/lib.linux-aarch64-3.6/torch/include/torch/csrc/api/include -L/home/zachoines/Documents/pytorch/build/lib.linux-aarch64-3.6/torch/lib -lc10_cuda -lc10 -ltorch -ltorch_cuda -ltorch_cpu
// cmake -DCMAKE_PREFIX_PATH=/home/zachoines/Documents/pytorch/build/lib.linux-aarch64-3.6/
// make TATS