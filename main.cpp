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
#include "./src/detection/RCNNDetector.h"
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


    // Test Racoon Detector
    std::string path = get_current_dir_name();
    std::string saved_model_file = "/models/rcnn/jit_raccoon_detector.pt";
	std::string saved_model_file_path = path + saved_model_file;
    Detect::RCNNDetector rd(saved_model_file_path);
    rd.setTargetLabel(0);

    std::string imageFile = "/images/raccoon-1.jpg";
    std::string imageFilePath = path + imageFile;
    cv::Mat input_image = cv::imread(imageFilePath, cv::IMREAD_UNCHANGED);
    Detect::DetectionData result = rd.detect(input_image);


    // Test Servos
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

    wire->delay(500);
    for (uint16_t microsec = USMIN ; microsec < USMAX; microsec += 5) {
        pwm.writeMicroseconds(servonum, microsec);
        wire->delay(50);
    }

    wire->delay(500);
    for (uint16_t microsec = USMAX; microsec > USMIN; microsec -= 5) {
        pwm.writeMicroseconds(servonum, microsec);
        wire->delay(50);
    }

    // Bring back to zero
    wire->delay(500);
    int pulselen = Utility::angleToTicks(0.0);
    pwm.setPWM(servonum, 0, pulselen);

}

// g++ -o test /home/zachoines/Documents/repos/test/pytorch/test.cpp -std=gnu++17 -Wl,--no-as-needed -g -I/home/zachoines/Documents/pytorch/build/lib.linux-aarch64-3.6/torch/include -I/home/zachoines/Documents/pytorch/build/lib.linux-aarch64-3.6/torch/include/torch/csrc/api/include -L/home/zachoines/Documents/pytorch/build/lib.linux-aarch64-3.6/torch/lib -lc10_cuda -lc10 -ltorch -ltorch_cuda -ltorch_cpu
// cmake -DCMAKE_PREFIX_PATH=/home/zachoines/Documents/pytorch/build/lib.linux-aarch64-3.6/
// make TATS