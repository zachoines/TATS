// C++ libs
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

// Pytorch imports
#include <torch/torch.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <chrono>

// OpenCV imports
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/video/tracking.hpp"

// Local imports
#include "./src/detection/CascadeDetector.h"
#include "./src/detection/RCNNDetector.h"
#include "./src/detection/YoloDetector.h"
#include "./src/wire/Wire.h"
#include "./src/servo/PCA9685.h"
#include "./src/servo/ServoKit.h"
#include "./src/util/util.h"

int main(int argc, char** argv)
{

    // Setup Torch defaults
    // auto default_dtype = caffe2::TypeMeta::Make<float>();
	// torch::set_default_dtype(default_dtype);

    // Camera related initializations
    int capture_width = 1280 ;
    int capture_height = 720 ;
    int display_width = 1280 ;
    int display_height = 720 ;
    int framerate = 60 ;
    int flip_method = 0 ;
    std::string pipeline = Utility::gstreamer_pipeline(capture_width, capture_height, display_width, display_height, framerate, flip_method);

    // Setup I2C and PWM
    control::Wire* wire = new control::Wire();
    control::PCA9685* pwm = new control::PCA9685(0x40, wire);
    control::ServoKit* servos = new control::ServoKit(pwm);

    servos->initServo({
        .servoNum = 0,
        .minAngle = -65.0,
        .maxAngle = 65.0,
        .minMs = 0.9,
        .maxMs = 2.1,
        .resetAngle = 0.0
    });

    servos->initServo({
        .servoNum = 1,
        .minAngle = -65.0,
        .maxAngle = 65.0,
        .minMs = 0.9,
        .maxMs = 2.1,
        .resetAngle = 0.0
    });

    servos->setAngle(0, 0.0);
    servos->setAngle(1, 0.0);

    // Test The servos
    for (int angle = -70; angle < 70; angle += 5) {
        servos->setAngle(0, angle);
        servos->setAngle(1, angle);
        wire->delay(250);
    }
    servos->setAngle(0, 0.0);
    servos->setAngle(1, 0.0);
    
    // Check for CUDA
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        device_type = torch::kCUDA;
        std::cout << "CUDA is available! Training on GPU." << std::endl;
    } else {
        device_type = torch::kCPU;
        std::cout << "CUDA is not available! Training on CPU." << std::endl;
    }

    // Setup Camera
    cv::VideoCapture camera(pipeline, cv::CAP_GSTREAMER);
    if (!camera.isOpened())
	{
		throw std::runtime_error("cannot initialize camera");
	} else {
        if (Utility::GetImageFromCamera(camera).empty()) {
            throw std::runtime_error("Issue reading frame!");
        }
    }

    // Uno class names
    std::vector<std::string> class_names = {
        "0", "1", "10", "11", "12", "13", "14", "2", "3", "4", "5", "6", "7", "8", "9"
    };

    // load the network
    std::string path = get_current_dir_name();
	std::string weights = path + "/models/yolo/yolo_uno.torchscript.pt";
    Detect::YoloDetector detector = Detect::YoloDetector(weights, class_names);

    // Detect loop
    while (true) {
        cv::Mat input_image = Utility::GetImageFromCamera(camera);
        if (input_image.empty()) {
            std::cout << "Issue getting frame from camera!!" << std::endl;
            continue;
        }

        auto results = detector.detect(input_image, true, 1);
        cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
        cv::imshow("Result", input_image);
        cv::waitKey(1);
    }

    cv::destroyAllWindows();
    
    // Test Servos
    // for (int angle = -70; angle < 70; angle += 5) {
    //     int pulselen = Utility::angleToTicks(angle);
    //     pwm.setPWM(servonum, 0, pulselen);
    //     wire->delay(500);
    // }

    // wire->delay(500);
    // for (uint16_t pulselen = SERVOMAX; pulselen > SERVOMIN; pulselen+=5) {
    //     pwm.setPWM(servonum, 0, pulselen);
    //     wire->delay(500);
    // }

    // wire->delay(500);
    // for (uint16_t microsec = USMIN ; microsec < USMAX; microsec += 5) {
    //     pwm.writeMicroseconds(servonum, microsec);
    //     wire->delay(50);
    // }

    // wire->delay(500);
    // for (uint16_t microsec = USMAX; microsec > USMIN; microsec -= 5) {
    //     pwm.writeMicroseconds(servonum, microsec);
    //     wire->delay(50);
    // }

    // // Bring back to zero
    // wire->delay(500);
    // int pulselen = Utility::angleToTicks(0.0);
    // pwm.setPWM(servonum, 0, pulselen); 

}

// g++ -o test /home/zachoines/Documents/repos/test/pytorch/test.cpp -std=gnu++17 -Wl,--no-as-needed -g -I/home/zachoines/Documents/pytorch/build/lib.linux-aarch64-3.6/torch/include -I/home/zachoines/Documents/pytorch/build/lib.linux-aarch64-3.6/torch/include/torch/csrc/api/include -L/home/zachoines/Documents/pytorch/build/lib.linux-aarch64-3.6/torch/lib -lc10_cuda -lc10 -ltorch -ltorch_cuda -ltorch_cpu
// cmake -DCMAKE_PREFIX_PATH=/home/zachoines/Documents/pytorch/build/lib.linux-aarch64-3.6/
// make TATS

/*
    torch::Device cpu(torch::kCPU);
    torch::Device cuda(torch::kCUDA);

    // Set up Tensorflow DNN
    std::string path = get_current_dir_name();
    std::string frozen_model_pbtxt = path + "/models/frozen_models/FacesMotorbikesairplanesModel.pbtxt";
	std::string frozen_model_pb = path + "/models/frozen_models/FacesMotorbikesairplanesModel.pb";
    std::vector<std::vector<cv::Mat>> outputblobs;
    cv::Mat display_image, input_image;
    cv::dnn::Net tensorflowDetector = cv::dnn::readNetFromTensorflow(frozen_model_pb);
    tensorflowDetector.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    tensorflowDetector.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    // One time initialization of DNN
    std::string imageFile = "/images/faces/image_0001.jpg";
    std::string imageFilePath = path + imageFile;
    input_image = cv::imread(imageFilePath, cv::IMREAD_UNCHANGED);
    input_image.convertTo(input_image, CV_32F, 1 / 255.0);
    tensorflowDetector.setInput(cv::dnn::blobFromImage(input_image, 1.0, cv::Size(224, 224), 0.0, false, false, CV_32F));
    tensorflowDetector.forward(outputblobs, {"functional_1/box_output/Sigmoid", "functional_1/label_output/Softmax"}); 

    outputblobs.clear();
    input_image = Utility::GetImageFromCamera(camera);
    display_image = input_image.clone();
    input_image.convertTo(input_image, CV_32F, 1 / 255.0);
    cv::resize(input_image, input_image, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);
    tensorflowDetector.setInput(cv::dnn::blobFromImage(input_image, 1.0, cv::Size(224, 224), 0.0, true, false, CV_32F));
    tensorflowDetector.forward(outputblobs, {"functional_1/box_output/Sigmoid", "functional_1/label_output/Softmax"});
    cv::Mat box = outputblobs.at(0).at(0);
    cv::Mat probs = outputblobs.at(1).at(0);
    
    int h = display_image.rows;
    int w = display_image.cols;

    // Print what we see
    std::cout << box << std::endl;
    std::cout << probs << std::endl;
    
    // Box Image dims
    double startYProb = box.at<float>(0);
    double endYProb = box.at<float>(1);
    double startXProb = box.at<float>(2);
    double endXProb = box.at<float>(3);
    
    int startX = static_cast<int>(startXProb * static_cast<double>(w));
    int startY = static_cast<int>(startYProb * static_cast<double>(h));
    int endX = static_cast<int>(endXProb * static_cast<double>(w));
    int endY =  static_cast<int>(endYProb * static_cast<double>(h));

    // Argmax of probs
    double minVal; 
    double maxVal; 
    cv::Point minLoc; 
    cv::Point maxLoc;

    cv::minMaxLoc(probs, &minVal, &maxVal, &minLoc, &maxLoc );
    Utility::drawPred(maxVal, startX, startY, endX, endY, display_image, "test");
    cv::imshow("CSI Camera", display_image);
*/



