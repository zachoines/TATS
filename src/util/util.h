#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <chrono>

#include <unistd.h>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/tracking.hpp"

namespace Utility {


	static void msleep(int milli) {
		std::this_thread::sleep_for(std::chrono::milliseconds(milli));
	}

    static int mapOutput(int x, int in_min, int in_max, int out_min, int out_max) {
		return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
	}

	static double mapOutput(double x, double in_min, double in_max, double out_min, double out_max) {
		return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
	}

	static double normalize(double x, double min, double max) {
		return (x - min) / (max - min);
	}
   
    static bool fileExists(std::string fileName){
		std::ifstream test(fileName);
		return (test) ? true : false;
	}

    // Draw the predicted bounding box
	static void drawPred(float conf, int left, int top, int right, int bottom, cv::Mat& frame, std::string text = "")
	{
		using namespace cv;
		rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255));
		std::string label = cv::format("%.2f, %s", conf, text.c_str());
		int baseLine;
		cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = cv::max(top, labelSize.height);
		cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
	}

	// Get a frame from the camera
	static cv::Mat GetImageFromCamera(cv::VideoCapture* camera)
	{
		cv::Mat frame;
		*camera >> frame;
		return frame;
	}

	// Create GStreamer camera config
	static std::string gstreamer_pipeline (int camera, int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
		return "nvarguscamerasrc sensor-id=" + std::to_string(camera) + " ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
			std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) +
			"/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
			std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
	}

	static void appendLineToFile(std::string filepath, std::string line)
	{
		std::ofstream file; 
		file.open(filepath, std::ios::out | std::ios::app);
		if (file.fail())
			throw std::ios_base::failure(std::strerror(errno));

		file.exceptions(file.exceptions() | std::ios::failbit | std::ifstream::badbit);

		file << line << std::endl;
		file.close();
	}

	static cv::Ptr<cv::Tracker> createOpenCVTracker(int type) {
		cv::Ptr<cv::Tracker> tracker;
		switch (type)
		{
		case 0:
			tracker = cv::TrackerCSRT::create();
			break;
		case 1:
			tracker = cv::TrackerMOSSE::create();
			break;
		case 2:
			tracker = cv::TrackerGOTURN::create();
			break;
		default:
			tracker = cv::TrackerCSRT::create();
			break;
		}
		
		return tracker;
	}

	/* 
		####################################
		## When alt 'true' #################
		## Reward bounds are -2.0 to +2.0 ##
		####################################
		
		Basline error/rewards between -0.5 and 0.5 baased on total current error

		+0.0 bonus within { threshold }% error
		0.0 to +1.0 reward for marginally better transitions

		-2.0 punishment for done state
		0.0 to -1.0 punishment for marginally worse transitions
	
		######################################
		## When alt 'false' ##################
		## Reward bounds from -3.0 to 0.0 ####
		######################################
		
		-1.0 for done state
		0.0 to -1.0 - { threshold } depending on current error
		0.0 to -1.0 for marginally worse transitions
		0.0 reward for any improvement

		#####################
		## key differences ##
		#####################
		
		* The first incorperates positive rewards
		* The first will show a nicer episode average graph 
			* Episode average rewards will increase exponentially. 
			* Step rewards will be noisy and less clear improvement
		
		* The first only has negative rewards
		* The second will show a nicer step rewards graph 
			* Step error will decrease logorithmically. 
			* Episode average rewards will decrease exponentially (accumulation of error)
		
		* Both will increase episode steps exponentially
		* They both have similar loss curves 
		* Both teach fundamentally similar skills 
			* To favor trasitions that lead to lower error

	*/

	static double pidErrorToReward(double n, double o, double max, bool d, double threshold = 0.005, bool alt = true) {


		double absMax = max;
		double errorThreshold = threshold;
		bool done = d;
		double center = absMax;

		// Relative weight to rewards/error
		double w1 = 1.0;
		double w2 = 1.0; 
		double w3 = 1.0; 

		// Rewards
		double r1 = 0.0; // Threshhold bonus, alive bonus, done rewards/error
		double r2 = 0.0; // baseline error
		double r3 = 0.0; // Transition rewards/error

		// scale from 0.0 to 1.0
		double errorOldScaled = std::fabs(center - std::fabs(o)) / center;
		double errorNewScaled = std::fabs(center - std::fabs(n)) / center;

		// If marginal transitional rewards are considered
		if (alt) {

			if (done) {
				r1 = -1.0;
			} else if (errorNewScaled <= errorThreshold) {
				r1 = 0.0;
			}
			                                 
			// r2 += Utility::mapOutput(1.0 - errorNewScaled, 0.0, 1.0, -0.50, 0.50);
			// r2 += errorNewScaled;

			// Rewards R3
			bool direction_old = false;
			bool direction_new = false;

			double targetCenterNew = n;
			double targetCenterOld = o;

			if (targetCenterNew == targetCenterOld) {
				r3 = 0.0;
			} else {

				// The target in ref to the center of frame. Left is F, right is T.
				if (targetCenterNew < center) { // target is left of frame center
					direction_new = false;
				}
				else { // target is right of frame center
					direction_new = true;
				}

				if (targetCenterOld < center) { // target is left of frame center
					direction_old = false;
				}
				else { // target is right of frame center
					direction_old = true;
				}

				//  Both to the right of frame center, situation #1;
				if (direction_old && direction_new) {

					double reward = std::fabs(errorNewScaled - errorOldScaled);

					if (targetCenterNew > targetCenterOld) { // frame center has moved furthure to object's left
						r3 = -reward;
					}
					else { // frame center has moved closer to object's left
						r3 = reward;
					}
				}

				// both to left of frame center, situation #2
				else if (!direction_old && !direction_new) {

					double reward = std::fabs(errorOldScaled - errorNewScaled);

					if (targetCenterNew > targetCenterOld) {  // frame center has moved closer to objects right
						r3 = reward;
					}
					else { // frame center has moved further from objects right
						r3 = -reward;
					}

				}

				// Frame center has overshot target. Old is oposite sides of center to new, situation #3
				else  { 

					double error_old_corrected = std::fabs(std::fabs(targetCenterOld) - center);
					double error_new_corrected = std::fabs(std::fabs(targetCenterNew) - center);
					double difference = std::fabs(error_new_corrected - error_old_corrected);
					double reward = difference / center;

					if (error_old_corrected > error_new_corrected) {  // If move has resulted in a marginally lower error (closer to center)
						r3 = reward;
					}
					else {
						r3 = -reward;
					}
				}
			}

			return ((w1 * r1) + (w2 * r2) + (w3 * r3));
	
		} else {
			
	        // Another varient with only negative rewards
			// if (errorNewScaled < errorThreshold) {
			// 	r1 = 0.0;
			// } else {
			r1 = - errorNewScaled;
			// // }

			// if (errorNewScaled <= errorOldScaled) {
			//  	r2 = 0.0;
			// }
			// else {
			//  	r2 = errorNewScaled - errorOldScaled;
			// }
			
			// Punish done, scale and clamp from -1 to 1
			// return std::clamp<double>((w1 * r1) + (w2 * r2) + ( done ? -1.0 : 0.0), -1.0, 0.0);	
			return std::clamp<double>((w1 * r1) + ( done ? -1.0 : 0.0), -2.0, 0.0);	
		}	
	}

	static double predictedObjectLocationToReward(double pred, double target, double max, bool done) {
		return (done) ? 0.0 : - std::clamp<double>(2.0 * (std::fabs(target - pred) / max), 0.0, 1.0);
	}

	// Scale from -1.0 to 1.0 to low to high
	static double rescaleAction(double action, double min, double max) {
		return std::clamp<double>((min + (0.5 * (action + 1.0) * (max - min))), min, max);
	}
}