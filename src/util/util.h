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

	/***
	 * @brief Rounds to the nearest tenth 
	 * @param value Number to round
	 * @return rounded value
	 */
	static double roundToNearestTenth(double value){
		return double(int(value * 10 + 0.5)) / 10;
	}

	
	/***
	 * @brief Returns true if object is headinig right/up, and false left/down 
	 * @param error Current difference between object center and frame center
	 * @param inverted Reverses direction
	 * @return boolean
	 */
	static bool calculateDirectionOfObject(double error, bool inverted) {
		bool direction = error < 0.0;
		return inverted ? !direction : direction;
	}

	/***
	 * @brief Sleep for specified milliseconds 
	 * @param milli Number of milliseconds to sleep for
	 * @return void
	 */
	static void msleep(int milli) {
		std::this_thread::sleep_for(std::chrono::milliseconds(milli));
	}

	/***
	 * @brief Maps x from [in-range min, in-range max] to [out-range min, out-range max]
	 * @param x Input value
	 * @param in_min Input minimum
	 * @param in_max Input maximum
	 * @param out_min Ouput minimim
	 * @param out_max Output maximum
	 * @return integer
	 */
    static int mapOutput(int x, int in_min, int in_max, int out_min, int out_max) {
		return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
	}

	/***
	 * @brief Maps x from [in-range min, in-range max] to [out-range min, out-range max]
	 * @param x Input value
	 * @param in_min Input minimum
	 * @param in_max Input maximum
	 * @param out_min Ouput minimim
	 * @param out_max Output maximum
	 * @return double
	 */
	static double mapOutput(double x, double in_min, double in_max, double out_min, double out_max) {
		return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
	}

	/***
	 * @brief normalizes x from 0.0 to 1.0
	 * @param x Input value
	 * @param min X's minimum
	 * @param max X's maximum
	 * @return double
	 */
	static double normalize(double x, double min, double max) {
		return (x - min) / (max - min);
	}
   
	/***
	 * @brief Determines if a file exists at provided dir
	 * @param filename Full path of file to test
	 * @return boolean
	 */
    static bool fileExists(std::string fileName){
		std::ifstream test(fileName);
		return (test) ? true : false;
	}

	/***
	 * @brief Draw the predicted bounding box with other information
	 * @param conf Confidence value
	 * @param left Top left point (x)
	 * @param top Top left point (y)
	 * @param right Botton right point (x)
	 * @param bottom Botton right point (y)
	 * @param frame Input image to write on
	 * @param text Additional text to put on frame
	 * @return void
	 */
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

	/***
	 * @brief get image frome the camera
	 * @param camera Pointer to camera object
	 * @return cv::Mat image
	 */
	static cv::Mat GetImageFromCamera(cv::VideoCapture* camera)
	{
		cv::Mat frame;
		*camera >> frame;
		return frame;
	}

	/***
	 * @brief Create GStreamer camera config
	 * @param camera Camera number for g_streamer
	 * @param capture_width Width in pixels for capture
	 * @param capture_height Height in pixels capture
	 * @param display_width Width to scale image to 
	 * @param display_height Height to scale image to
	 * @param framerate Capture framerate
	 * @param flip_method Amount to flip image 0 - 3
	 * @return cv::Mat image
	 */
	static std::string gstreamer_pipeline (int camera, int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
		return "nvarguscamerasrc sensor-id=" + std::to_string(camera) + " ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
			std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) +
			"/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
			std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
	}

	/***
	 * @brief Add line of text to end of file
	 * @param filepath Input file to write line too
	 * @param line Line to write to file
	 * @return void
	 */
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

	/***
	 * @brief generate opencv tracker object
	 * @param type Type of opencv tracker
	 * @return Pointer to tracker object
	 */
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
		## Reward bounds are -2.0 to +1.0 ##
		####################################
		
		Done state: 
			-2.0
		Basline rewards: 
			-0.5 <--> 0.5
		Transition bonus:
			marginally better: 0.0 <--> 0.5
			marginally worse: -0.5 <--> 0.0
	
		######################################
		## When alt 'false' ##################
		## Reward bounds from -2.0 to 0.0 ####
		######################################
		
		Done state: 
			-2.0
		Basline rewards:
			if withing threshold: 0.0 
			otherwise: -0.5 <--> 0.0
		Transition bonus:
			marginally worse: -.50 <--> 0.0
			marginally better: 0.0

		#####################
		## key differences ##
		#####################
		
		* The first incorperates positive rewards		
		* The second only has negative rewards
		* They both have similar loss curves 
		* Both teach fundamentally similar skills -- to favor trasitions that lead to lower error
		* Different average vs. step reward graphs

	*/
	static double pidErrorToReward(int newError, int oldError, int center, bool done, double threshold = 0.05, bool alt = true) {
		
		// Rewards
		double r1 = 0.0; // baseline error
		double r2 = 0.0; // transition error
		double w1 = 0.5; // baseline error weight
		double w2 = 0.5; // transition error weight
		
		// Indicate target direction
		const bool RIGHT = true;
		const bool LEFT = false;

		// scale from 0.0 to 1.0
		double errorOldScaled = static_cast<double>(std::abs(center - oldError)) / static_cast<double>(center);
		double errorNewScaled = static_cast<double>(std::abs(center - newError)) / static_cast<double>(center);

		if (alt) {

			// Baseline rewards
			if (done) {
				return -2.0;
			} else {
				r1 = Utility::mapOutput(1.0 - errorNewScaled, 0.0, 1.0, -1.0, 1.0);
			}                            
			

			// Transition rewards
			bool direction_old = false;
			bool direction_new = false;

			double targetCenterNew = newError;
			double targetCenterOld = oldError;

			if (targetCenterNew == targetCenterOld) {
				r2 = 0.0;
			} else {

				// Determine target deirection in ref to the center of frame. Left is F, right is T.
				targetCenterNew < center ? direction_new = LEFT : direction_new = RIGHT;
				targetCenterOld < center ? direction_old = LEFT : direction_old = RIGHT;

				//  Both to the right of frame center, situation #1;
				if (direction_old == RIGHT && direction_new == RIGHT) {

					double reward = std::fabs(errorNewScaled - errorOldScaled);

					if (targetCenterNew > targetCenterOld) { // frame center has moved furthure to object's left
						r2 = -reward;
					}
					else { // frame center has moved closer to object's left
						r2 = reward;
					}
				}

				// both to left of frame center, situation #2
				else if (direction_old == LEFT && direction_new == LEFT) {

					double reward = std::fabs(errorOldScaled - errorNewScaled);

					if (targetCenterNew > targetCenterOld) {  // frame center has moved closer to objects right
						r2 = reward;
					}
					else { // frame center has moved further from objects right
						r2 = -reward;
					}

				}

				// Frame center has overshot target. Old is oposite sides of center to new, situation #3
				else  { 

					double error_old_corrected = std::fabs(std::fabs(targetCenterOld) - center);
					double error_new_corrected = std::fabs(std::fabs(targetCenterNew) - center);
					double difference = std::fabs(error_new_corrected - error_old_corrected);
					double reward = difference / center;

					if (error_old_corrected > error_new_corrected) {  // If move has resulted in a marginally lower error (closer to center)
						r2 = reward;
					}
					else {
						r2 = -reward;
					}
				}
			}

			return std::clamp<double>((w1 * r1) + (w2 * r2), -1.0, 1.0);
	
		} else {
			if (errorNewScaled <= threshold) {
				r1 = 0.0;
			} else {
				r1 = - errorNewScaled;
			}
			
			if (errorNewScaled <= errorOldScaled) {
				r2 = 0.0;
			}
			else {
				r2 = errorNewScaled - errorOldScaled;
			}
			
			return done ? -2.0 : ((r1 * w1) + (r2 * w2));	
	       
		}	
	}

	/* 
		####################################
		## When alt 'true' #################
		## Reward bounds are -1.0 to +0.5 ##
		####################################
		
		Done state: 
			-1.0
		Rewards: 
			-0.5 <--> 0.5
	
		######################################
		## When alt 'false' ##################
		## Reward bounds from -1.0 to 0.0 ####
		######################################
		
		Done state: 
			-1.0
		Rewards:			
			otherwise: -1.0 <--> 0.0


	*/
	static double predictedObjectLocationToReward(int pred, int target, int max, bool done, double threshold = 0.05, bool alt = true) {

		if (done) {
			return -1.0;
		}

		if (alt) {
			return Utility::mapOutput(1.0 - (std::abs(target - pred) / static_cast<double>(max)), 0.0, 1.0, -.5, .5);
		} else {
			double error = 1.0 - (static_cast<double>(std::abs(target - pred)) / static_cast<double>(max));

			if (error <= threshold) {
				return 0.0;
			} else {
				return error;
			}
		}
	}

	/***
	 * @brief Scale from -1.0 to 1.0 to low to high
	 * @param action Input to be scaled
	 * @param min Min of input
	 * @param max Max of input
	 * @return double
	 */
	static double rescaleAction(double action, double min, double max) {
		return std::clamp<double>((min + (0.5 * (action + 1.0) * (max - min))), min, max);
	}
}