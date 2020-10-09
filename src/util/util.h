#include <iostream>
#include <fstream>

namespace Utility {

    static int mapOutput(int x, int in_min, int in_max, int out_min, int out_max) {
		return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
	}

	static double mapOutput(double x, double in_min, double in_max, double out_min, double out_max) {
		return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
	}

    static int calcTicks(float impulseMs, int hertz = 50, int pwm = 4096)
	{
		float cycleMs = 1000.0f / hertz;
		return (int)(pwm * impulseMs / cycleMs + 0.5f);
	}

	static int angleToTicks(double angle, int Hz = 50, double minAngle = -150.0, double maxAngle = 150.0, double minMs = 0.5, double maxMs = 2.5) {
		double millis = Utility::mapOutput(angle, minAngle, maxAngle, minMs, maxMs);
		int tick = calcTicks(millis, Hz);
	}

    static bool fileExists(std::string fileName){
		std::ifstream test(fileName);
		return (test) ? true : false;
	}

    // Draw the predicted bounding box
	static void drawPred(float conf, int left, int top, int right, int bottom, cv::Mat& frame)
	{
		using namespace cv;
		rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255));
		std::string label = cv::format("%.2f", conf);
		int baseLine;
		cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = cv::max(top, labelSize.height);
		putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
	}
}