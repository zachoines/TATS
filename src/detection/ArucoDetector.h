#pragma once
#include "ObjectDetector.h"
#include "opencv2/highgui.hpp" 
#include "opencv2/imgproc.hpp" 
#include "opencv2/aruco.hpp"

namespace Detect {
	class ArucoDetector : public ObjectDetector
	{
	public:
		ArucoDetector(int dict = cv::aruco::DICT_6X6_250);
		std::vector<struct DetectionData> detect(cv::Mat& image);
		std::vector<struct DetectionData> detect(cv::Mat& image, bool draw, int numDrawn);
	private:
        std::vector<int> _ids;
        std::vector<std::vector<cv::Point2f>> _allCorners;
        cv::Ptr<cv::aruco::Dictionary> _dictionary;
	};
};

