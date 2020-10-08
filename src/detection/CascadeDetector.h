#pragma once
#include "ObjectDetector.h"
#include "opencv2/highgui.hpp" 
#include "opencv2/imgproc.hpp" 

namespace TATS {
	class CascadeDetector : public ObjectDetector
	{
	public:
		CascadeDetector(std::string path);
		struct DetectionData detect(cv::Mat& src, std::string target);
		struct DetectionData detect(cv::Mat& src, std::string target, bool draw);
	private:
		float scale = 1.;
		cv::CascadeClassifier cascade;
	};
};

