#pragma once
#include "ObjectDetector.h"
#include "opencv2/highgui.hpp" 
#include "opencv2/imgproc.hpp" 

namespace Detect {
	class CascadeDetector : public ObjectDetector
	{
	public:
		CascadeDetector(std::string path);
		struct DetectionData detect(cv::Mat& image);
		struct DetectionData detect(cv::Mat& image, bool draw);
		void setInputColor(int code);
	private:
		int input_color = 1; // 0 is color; 1 is gray
		float scale = 1.;
		cv::CascadeClassifier cascade;
	};
};

