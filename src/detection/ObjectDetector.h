#pragma once
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <vector>

#include "opencv2/opencv.hpp"
#include "opencv4/opencv2/dnn/dnn.hpp"
#include "opencv2/core/core.hpp"

namespace Detect {
	struct DetectionData
	{
		int targetCenterX;
		int targetCenterY;
		double confidence;
		bool found;
		std::string target;
		int label;
		cv::Rect boundingBox;
	};
	class ObjectDetector
	{
	protected:
		std::vector<std::string> class_names;
		cv::Mat* last_frame = nullptr;
		
	public:
		ObjectDetector();
		virtual struct DetectionData detect(cv::Mat& image) = 0;
	};
}

