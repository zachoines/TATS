#pragma once
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <vector>

#include "opencv2/opencv.hpp"
#include "opencv4/opencv2/dnn/dnn.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/core/core.hpp"

namespace Detect {
	struct DetectionData
	{
		double confidence;
		bool found;
		int label;
		std::string target;
		cv::Point2d center;
		cv::Rect2d boundingBox;
	};
	class ObjectDetector
	{
	protected:
		std::vector<std::string> class_names;
		
	public:
		ObjectDetector();
		virtual std::vector<struct DetectionData> detect(cv::Mat& image) = 0;
		virtual std::vector<struct DetectionData> detect(cv::Mat& image, bool draw, int numDrawn) = 0;
	};
}

