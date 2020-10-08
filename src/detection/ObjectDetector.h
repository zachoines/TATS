#pragma once
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <vector>

#include "opencv2/opencv.hpp"
#include "opencv4/opencv2/dnn/dnn.hpp"
#include "opencv2/core/core.hpp"

namespace TATS {
	struct Rect {
		int x;
		int y;
		int height;
		int width;
	};
	struct DetectionData
	{
		int targetCenterX;
		int targetCenterY;
		double confidence;
		bool found;
		std::string target;
		struct Rect boundingBox;
	};
	class ObjectDetector
	{
	protected:
		std::vector<std::string> class_names;
		cv::Mat* last_frame = nullptr;
		cv::dnn::Net* net = nullptr;
		int input_color = 1; // 0 is color; 1 is gray

	public:
		ObjectDetector();
		virtual struct DetectionData detect(cv::Mat& src, std::string target) = 0;
		void setInputColor(int code);
	};
}

