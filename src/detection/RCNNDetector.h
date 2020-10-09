#pragma once
#include "ObjectDetector.h"

#include <torch/torch.h>
#include <torch/script.h>

namespace Detect {
	class RCNNDetector : public ObjectDetector
	{
	private:
        int num_rects = 100;
        int targetLabel = 0;
        torch::jit::script::Module module;
        torch::Device device = torch::kCPU;

	public:
		RCNNDetector(std::string path);
		struct DetectionData detect(cv::Mat& image);
		struct DetectionData detect(cv::Mat& image, bool draw);
        void setTargetLabel(int label);
	};
}
