#pragma once
#include "ObjectDetector.h"

#include <torch/torch.h>
#include <torch/script.h>

namespace Detect {
	/* 
		Object detector based on the RCNN Architecture. Loads any Pytorch JIT model.  
	
		//// Example usage
		Detect::RCNNDetector rd(saved_model_file_path);
		rd.setTargetLabel(1); // Whatever object you are looking for
		cv::Mat input_image = cv::imread(imageFilePath, cv::IMREAD_UNCHANGED);
		td::vector<struct DetectionData> results = rd.detect(input_image, true, 1); 
	*/
	class RCNNDetector : public ObjectDetector
	{
	private:
        int num_rects = 100;
		int max_detections = 5;
        int targetLabel = 0;
        torch::jit::script::Module module;
        torch::Device device = torch::kCPU;

	public:
		RCNNDetector(std::string path);
		std::vector<struct DetectionData> detect(cv::Mat& image);
		std::vector<struct DetectionData> detect(cv::Mat& image, bool draw, int numDrawn);
        void setTargetLabel(int label);
	};
}

