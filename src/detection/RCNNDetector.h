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
		Detect::DetectionData result = rd.detect(input_image); 
	*/
	class RCNNDetector : public ObjectDetector
	{
	private:
        int num_rects = 10;
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

