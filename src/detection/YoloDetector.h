#pragma once
#include "ObjectDetector.h"

#include <string>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <memory>
#include "torch/script.h"
#include "torch/torch.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "../util/util.h"


namespace Detect { 
	/* 
		Object detector based on Yolo5 backend. Loads Yolo5 JIT model.  

		See https://github.com/ultralytics/yolov5 to train your own model!
    	Thanks to https://github.com/yasenh/libtorch-yolov5/ for libtorch implementation!
	
	*/

	enum Det {
        tl_x = 0,
        tl_y = 1,
        br_x = 2,
        br_y = 3,
        score = 4,
        class_idx = 5
    };

	class YoloDetector : public ObjectDetector
	{
	public:
		YoloDetector(std::string path, std::vector<std::string> classNames);
		~YoloDetector();
		
        std::vector<DetectionData> detect(cv::Mat& image);
		std::vector<DetectionData> detect(cv::Mat& image, bool draw, int numDrawn);

        void setTargets(std::vector<int> targetLabels);
		void setConfidence(float conf_threshold);
		void setIOU(float iou_threshold);

	private:
		std::vector<std::string> class_names;
        std::vector<int> targetLabels;
		
		torch::jit::script::Module module_;
        torch::Device device_ = torch::kCPU;
        
		bool half_;
		float confidence;
		float conf_threshold = 0.4;
		float iou_threshold = 0.6;
		int item_attr_size = 5;

        void loadModule(const std::string& model_path);

        std::vector<std::vector<DetectionData>> run(const cv::Mat& img);

        std::vector<float> LetterboxImage(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size = cv::Size(640, 640));

        std::vector<std::vector<DetectionData>> PostProcessing(const torch::Tensor& detections, float pad_w, float pad_h, float scale, const cv::Size& img_shape, float conf_thres, float iou_thres);

        void ScaleCoordinates(std::vector<DetectionData>& data,float pad_w, float pad_h, float scale, const cv::Size& img_shape);

        torch::Tensor xywh2xyxy(const torch::Tensor& x);

        void Tensor2Detection(const at::TensorAccessor<float, 2>& offset_boxes, const at::TensorAccessor<float, 2>& det, std::vector<cv::Rect>& offset_box_vec, std::vector<float>& score_vec);
	};
};