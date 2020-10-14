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
		std::vector<struct DetectionData> detect(cv::Mat& image);
		std::vector<struct DetectionData> detect(cv::Mat& image, bool draw, int numDrawn);

		/***
         * @brief set the indices of target to be returned
         * @param targetLabels - Desired class indexes
		 */	
        void setTargets(std::vector<int> targetLabels);

		/***
         * @brief set confidence threshold
         * @param conf_threshold - confidence threshold
		 */
		void setConfidence(float conf_threshold);

		/***
         * @brief set Intersection over Union threshold
         * @param iou_threshold - IoU threshold for nms
		 */		
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

		/***
         * @brief loads model from disk
         * @param model_path - path of the TorchScript weight file
         */
        void loadModule(const std::string& model_path);

        /***
         * @brief inference module
         * @param img - input image
         * @param conf_threshold - confidence threshold
         * @param iou_threshold - IoU threshold for nms
         * @return detection result - bounding box, score, class index
         */
        std::vector<struct DetectionData> run(const cv::Mat& img);

        /***
         * @brief Padded resize
         * @param src - input image
         * @param dst - output image
         * @param out_size - desired output size
         * @return padding information - pad width, pad height and zoom scale
         */
        std::vector<float> LetterboxImage(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size = cv::Size(640, 640));

        torch::Tensor GetBoundingBoxIoU(const torch::Tensor& box1, const torch::Tensor& box2);

        /***
         * @brief Performs Non-Maximum Suppression (NMS) on inference results
         * @param detections - inference results from the network, example [1, 25200, 85], 85 = 4(xywh) + 1(obj conf) + 80(class score)
         * @return detections with shape: nx7 (batch_index, x1, y1, x2, y2, score, classification)
         */
        torch::Tensor PostProcessing(const torch::Tensor& detections);

        /***
         * @brief Rescale coordinates to original input image
         * @param data - detection result after inference and nms
         * @param pad_w - width padding
         * @param pad_h - height padding
         * @param scale - zoom scale
         * @param img_shape - original input image shape
         * @return rescaled detections
         */
        std::vector<struct DetectionData> ScaleCoordinates(const at::TensorAccessor<float, 2>& data, float pad_w, float pad_h, float scale, const cv::Size& img_shape);
	};
};